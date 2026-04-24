"""Generate synthetic corruption data for repair-SFT (Step 2).

For each GT sketch+extrude program, creates 2 corrupted versions:
  type1: sketch+extrude -> box() with exact bbox match
  type2: sketch+extrude -> box() with mild extent distortion (±15%)

Output: JSONL at data/repair_sft/repair_pairs.jsonl
  Each record: {stem, corruption_type, gt_code, corrupt_code, gt_render, corrupt_render}

GT programs: high-IoU (≥0.95) cadrille-rl predictions from deepcad_rl_img
  that use sketch+extrude but not box(). These are the 'correct' codes.

Usage
-----
  python3 tools/gen_repair_data.py --n 200 --out data/repair_sft
  python3 tools/gen_repair_data.py --n 200 --out data/repair_sft --dry-run
"""

import argparse
import ast
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
from rl.dataset import render_img

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# GT program extraction helpers
# ---------------------------------------------------------------------------

def has_sketch_no_box(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    has_sketch = has_box = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            if n.attr == 'sketch':
                has_sketch = True
            if n.attr == 'box':
                has_box = True
    return has_sketch and not has_box


# ---------------------------------------------------------------------------
# Mesh execution + bbox extraction
# ---------------------------------------------------------------------------

_BBOX_WORKER = textwrap.dedent('''\
    import sys, json, warnings, signal
    import numpy as np
    import trimesh
    import cadquery as cq  # noqa

    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(36)

    try:
        code = json.loads(sys.stdin.read())["code"]
        try:
            code_obj = compile(code, "<string>", "exec")
        except SyntaxError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            sys.exit(0)

        g = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code_obj, g)
        compound = g["r"].val()
        bb = compound.BoundingBox()
        signal.alarm(0)
        print(json.dumps({
            "ok": True,
            "xmin": bb.xmin, "xmax": bb.xmax,
            "ymin": bb.ymin, "ymax": bb.ymax,
            "zmin": bb.zmin, "zmax": bb.zmax,
        }))
    except TimeoutError:
        print(json.dumps({"ok": False, "error": "timeout"}))
    except Exception as e:
        signal.alarm(0)
        print(json.dumps({"ok": False, "error": str(e)[:200]}))
    sys.stdout.flush()
''')

_bbox_worker_path: str | None = None


def _get_bbox_worker() -> str:
    global _bbox_worker_path
    if _bbox_worker_path and os.path.exists(_bbox_worker_path):
        return _bbox_worker_path
    fd, p = tempfile.mkstemp(suffix='.py', prefix='bbox_worker_')
    with os.fdopen(fd, 'w') as f:
        f.write(_BBOX_WORKER)
    _bbox_worker_path = p
    return p


def get_bbox(code: str) -> dict | None:
    payload = json.dumps({'code': code})
    try:
        proc = subprocess.run(
            [sys.executable, _get_bbox_worker()],
            input=payload, capture_output=True, text=True, timeout=42,
            env={**os.environ, 'LD_LIBRARY_PATH': '/workspace/.local/lib'})
        if proc.stdout.strip():
            r = json.loads(proc.stdout.strip())
            return r if r.get('ok') else None
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Corruption generators
# ---------------------------------------------------------------------------

def _box_code(bb: dict, distort: float = 0.0) -> str:
    """Generate box() code matching the bbox, with optional ±distort fraction."""
    cx = (bb['xmin'] + bb['xmax']) / 2
    cy = (bb['ymin'] + bb['ymax']) / 2
    cz = (bb['zmin'] + bb['zmax']) / 2
    dx = bb['xmax'] - bb['xmin']
    dy = bb['ymax'] - bb['ymin']
    dz = bb['zmax'] - bb['zmin']
    if distort > 0:
        rng = np.random.uniform(1 - distort, 1 + distort, 3)
        dx, dy, dz = dx * rng[0], dy * rng[1], dz * rng[2]
    dx = max(dx, 0.01)
    dy = max(dy, 0.01)
    dz = max(dz, 0.01)
    return (
        f"import cadquery as cq\n"
        f"r = cq.Workplane('XY').workplane(offset={cz - dz/2:.4f})"
        f".moveTo({cx:.4f},{cy:.4f}).box({dx:.4f},{dy:.4f},{dz:.4f})"
    )


def make_corruptions(code: str, bb: dict) -> dict[str, str]:
    return {
        'type1': _box_code(bb, distort=0.0),
        'type2': _box_code(bb, distort=0.15),
    }


# ---------------------------------------------------------------------------
# Render corrupted code to PNG
# ---------------------------------------------------------------------------

_EXEC_WORKER = textwrap.dedent('''\
    import sys, json, io, warnings, signal
    import trimesh
    import cadquery as cq  # noqa

    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(36)

    try:
        p = json.loads(sys.stdin.read())
        code, out_stl = p["code"], p["out_stl"]
        try:
            code_obj = compile(code, "<string>", "exec")
        except SyntaxError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            sys.exit(0)
        g = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code_obj, g)
        compound = g["r"].val()
        verts, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        buf = trimesh.exchange.stl.export_stl(mesh)
        with open(out_stl, "wb") as f:
            f.write(buf)
        signal.alarm(0)
        print(json.dumps({"ok": True}))
    except TimeoutError:
        print(json.dumps({"ok": False, "error": "timeout"}))
    except Exception as e:
        signal.alarm(0)
        print(json.dumps({"ok": False, "error": str(e)[:200]}))
    sys.stdout.flush()
''')

_exec_worker_path: str | None = None


def _get_exec_worker() -> str:
    global _exec_worker_path
    if _exec_worker_path and os.path.exists(_exec_worker_path):
        return _exec_worker_path
    fd, p = tempfile.mkstemp(suffix='.py', prefix='exec_worker_')
    with os.fdopen(fd, 'w') as f:
        f.write(_EXEC_WORKER)
    _exec_worker_path = p
    return p


def execute_code_to_stl(code: str, out_stl: str) -> bool:
    """Execute CadQuery code in subprocess, save mesh to STL."""
    payload = json.dumps({'code': code, 'out_stl': out_stl})
    try:
        proc = subprocess.run(
            [sys.executable, _get_exec_worker()],
            input=payload, capture_output=True, text=True, timeout=42,
            env={**os.environ, 'LD_LIBRARY_PATH': '/workspace/.local/lib'})
        if proc.stdout.strip():
            r = json.loads(proc.stdout.strip())
            return r.get('ok', False)
        return False
    except Exception:
        return False


def render_stl_to_png(stl_path: str, out_png: str) -> bool:
    """Render STL to 4-view PNG using render_img (open3d headless build)."""
    try:
        item = render_img(stl_path)
        item['video'][0].save(out_png)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n',            type=int, default=200,
                        help='Number of GT programs to use (default: 200)')
    parser.add_argument('--out',          default='data/repair_sft')
    parser.add_argument('--analysis-dir', default='data/analysis/deepcad_rl_img')
    parser.add_argument('--gt-render-dir',default='data/deepcad_test_mesh')
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--dry-run',      action='store_true')
    args = parser.parse_args()

    analysis_dir  = _REPO / args.analysis_dir
    gt_render_dir = _REPO / args.gt_render_dir
    out_dir       = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'corrupt_renders').mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect GT programs
    # ------------------------------------------------------------------
    print('Scanning for high-IoU sketch+extrude programs...')
    with open(analysis_dir / 'metadata.jsonl') as f:
        meta = [json.loads(l) for l in f]

    candidates = []
    for row in meta:
        if row.get('iou') is None or float(row['iou']) < 0.95:
            continue
        py = analysis_dir / f"{row['case_id']}_pred.py"
        gt_render = gt_render_dir / f"{row['case_id']}_render.png"
        if not (py.exists() and gt_render.exists()):
            continue
        code = py.read_text()
        if has_sketch_no_box(code):
            candidates.append({'stem': row['case_id'], 'iou': float(row['iou']),
                                'code': code, 'gt_render': str(gt_render)})

    random.shuffle(candidates)
    selected = candidates[:args.n]
    print(f'  {len(candidates)} candidates → using {len(selected)}')

    if args.dry_run:
        print('Dry run — exiting.')
        return

    # ------------------------------------------------------------------
    # 2. Get bboxes
    # ------------------------------------------------------------------
    print(f'\nExtracting bboxes ({args.workers} workers)...')
    bboxes: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(get_bbox, c['code']): c for c in selected}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='bbox'):
            c = futs[fut]
            try:
                bb = fut.result()
                if bb:
                    bboxes[c['stem']] = bb
            except Exception:
                pass
    print(f'  {len(bboxes)}/{len(selected)} bbox extractions succeeded')

    # ------------------------------------------------------------------
    # 3. Generate corruptions + render
    # ------------------------------------------------------------------
    print('\nGenerating corruptions and rendering...')
    pairs = []
    failed_render = 0

    # Step 1: execute corrupted codes → STLs (subprocess, parallel)
    stl_dir = out_dir / 'corrupt_stls'
    stl_dir.mkdir(exist_ok=True)

    def exec_one(c):
        stem = c['stem']
        bb = bboxes.get(stem)
        if not bb:
            return stem, {}
        corruptions = make_corruptions(c['code'], bb)
        stls = {}
        for ctype, corrupt_code in corruptions.items():
            out_stl = str(stl_dir / f'{stem}_{ctype}.stl')
            if execute_code_to_stl(corrupt_code, out_stl):
                stls[ctype] = (corrupt_code, out_stl)
        return stem, stls

    selected_with_bbox = [c for c in selected if c['stem'] in bboxes]
    stl_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(exec_one, c): c for c in selected_with_bbox}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='exec→stl'):
            try:
                stem, stls = fut.result()
                stl_results[stem] = stls
            except Exception:
                pass

    n_stls = sum(len(v) for v in stl_results.values())
    print(f'  {n_stls}/{len(selected_with_bbox)*2} STL executions succeeded')

    # Step 2: render STLs → PNGs in main process (open3d requires main process)
    print('Rendering STLs to PNGs (main process)...')
    corrupt_render_dir = out_dir / 'corrupt_renders'
    corrupt_render_dir.mkdir(exist_ok=True)

    stem_map = {c['stem']: c for c in selected_with_bbox}
    for stem, stls in tqdm(stl_results.items(), desc='render'):
        c = stem_map[stem]
        for ctype, (corrupt_code, stl_path) in stls.items():
            out_png = str(corrupt_render_dir / f'{stem}_{ctype}.png')
            if render_stl_to_png(stl_path, out_png):
                pairs.append({
                    'stem': stem,
                    'corruption_type': ctype,
                    'gt_code': c['code'],
                    'corrupt_code': corrupt_code,
                    'gt_render': c['gt_render'],
                    'corrupt_render': out_png,
                    'action': 'SWITCH_TO_SKETCH_EXTRUDE',
                })
            else:
                failed_render += 1

    print(f'  Generated {len(pairs)} pairs '
          f'({failed_render} render failures)')

    # ------------------------------------------------------------------
    # 4. Train/val split and save
    # ------------------------------------------------------------------
    random.shuffle(pairs)
    n_val = max(40, int(len(pairs) * 0.1))
    train_pairs = pairs[n_val:]
    val_pairs   = pairs[:n_val]

    train_path = out_dir / 'train.jsonl'
    val_path   = out_dir / 'val.jsonl'
    with open(train_path, 'w') as f:
        for p in train_pairs:
            f.write(json.dumps(p) + '\n')
    with open(val_path, 'w') as f:
        for p in val_pairs:
            f.write(json.dumps(p) + '\n')

    print(f'\nSaved:')
    print(f'  train: {len(train_pairs)} pairs → {train_path}')
    print(f'  val:   {len(val_pairs)} pairs → {val_path}')
    print(f'\nType breakdown:')
    from collections import Counter
    ct = Counter(p['corruption_type'] for p in pairs)
    for k, v in sorted(ct.items()):
        print(f'  {k}: {v}')
    print('\nDone. Next: run tools/train_repair_lora.py')


if __name__ == '__main__':
    main()
