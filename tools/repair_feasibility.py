"""Zero-shot repair feasibility test for action-conditioned CAD repair.

Validates whether Cadrille-RL can repair 'wrong_primitive' failures given
action-conditioned input.  Three ablation conditions (zero-shot, no fine-tuning):

  A: GT views + bad code                            (no pred render, no action)
  B: GT views + pred render + bad code              (no action)
  C: GT views + pred render + bad code + action     (full — our method)

Metrics per condition:
  mean / median ΔIoU, mean / median ΔCD,
  % cases with ΔIoU > 0.05, valid rate

Selection: wrong_primitive cases from deepcad_rl_img
  (pred code has box() call but no .sketch() call), IoU in [0.3, 0.88]

Usage
-----
  python3 tools/repair_feasibility.py --n 100
  python3 tools/repair_feasibility.py --n 100 --out data/repair_feasibility --dry-run
"""

import argparse
import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
from cadrille import Cadrille, collate


# ---------------------------------------------------------------------------
# Subprocess scorer — IoU + CD in one call
# ---------------------------------------------------------------------------

_SCORE_WORKER = textwrap.dedent('''\
    import sys, json, io, os, warnings, signal
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq  # noqa

    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(40)

    def _transform(m):
        m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
        ext = np.max(m.extents)
        if ext > 1e-7:
            m.apply_scale(2.0 / ext)
        return m

    def _cd(gt_mesh, pred_mesh, n=2048):
        try:
            gp, _ = trimesh.sample.sample_surface(gt_mesh, n)
            pp, _ = trimesh.sample.sample_surface(pred_mesh, n)
            d1, _ = cKDTree(gp).query(pp, k=1)
            d2, _ = cKDTree(pp).query(gp, k=1)
            return float(np.mean(d1**2) + np.mean(d2**2))
        except Exception:
            return None

    try:
        p = json.loads(sys.stdin.read())
        code, gt_path = p["code"], p["gt_path"]

        try:
            code_obj = compile(code, "<string>", "exec")
        except SyntaxError as e:
            signal.alarm(0)
            print(json.dumps({"iou": None, "cd": None,
                               "error_type": "syntax_error", "error_msg": str(e)[:200]}))
            sys.exit(0)

        try:
            g = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(code_obj, g)
            compound = g["r"].val()
            verts, faces = compound.tessellate(0.001, 0.1)
            pred = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
            assert len(pred.faces) > 2
            import io as _io
            buf = trimesh.exchange.stl.export_stl(pred)
            pred = trimesh.load(_io.BytesIO(buf), file_type="stl", force="mesh")
        except Exception as e:
            signal.alarm(0)
            print(json.dumps({"iou": None, "cd": None,
                               "error_type": "runtime_error", "error_msg": str(e)[:200]}))
            sys.exit(0)

        gt = trimesh.load_mesh(gt_path)

        # IoU
        try:
            p_t = _transform(pred.copy())
            g_t = _transform(gt.copy())
            iv = 0.0
            for gi in g_t.split():
                for pi in p_t.split():
                    s = gi.intersection(pi)
                    iv += s.volume if s is not None else 0.0
            gv = sum(m.volume for m in g_t.split())
            pv = sum(m.volume for m in p_t.split())
            uv = gv + pv - iv
            iou = float(iv / uv) if uv > 0 else 0.0
        except Exception as e:
            signal.alarm(0)
            print(json.dumps({"iou": None, "cd": None,
                               "error_type": "runtime_error", "error_msg": "iou:" + str(e)[:150]}))
            sys.exit(0)

        # CD (on normalised meshes)
        cd = _cd(_transform(gt.copy()), _transform(pred.copy()))

        signal.alarm(0)
        print(json.dumps({
            "iou": iou,
            "cd": cd,
            "error_type": "success" if iou > 0 else "zero_iou",
            "error_msg": None
        }))
    except TimeoutError:
        print(json.dumps({"iou": None, "cd": None, "error_type": "timeout", "error_msg": None}))
    except Exception as e:
        signal.alarm(0)
        print(json.dumps({"iou": None, "cd": None,
                           "error_type": "runtime_error", "error_msg": str(e)[:200]}))
    sys.stdout.flush()
''')

_worker_path: str | None = None


def _get_worker() -> str:
    global _worker_path
    if _worker_path and os.path.exists(_worker_path):
        return _worker_path
    fd, p = tempfile.mkstemp(suffix='.py', prefix='repair_worker_')
    with os.fdopen(fd, 'w') as f:
        f.write(_SCORE_WORKER)
    _worker_path = p
    return p


def score_code(code: str, gt_path: str) -> dict:
    payload = json.dumps({'code': code, 'gt_path': gt_path})
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker()],
            input=payload, capture_output=True, text=True, timeout=50,
            env={**os.environ, 'LD_LIBRARY_PATH': '/workspace/.local/lib'})
        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        return {'iou': None, 'cd': None, 'error_type': 'runtime_error',
                'error_msg': (proc.stderr or '')[-200:]}
    except subprocess.TimeoutExpired:
        return {'iou': None, 'cd': None, 'error_type': 'timeout', 'error_msg': None}
    except Exception as e:
        return {'iou': None, 'cd': None, 'error_type': 'runtime_error', 'error_msg': str(e)[:200]}


# ---------------------------------------------------------------------------
# Wrong-primitive detection
# ---------------------------------------------------------------------------

def is_wrong_primitive(code: str) -> bool:
    """Return True if code uses box() but no .sketch()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    has_box = False
    has_sketch = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == 'box':
            has_box = True
        if isinstance(node, ast.Attribute) and node.attr == 'sketch':
            has_sketch = True
    return has_box and not has_sketch


# ---------------------------------------------------------------------------
# Build model inputs for the three conditions
# ---------------------------------------------------------------------------

PROMPTS = {
    'A': (
        "The 4-view render shows the target 3D shape. The CadQuery code below uses"
        " box() instead of proper sketch+extrude and does not match the target.\n"
        "Rewrite the code to correctly reconstruct the target shape.\n\n"
        "Broken code:\n{code}"
    ),
    'B': (
        "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).\n"
        "The broken prediction uses box() fallback and fails to match the target.\n"
        "Rewrite the code to correctly reconstruct the target shape.\n\n"
        "Broken code:\n{code}"
    ),
    'C': (
        "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).\n"
        "Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced"
        " with a proper sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
}


def _hstack(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Stack two images horizontally, resizing img2 to match img1's height."""
    h = img1.height
    if img2.height != h:
        img2 = img2.resize((int(img2.width * h / img2.height), h), Image.LANCZOS)
    combined = Image.new('RGB', (img1.width + img2.width, h))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined


def build_item(cond: str, gt_img: Image.Image, pred_img: Image.Image, bad_code: str) -> dict:
    desc = PROMPTS[cond].format(code=bad_code)
    if cond == 'A':
        video = [gt_img]
    else:
        # Horizontally concatenate GT and pred renders into a single frame
        video = [_hstack(gt_img, pred_img)]
    return {'video': video, 'description': desc, 'file_name': 'repair'}



# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_batch(model, processor, items: list, max_new_tokens: int = 1024) -> list[str]:
    batch = collate(items, processor=processor, n_points=256, eval=True)
    gen_ids = model.generate(
        input_ids=batch['input_ids'].to(model.device),
        attention_mask=batch['attention_mask'].to(model.device),
        point_clouds=batch['point_clouds'].to(model.device),
        is_pc=batch['is_pc'].to(model.device),
        is_img=batch['is_img'].to(model.device),
        pixel_values_videos=(batch['pixel_values_videos'].to(model.device)
                              if batch.get('pixel_values_videos') is not None else None),
        video_grid_thw=(batch['video_grid_thw'].to(model.device)
                        if batch.get('video_grid_thw') is not None else None),
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        bad_words_ids=[[model.config.video_token_id]],
    )
    prompt_len = batch['input_ids'].shape[1]
    return [processor.decode(gen_ids[j, prompt_len:], skip_special_tokens=True)
            for j in range(len(items))]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def report_metrics(label: str, baseline_ious: list[float], results: list[dict]) -> dict:
    deltas_iou, deltas_cd = [], []
    valid, meaningful = 0, 0
    for base_iou, r in zip(baseline_ious, results):
        if r['iou'] is not None:
            valid += 1
            di = r['iou'] - base_iou
            deltas_iou.append(di)
            if di > 0.05:
                meaningful += 1
        if r['cd'] is not None and r.get('baseline_cd') is not None:
            deltas_cd.append(r['cd'] - r['baseline_cd'])

    n = len(results)
    print(f'\n--- {label} (n={n}) ---')
    if deltas_iou:
        print(f'  valid rate:          {valid}/{n} = {valid/n*100:.1f}%')
        print(f'  mean  ΔIoU:          {np.mean(deltas_iou):+.4f}')
        print(f'  median ΔIoU:         {np.median(deltas_iou):+.4f}')
        print(f'  % ΔIoU > 0.05:       {meaningful/n*100:.1f}%  ({meaningful}/{n})')
        if deltas_cd:
            print(f'  mean  ΔCD:           {np.mean(deltas_cd):+.6f}')
            print(f'  median ΔCD:          {np.median(deltas_cd):+.6f}')
        # IoU distribution
        arr = np.array(deltas_iou)
        print(f'  ΔIoU distribution:  '
              f'<-0.05:{(arr<-0.05).mean()*100:.0f}%  '
              f'[-0.05,0):{((arr>=-0.05)&(arr<0)).mean()*100:.0f}%  '
              f'[0,0.05):{((arr>=0)&(arr<0.05)).mean()*100:.0f}%  '
              f'≥0.05:{(arr>=0.05).mean()*100:.0f}%')
    else:
        print('  no valid repairs')
    return {'mean_delta_iou': float(np.mean(deltas_iou)) if deltas_iou else None,
            'median_delta_iou': float(np.median(deltas_iou)) if deltas_iou else None,
            'pct_meaningful': meaningful / n if n else 0,
            'valid_rate': valid / n if n else 0,
            'mean_delta_cd': float(np.mean(deltas_cd)) if deltas_cd else None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n',            type=int,  default=100,
                        help='Number of wrong_primitive cases to test (default: 100)')
    parser.add_argument('--out',          default='data/repair_feasibility')
    parser.add_argument('--checkpoint',   default='checkpoints/cadrille-rl')
    parser.add_argument('--analysis-dir', default='data/analysis/deepcad_rl_img')
    parser.add_argument('--gt-dir',       default='data/deepcad_test_mesh')
    parser.add_argument('--batch-size',   type=int, default=4)
    parser.add_argument('--score-workers',type=int, default=4)
    parser.add_argument('--dry-run',      action='store_true',
                        help='Select cases and print stats, skip inference')
    args = parser.parse_args()

    analysis_dir = _REPO / args.analysis_dir
    gt_dir       = _REPO / args.gt_dir
    out_dir      = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Select wrong_primitive cases
    # ------------------------------------------------------------------
    print('Scanning for wrong_primitive cases...')
    with open(analysis_dir / 'metadata.jsonl') as f:
        meta = [json.loads(l) for l in f]

    candidates = []
    for row in meta:
        if row.get('iou') is None:
            continue
        iou = float(row['iou'])
        if not (0.30 <= iou <= 0.88):
            continue
        stem = row['case_id']
        py_path = analysis_dir / f'{stem}_pred.py'
        if not py_path.exists():
            continue
        code = py_path.read_text()
        if not is_wrong_primitive(code):
            continue
        pred_render = analysis_dir / f'{stem}_pred_render.png'
        gt_render   = gt_dir / f'{stem}_render.png'
        gt_stl      = gt_dir / f'{stem}.stl'
        if not (pred_render.exists() and gt_render.exists() and gt_stl.exists()):
            continue
        candidates.append({'stem': stem, 'iou': iou, 'code': code,
                            'pred_render': pred_render, 'gt_render': gt_render,
                            'gt_stl': gt_stl})

    # sort by IoU (middle range most informative), pick n
    candidates.sort(key=lambda x: abs(x['iou'] - 0.6))
    selected = candidates[:args.n]
    print(f'  found {len(candidates)} wrong_primitive cases, using {len(selected)}')
    print(f'  IoU range: [{min(c["iou"] for c in selected):.3f}, '
          f'{max(c["iou"] for c in selected):.3f}]')

    if args.dry_run:
        print('Dry run — exiting before inference.')
        return

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f'\nLoading model from {args.checkpoint}...')
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28, padding_side='left')
    model = Cadrille.from_pretrained(
        str(_REPO / args.checkpoint), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')
    model.eval()
    print('  model loaded.')

    # ------------------------------------------------------------------
    # 3. Load images
    # ------------------------------------------------------------------
    print('Loading renders...')
    for c in selected:
        c['gt_img']   = Image.open(c['gt_render']).convert('RGB')
        c['pred_img'] = Image.open(c['pred_render']).convert('RGB')

    # ------------------------------------------------------------------
    # 4. Inference — three conditions
    # ------------------------------------------------------------------
    all_generated: dict[str, list[str]] = {'A': [], 'B': [], 'C': []}

    for cond in ['A', 'B', 'C']:
        print(f'\nRunning condition {cond}...')
        items = [build_item(cond, c['gt_img'], c['pred_img'], c['code']) for c in selected]
        codes: list[str] = []
        for i in tqdm(range(0, len(items), args.batch_size), desc=f'  infer-{cond}'):
            chunk = items[i:i + args.batch_size]
            codes.extend(infer_batch(model, processor, chunk))
        all_generated[cond] = codes

        # save generated codes
        cond_dir = out_dir / f'cond_{cond}'
        cond_dir.mkdir(exist_ok=True)
        for c, code in zip(selected, codes):
            (cond_dir / f'{c["stem"]}_repaired.py').write_text(code)

    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Score (parallel subprocess workers)
    # ------------------------------------------------------------------
    print('\nScoring repaired codes (IoU + CD)...')

    # Also score original bad codes for ΔCD baseline
    print('  Scoring original bad codes for CD baseline...')
    base_cds: list[float | None] = []
    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        futs = [pool.submit(score_code, c['code'], str(c['gt_stl'])) for c in selected]
        for c, fut in zip(selected, tqdm(futs, desc='  base-score')):
            r = fut.result()
            base_cds.append(r.get('cd'))

    all_results: dict[str, list[dict]] = {}
    for cond in ['A', 'B', 'C']:
        print(f'  Scoring condition {cond}...')
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
            futs = [pool.submit(score_code, code, str(c['gt_stl']))
                    for code, c in zip(all_generated[cond], selected)]
            for base_cd, fut in zip(base_cds, tqdm(futs, desc=f'  score-{cond}')):
                r = fut.result()
                r['baseline_cd'] = base_cd
                results.append(r)
        all_results[cond] = results

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    base_ious = [c['iou'] for c in selected]
    print('\n' + '=' * 60)
    print('REPAIR FEASIBILITY RESULTS')
    print('Success criteria: mean ΔIoU > 0, ≥20% cases with ΔIoU > 0.05')
    print('=' * 60)

    summary = {}
    for cond in ['A', 'B', 'C']:
        label = {
            'A': 'Cond A: GT views + bad code',
            'B': 'Cond B: GT views + pred render + bad code',
            'C': 'Cond C: GT views + pred render + bad code + action  [OURS]',
        }[cond]
        summary[cond] = report_metrics(label, base_ious, all_results[cond])

    # Verdict
    print('\n' + '=' * 60)
    print('VERDICT')
    c_metrics = summary['C']
    passed = (
        (c_metrics['mean_delta_iou'] or -1) > 0 and
        c_metrics['pct_meaningful'] >= 0.20 and
        c_metrics['valid_rate'] >= 0.70
    )
    if passed:
        print('✓ FEASIBILITY CONFIRMED — action-conditioned repair has signal.')
        print('  → Proceed to Step 2 (synthetic corruption + LoRA SFT).')
    else:
        print('✗ SIGNAL WEAK — revisit prompt design or task definition.')
        print('  Consider: prompt wording, base model capability, task scope.')

    # Also check if action (C) beats no-action (B) to isolate action contribution
    di_b = summary['B']['mean_delta_iou'] or 0
    di_c = summary['C']['mean_delta_iou'] or 0
    if di_c > di_b + 0.01:
        print(f'  Action bonus: C ({di_c:+.4f}) > B ({di_b:+.4f}) — action token adds value.')
    else:
        print(f'  Action bonus unclear: C ({di_c:+.4f}) vs B ({di_b:+.4f}).')

    # Save results
    out_json = out_dir / 'results.json'
    out_json.write_text(json.dumps({
        'n': len(selected),
        'baseline_mean_iou': float(np.mean(base_ious)),
        'conditions': summary,
        'per_case': [
            {'stem': c['stem'], 'baseline_iou': c['iou'],
             **{f'cond_{k}_iou': all_results[k][i].get('iou')
                for k, k2 in [('A','A'),('B','B'),('C','C')]},
             **{f'cond_{k}_cd': all_results[k][i].get('cd')
                for k, k2 in [('A','A'),('B','B'),('C','C')]}}
            for i, c in enumerate(selected)
        ]
    }, indent=2))
    print(f'\nResults saved to {out_json}')


if __name__ == '__main__':
    main()
