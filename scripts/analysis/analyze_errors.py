"""Full error taxonomy analysis for Cadrille SFT and RL models.

Runs cadrille-sft and cadrille-rl in both img and pc modalities on the full
DeepCAD and Fusion360 test sets. For each case saves: generated code, IoU,
error type, predicted STL (if execution succeeded), and 4-view render.

Error types:
  success        IoU > 0
  zero_iou       code ran, mesh valid, IoU = 0
  syntax_error   SyntaxError at compile time
  runtime_error  CadQuery / mesh operation failed at runtime
  timeout        execution exceeded 32 s
  render_fail    input rendering / point-cloud prep failed (skipped)

Pipeline (GPU + CPU parallel):
  GPU: generates one batch of codes
  CPU warm pool (spawn, pre-imported cadquery/trimesh): immediately scores each
       code as batches arrive — no subprocess startup overhead per case
  Main thread: renders completed STLs while GPU works on next batch

Output layout:
  data/analysis/{dataset}_{model}_{modality}/
    metadata.jsonl              one JSON line per case
    {stem}_pred.py              generated CadQuery code
    {stem}_pred.stl             predicted mesh  (if exec succeeded)
    {stem}_pred_render.png      4-view render   (if exec succeeded)

Resume:
  Cases already in metadata.jsonl are skipped.
  If {stem}_pred.py exists on disk, code is loaded from there (no re-inference).

Usage
-----
  # Full run (all combos):
  python3 tools/analyze_errors.py

  # Subset for quick smoke-test:
  python3 tools/analyze_errors.py --models sft --modalities img --datasets deepcad

  # Summary only:
  python3 tools/analyze_errors.py --summary-only
"""

import argparse
import io
import json
import os
import queue
import signal
import sys
import tempfile
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadrille import Cadrille, collate
from common.meshio import render_img

_REPO = Path(__file__).parent.parent
_CHECKPOINTS = {
    'sft': _REPO / 'checkpoints' / 'cadrille-sft',
    'rl':  _REPO / 'checkpoints' / 'cadrille-rl',
}
_DATASETS = {
    'deepcad':   _REPO / 'data' / 'deepcad_test_mesh',
    'fusion360': _REPO / 'data' / 'fusion360_test_mesh',
}


# ---------------------------------------------------------------------------
# Subprocess scorer — same approach as rl/reward.py but with STL save + IoU
# subprocess.run() per call: no persistent memory, OCP shared pages released
# after each call.  ThreadPoolExecutor manages concurrency.
# ---------------------------------------------------------------------------

_SCORE_WORKER = textwrap.dedent('''\
    import sys, json, io, os, warnings, signal
    import numpy as np
    import trimesh
    import cadquery as cq  # noqa

    def _alarm(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(36)

    def _transform(m):
        m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
        ext = np.max(m.extents)
        if ext > 1e-7:
            m.apply_scale(2.0 / ext)
        return m

    try:
        p = json.loads(sys.stdin.read())
        code, gt_path, out_stl = p["code"], p["gt_path"], p["out_stl"]

        # compile
        try:
            code_obj = compile(code, "<string>", "exec")
        except SyntaxError as e:
            signal.alarm(0)
            print(json.dumps({"iou": None, "error_type": "syntax_error",
                               "error_msg": str(e)[:200]}))
            sys.exit(0)

        # execute
        try:
            g = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(code_obj, g)
            compound = g["r"].val()
            verts, faces = compound.tessellate(0.001, 0.1)
            pred = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
            assert len(pred.faces) > 2
            buf = trimesh.exchange.stl.export_stl(pred)
            pred = trimesh.load(io.BytesIO(buf), file_type="stl", force="mesh")
        except Exception as e:
            signal.alarm(0)
            print(json.dumps({"iou": None, "error_type": "runtime_error",
                               "error_msg": str(e)[:200]}))
            sys.exit(0)

        # save STL
        try:
            with open(out_stl, "wb") as f:
                f.write(trimesh.exchange.stl.export_stl(pred))
        except Exception:
            pass

        # IoU
        try:
            p_m = _transform(pred)
            gt  = _transform(trimesh.load_mesh(gt_path))
            iv  = 0.0
            for gi in gt.split():
                for pi in p_m.split():
                    s = gi.intersection(pi)
                    iv += s.volume if s is not None else 0.0
            gv = sum(m.volume for m in gt.split())
            pv = sum(m.volume for m in p_m.split())
            uv = gv + pv - iv
            iou = float(iv / uv) if uv > 0 else 0.0
        except Exception as e:
            signal.alarm(0)
            print(json.dumps({"iou": None, "error_type": "runtime_error",
                               "error_msg": "iou:" + str(e)[:150]}))
            sys.exit(0)

        signal.alarm(0)
        print(json.dumps({"iou": iou,
                           "error_type": "success" if iou > 0 else "zero_iou",
                           "error_msg": None}))
    except TimeoutError:
        print(json.dumps({"iou": None, "error_type": "timeout", "error_msg": None}))
    except Exception as e:
        signal.alarm(0)
        print(json.dumps({"iou": None, "error_type": "runtime_error",
                           "error_msg": str(e)[:200]}))
    sys.stdout.flush()
''')

_worker_script_path: Optional[str] = None


def _get_worker_script() -> str:
    global _worker_script_path
    if _worker_script_path and os.path.exists(_worker_script_path):
        return _worker_script_path
    fd, p = tempfile.mkstemp(suffix='.py', prefix='cad_ana_worker_')
    with os.fdopen(fd, 'w') as f:
        f.write(_SCORE_WORKER)
    _worker_script_path = p
    return p


def _score_one(code: str, gt_path: str, out_stl: str) -> dict:
    """Run code in a subprocess, compute IoU, save STL. Returns metadata dict."""
    import subprocess
    payload = json.dumps({'code': code, 'gt_path': gt_path, 'out_stl': out_stl})
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker_script()],
            input=payload, capture_output=True, text=True, timeout=42)
        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        return {'iou': None, 'error_type': 'runtime_error',
                'error_msg': (proc.stderr or '')[-200:]}
    except subprocess.TimeoutExpired:
        return {'iou': None, 'error_type': 'timeout', 'error_msg': None}
    except Exception as e:
        return {'iou': None, 'error_type': 'runtime_error', 'error_msg': str(e)[:200]}


# ---------------------------------------------------------------------------
# Input-preparation helpers
# ---------------------------------------------------------------------------

def _fps(points: np.ndarray, k: int) -> np.ndarray:
    n = len(points)
    selected = np.zeros(k, dtype=int)
    dists = np.full(n, np.inf)
    cur = 0
    for i in range(k):
        selected[i] = cur
        diff = points - points[cur]
        d = (diff * diff).sum(axis=1)
        np.minimum(dists, d, out=dists)
        cur = int(np.argmax(dists))
    return points[selected]


def build_img_item(stl_path: Path) -> dict | None:
    try:
        item = render_img(str(stl_path))
        item['description'] = 'Generate cadquery code'
        item['file_name']   = stl_path.stem
        return item
    except Exception:
        return None


def build_pc_item(stl_path: Path, n_points: int = 256) -> dict | None:
    try:
        mesh = trimesh.load(str(stl_path))
        pts, _ = trimesh.sample.sample_surface(mesh, 8192)
        pc = _fps(np.asarray(pts, dtype=np.float32), n_points)
        pc = (pc - 0.5) * 2
        return {'point_cloud': pc, 'description': 'Generate cadquery code',
                'file_name': stl_path.stem}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-batch GPU inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer_batch(model, processor, items: list, max_new_tokens: int) -> list[str]:
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
# Per-run logic — GPU + warm pool pipelined
# ---------------------------------------------------------------------------

def _collect_result(fut, stl, code, out_dir: Path, mf) -> None:
    """Render STL (if saved) and write one metadata line."""
    try:
        result = fut.result()
    except Exception as e:
        result = {'iou': None, 'error_type': 'runtime_error', 'error_msg': str(e)[:200]}
    pred_stl = out_dir / f'{stl.stem}_pred.stl'
    if pred_stl.exists():
        try:
            rendered = render_img(str(pred_stl))
            rendered['video'][0].save(str(out_dir / f'{stl.stem}_pred_render.png'))
        except Exception:
            pass
    mf.write(json.dumps({'case_id': stl.stem,
                         'iou':        result.get('iou'),
                         'error_type': result.get('error_type'),
                         'error_msg':  result.get('error_msg'),
                         'code_len':   len(code)}) + '\n')
    mf.flush()


def run_analysis(model, processor, stl_files: list[Path], out_dir: Path,
                 modality: str, batch_size: int, max_new_tokens: int,
                 score_workers: int) -> None:
    """GPU inference + CPU scoring, fully pipelined.

    Architecture:
      prep_pool (4 threads) → prep_queue → GPU infer (main thread)
                                                  ↓ per batch
                                           score_pool (subprocess workers)

    prepare (FPS sampling / render load) runs in background threads so GPU
    starts as soon as the first batch is ready — no blocking wait for all items.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'

    # --- resume: load already-completed case IDs ----------------------------
    done: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['case_id'])
                except Exception:
                    pass
    todo = [p for p in stl_files if p.stem not in done]
    print(f'    {len(done)} cached, {len(todo)} remaining')
    if not todo:
        return

    build_fn  = build_img_item if modality == 'img' else build_pc_item
    # pc mode: FPS is numpy-heavy; more threads squeeze more GIL-release parallelism
    # img mode: loads pre-rendered PNG (fast IO) → 2 threads enough
    n_prep = 8 if modality == 'pc' else 2

    # bounded queue so prep workers don't race ahead and eat memory
    prep_q: queue.Queue = queue.Queue(maxsize=batch_size * 8)

    def _prep_worker() -> None:
        """Submit all build_fn calls, put (stl, item_or_None) into queue."""
        with ThreadPoolExecutor(max_workers=n_prep) as prep_pool:
            futs = {prep_pool.submit(build_fn, stl): stl for stl in todo}
            for fut in as_completed(futs):
                stl = futs[fut]
                try:
                    item = fut.result()
                except Exception:
                    item = None
                prep_q.put((stl, item))
        prep_q.put(None)  # sentinel

    prep_thread = threading.Thread(target=_prep_worker, daemon=True)
    prep_thread.start()

    score_pool = ThreadPoolExecutor(max_workers=score_workers)
    pending: dict = {}   # future → (stl, code)
    n_scored = 0

    def _drain(mf) -> int:
        drained = 0
        for fut in [f for f in list(pending) if f.done()]:
            stl, code = pending.pop(fut)
            _collect_result(fut, stl, code, out_dir, mf)
            drained += 1
        return drained

    def _submit_batch(chunk_stls, chunk_items, mf, pbar) -> None:
        nonlocal n_scored
        # separate cached vs needs-inference
        infer_stls, infer_items, cached_pairs = [], [], []
        for stl, item in zip(chunk_stls, chunk_items):
            py_path = out_dir / f'{stl.stem}_pred.py'
            if py_path.exists():
                cached_pairs.append((stl, py_path.read_text()))
            else:
                infer_stls.append(stl)
                infer_items.append(item)

        # GPU inference for items without cached code
        if infer_items:
            codes = _infer_batch(model, processor, infer_items, max_new_tokens)
            for stl, code in zip(infer_stls, codes):
                (out_dir / f'{stl.stem}_pred.py').write_text(code)
                cached_pairs.append((stl, code))

        # submit all to score pool
        for stl, code in cached_pairs:
            out_stl = str(out_dir / f'{stl.stem}_pred.stl')
            fut = score_pool.submit(_score_one, code, str(stl), out_stl)
            pending[fut] = (stl, code)

        n_scored += _drain(mf)
        pbar.n = n_scored
        pbar.refresh()

    print(f'    Streaming inference+scoring on {len(todo)} cases '
          f'(batch={batch_size}, prep={n_prep} threads, score={score_workers} workers)...')

    chunk_stls, chunk_items = [], []
    pbar = tqdm(total=len(todo), desc='    scored', unit='case')

    with open(meta_path, 'a') as mf:
        while True:
            item_pkg = prep_q.get()
            if item_pkg is None:
                break
            stl, item = item_pkg

            if item is None:
                mf.write(json.dumps({'case_id': stl.stem, 'iou': None,
                                     'error_type': 'render_fail',
                                     'error_msg': 'input_prep_failed',
                                     'code_len': 0}) + '\n')
                mf.flush()
                continue

            chunk_stls.append(stl)
            chunk_items.append(item)

            if len(chunk_stls) >= batch_size:
                _submit_batch(chunk_stls, chunk_items, mf, pbar)
                chunk_stls, chunk_items = [], []

        # flush partial last batch
        if chunk_stls:
            _submit_batch(chunk_stls, chunk_items, mf, pbar)

        # drain remaining scoring futures
        for fut in tqdm(as_completed(dict(pending)), total=len(pending),
                        desc='    drain', leave=False):
            if fut in pending:
                stl, code = pending.pop(fut)
                _collect_result(fut, stl, code, out_dir, mf)
                n_scored += 1
        pbar.n = n_scored
        pbar.refresh()
        pbar.close()

    score_pool.shutdown(wait=False)
    prep_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(out_root: Path) -> None:
    print('\n' + '=' * 64)
    print('SUMMARY')
    print('=' * 64)
    for run_dir in sorted(out_root.iterdir()):
        meta = run_dir / 'metadata.jsonl'
        if not meta.exists():
            continue
        rows = []
        with open(meta) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if not rows:
            continue
        total = len(rows)
        from collections import Counter
        et_counts = Counter(r.get('error_type', 'unknown') for r in rows)
        ious = [r['iou'] for r in rows if r.get('iou') and r['iou'] > 0]
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        print(f'\n{run_dir.name}  (n={total})')
        print(f'  mean_iou (success only): {mean_iou:.4f}')
        print(f'  error breakdown: ' +
              '  '.join(f'{k}={v}' for k, v in sorted(et_counts.items())))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models',         nargs='+', default=['sft', 'rl'],
                        choices=['sft', 'rl'])
    parser.add_argument('--modalities',     nargs='+', default=['img', 'pc'],
                        choices=['img', 'pc'])
    parser.add_argument('--datasets',       nargs='+', default=['deepcad', 'fusion360'],
                        choices=['deepcad', 'fusion360'])
    parser.add_argument('--out-dir',        default='data/analysis')
    parser.add_argument('--batch-size',     type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--score-workers',  type=int, default=4,
                        help='Concurrent subprocess workers for CadQuery scoring (default: 4). '
                             'Uses subprocess-per-call — no persistent memory overhead.')
    parser.add_argument('--summary-only',   action='store_true')
    args = parser.parse_args()

    out_root = _REPO / args.out_dir

    if args.summary_only:
        print_summary(out_root)
        return

    for model_name in args.models:
        ckpt = _CHECKPOINTS[model_name]
        print(f'\n{"=" * 64}')
        print(f'Model: {model_name}  |  ckpt: {ckpt}')
        processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen2-VL-2B-Instruct', min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28, padding_side='left')
        model = Cadrille.from_pretrained(
            str(ckpt), torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', device_map='auto')
        model.eval()

        for modality in args.modalities:
            for dataset_name in args.datasets:
                test_dir  = _DATASETS[dataset_name]
                stl_files = sorted(test_dir.glob('*.stl'))
                run_name  = f'{dataset_name}_{model_name}_{modality}'
                out_dir   = out_root / run_name
                print(f'\n  [{run_name}]  {len(stl_files)} STLs → {out_dir}')
                run_analysis(model, processor, stl_files, out_dir,
                             modality, args.batch_size, args.max_new_tokens,
                             args.score_workers)

        del model
        torch.cuda.empty_cache()

    print_summary(out_root)


if __name__ == '__main__':
    main()
