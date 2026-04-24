"""Multi-dataset, multi-temperature, multi-sample evaluation driver.

Metrics per (dataset, split, temperature):
  iou_at_1            — mean IoU of first sample (temp=0 greedy for 0.0, else first temp sample)
  max_iou_at_N        — per-item max IoU across N samples, averaged across items
  exec_rate           — fraction of the N × n_items generations that executed OK
  + feature_recall    — BenchCAD only; hit rate for each GT-positive feature tag

Usage (smoke):
  python -m eval.bench_sweep \\
      --ckpt checkpoints/sft-s5k.../checkpoint-final \\
      --datasets benchcad,deepcad,fusion360 --limit 10 \\
      --temps 0,0.4,0.5,0.75,1.0,1.25 --n-samples 16 \\
      --out eval_outputs/sweep_$(date +%H%M)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from cadrille import Cadrille, collate  # noqa: E402
from common.meshio import render_img  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from dataset import mesh_to_point_cloud  # noqa: E402
from eval.features import feature_recall, aggregate_feature_recall  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset loading — normalises every dataset into a list of items with this shape:
#   {
#     'uid': str,
#     'gt_mesh_path': str,          # local STL
#     'gt_code': Optional[str],     # for BenchCAD; None otherwise
#     'feature_tags': Optional[dict], # BenchCAD only
#     'modality_inputs': {'video': [PIL], 'point_cloud': np.ndarray},  # both pre-rendered
#   }
# ---------------------------------------------------------------------------

def _load_benchcad(limit: int, seed: int = 42) -> dict[str, list[dict]]:
    """Returns {split_name: [items]} for Hula0401/test_bench.

    Uses row['composite_png'] as the img input (matches training distribution).
    GT STL materialised via subprocess exec of gt_code (eval/bench.py helper).
    """
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    ds = load_dataset('Hula0401/test_bench', token=token)

    cache_dir = Path('data/_sweep_cache/benchcad')
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, list[dict]] = {}
    for split_name in ['test_iid', 'test_ood_family', 'test_ood_plane']:
        sp_rows = list(ds[split_name])
        rng = random.Random(seed)
        rng.shuffle(sp_rows)
        sp_rows = sp_rows[:limit] if limit else sp_rows
        items: list[dict] = []
        for row in sp_rows:
            uid = row.get('stem') or row.get('uid') or row.get('file_name')
            stl_path = cache_dir / f'{uid}.stl'
            if not stl_path.exists():
                from eval.bench import _exec_gt_code
                p = _exec_gt_code(row['gt_code'], timeout=60.0)
                if p is None:
                    continue
                os.rename(p, stl_path)
            items.append({
                'uid': uid,
                'gt_mesh_path': str(stl_path),
                'gt_code': row.get('gt_code'),
                'feature_tags': row.get('feature_tags'),
                'composite_png': row.get('composite_png'),
                'dataset': 'benchcad',
                'split': split_name,
            })
        out[f'benchcad/{split_name}'] = items
        print(f'  benchcad/{split_name}: {len(items)} items', flush=True)
    return out


def _load_stl_dir(dir_path: str, dataset_name: str, limit: int, seed: int = 42) -> dict[str, list[dict]]:
    """DeepCAD / Fusion360 test sets: just STL files in a directory."""
    files = sorted(Path(dir_path).glob('*.stl'))
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    shuffled = shuffled[:limit] if limit else shuffled
    items = [{
        'uid': p.stem,
        'gt_mesh_path': str(p),
        'gt_code': None,
        'feature_tags': None,
        'dataset': dataset_name,
        'split': 'test',
    } for p in shuffled]
    print(f'  {dataset_name}: {len(items)} items from {dir_path}', flush=True)
    return {f'{dataset_name}/test': items}


def load_all(datasets: list[str], limit: int, seed: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in datasets:
        if d == 'benchcad':
            out.update(_load_benchcad(limit, seed))
        elif d == 'deepcad':
            out.update(_load_stl_dir('data/deepcad_test_mesh', 'deepcad', limit, seed))
        elif d == 'fusion360':
            out.update(_load_stl_dir('data/fusion360_test_mesh', 'fusion360', limit, seed))
        else:
            raise ValueError(f'unknown dataset: {d}')
    return out


# ---------------------------------------------------------------------------
# Per-item modality inputs — always pc for sweep (cheaper, one modality fixed)
# ---------------------------------------------------------------------------

def build_modality_inputs(item: dict, modality: str, n_points: int = 256) -> dict:
    """Build a single-modality example dict for the collate function."""
    if modality == 'pc':
        import trimesh
        mesh = trimesh.load(item['gt_mesh_path'])
        pc = mesh_to_point_cloud(mesh, n_points)
        pc = (pc - 0.5) * 2
        return {
            'point_cloud': pc,
            'description': 'Generate cadquery code',
            'file_name': item['uid'],
            'gt_mesh_path': item['gt_mesh_path'],
        }
    elif modality == 'img':
        # Strict: only use pre-rendered PNGs. NEVER fall back to on-the-fly Open3D
        # render — that would add a heavy step in the hot loop.
        from PIL import Image
        cpng = item.get('composite_png')
        if cpng is not None:
            if isinstance(cpng, dict):
                img = Image.open(io.BytesIO(cpng['bytes']))
            elif isinstance(cpng, (bytes, bytearray)):
                img = Image.open(io.BytesIO(cpng))
            else:
                img = cpng
        else:
            # DeepCAD / Fusion360: expect {stem}_render.png next to {stem}.stl
            stl = item['gt_mesh_path']
            png_path = stl[:-4] + '_render.png'
            if not os.path.exists(png_path):
                raise FileNotFoundError(
                    f'img modality requires pre-rendered PNG but {png_path} is missing. '
                    f'Download *_test_renders.zip from HF or drop the item.')
            img = Image.open(png_path)
        img = img.convert('RGB').resize((128, 128))
        return {
            'video': [img],
            'description': 'Generate cadquery code',
            'file_name': item['uid'],
            'gt_mesh_path': item['gt_mesh_path'],
        }
    raise ValueError(modality)


# ---------------------------------------------------------------------------
# Batched generation (adapted from eval/passk.py::_generate_one_batch)
# ---------------------------------------------------------------------------

_GEN_KEYS = ('input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img',
             'pixel_values_videos', 'video_grid_thw')


@torch.no_grad()
def generate_batch(model, chunk: list[dict], processor, max_new_tokens: int,
                   temperature: float, device) -> list[str]:
    collate_items = [{k: v for k, v in ex.items() if k != 'gt_mesh_path'} for ex in chunk]
    batch = collate(collate_items, processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]
    if temperature == 0:
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False)
    else:
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True,
                      temperature=temperature, top_p=1.0, top_k=50)
    batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items() if k in _GEN_KEYS}
    out = model.generate(**batch_gpu, **gen_kw)
    return [processor.decode(out[j, prompt_len:], skip_special_tokens=True)
            for j in range(len(chunk))]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_one(code: str, gt_mesh_path: str, timeout: float = 32.0) -> tuple[float, float, bool]:
    iou, cd = compute_metrics(code, gt_mesh_path, timeout)
    ok = iou > -1.0
    return iou, cd, ok


# ---------------------------------------------------------------------------
# Per-(dataset, split, temp) sweep
# ---------------------------------------------------------------------------

def sweep_one_temp(
    items: list[dict],
    model, processor, device,
    temp: float,
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    modality: str,
    score_workers: int,
) -> dict:
    # Build inputs once per item (same across all samples at this temp).
    example_inputs = [build_modality_inputs(it, modality) for it in items]

    # n_samples=1 when temp=0 (deterministic greedy); loop n_samples otherwise.
    n = 1 if temp == 0 else n_samples

    # per-item list of decoded strings
    all_codes: list[list[str]] = [[] for _ in items]
    for s in range(n):
        for start in range(0, len(items), batch_size):
            chunk = example_inputs[start:start + batch_size]
            codes = generate_batch(model, chunk, processor, max_new_tokens, temp, device)
            for j, c in enumerate(codes):
                all_codes[start + j].append(c)
        print(f'    [t={temp:.2f}] pass {s+1}/{n} generated', flush=True)

    # Pipelined scoring
    score_pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []
    for i, codes in enumerate(all_codes):
        for c in codes:
            fut = score_pool.submit(score_one, c, items[i]['gt_mesh_path'])
            pending.append((fut, i, c))

    # Collect
    per_item = [[] for _ in items]
    for fut, i, code in pending:
        iou, cd, ok = fut.result()
        per_item[i].append({'code': code, 'iou': iou, 'cd': cd, 'ok': ok})
    score_pool.shutdown(wait=False)

    # Aggregate
    exec_hits = sum(1 for runs in per_item for r in runs if r['ok'])
    exec_total = sum(len(runs) for runs in per_item)
    valid_ious = [r['iou'] for runs in per_item for r in runs if r['ok']]
    per_item_max = []
    per_item_first = []
    for runs in per_item:
        valid = [r['iou'] for r in runs if r['ok']]
        per_item_max.append(max(valid) if valid else None)
        per_item_first.append(runs[0]['iou'] if runs and runs[0]['ok'] else None)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    summary = {
        'temp': temp,
        'n_samples': n,
        'n_items': len(items),
        'exec_rate': exec_hits / exec_total if exec_total else 0.0,
        'iou_first': _mean(per_item_first),
        'iou_max_at_N': _mean(per_item_max),
        'mean_iou_over_all': float(np.mean(valid_ious)) if valid_ious else None,
    }

    # Feature recall (BenchCAD only — items carry gt_code + feature_tags)
    if items and items[0].get('feature_tags') is not None:
        rows = []
        for i, runs in enumerate(per_item):
            best = None
            best_code = None
            for r in runs:
                if r['ok'] and (best is None or r['iou'] > best):
                    best = r['iou']
                    best_code = r['code']
            if best_code is not None:
                rows.append({'feature_recall': feature_recall(items[i]['feature_tags'], best_code)})
        summary['feature_recall_at_bestN'] = aggregate_feature_recall(rows)

    # Keep per-item raw for disk
    summary['_per_item'] = [
        {'uid': items[i]['uid'],
         'runs': [{'iou': r['iou'], 'cd': r['cd'], 'ok': r['ok']} for r in per_item[i]]}
        for i in range(len(items))
    ]
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base-model', default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--datasets', default='benchcad,deepcad,fusion360')
    ap.add_argument('--temps', default='0,0.4,0.5,0.75,1.0,1.25')
    ap.add_argument('--n-samples', type=int, default=16,
                    help='Samples per item at non-zero temperature')
    ap.add_argument('--limit', type=int, default=20,
                    help='Items per dataset split (BenchCAD has 3 splits, so total=3*limit for benchcad)')
    ap.add_argument('--modality', default='img', choices=['pc', 'img'],
                    help='Img matches the strongest eval path (training composite_png for BenchCAD).')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    temps = [float(t.strip()) for t in args.temps.split(',') if t.strip()]

    print(f'=== eval.bench_sweep ===', flush=True)
    print(f'ckpt   : {args.ckpt}', flush=True)
    print(f'datasets: {datasets}', flush=True)
    print(f'temps  : {temps}', flush=True)
    print(f'n_samp : {args.n_samples}', flush=True)
    print(f'limit  : {args.limit}', flush=True)
    print(f'out    : {out_dir}', flush=True)

    # --- Load datasets ---
    print('Loading datasets ...', flush=True)
    splits = load_all(datasets, args.limit, args.seed)
    total_items = sum(len(v) for v in splits.values())
    print(f'Total: {total_items} items across {len(splits)} splits', flush=True)

    # --- Load model ---
    print(f'Loading processor ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')
    print(f'Loading model from {args.ckpt} ...', flush=True)
    model = Cadrille.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')
    model.eval()
    device = next(model.parameters()).device

    # --- Sweep ---
    all_results: list[dict] = []
    for split_key, items in splits.items():
        if not items:
            continue
        print(f'\n--- {split_key} ({len(items)} items) ---', flush=True)
        for t in temps:
            summary = sweep_one_temp(
                items, model, processor, device,
                temp=t, n_samples=args.n_samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                modality=args.modality,
                score_workers=args.score_workers,
            )
            summary['split_key'] = split_key
            all_results.append(summary)
            print(f'  t={t:.2f} n={summary["n_samples"]:>2}  '
                  f'exec={summary["exec_rate"]*100:.1f}%  '
                  f'iou_first={summary["iou_first"]}  '
                  f'iou_max@{summary["n_samples"]}={summary["iou_max_at_N"]}', flush=True)

    # --- Persist ---
    (out_dir / 'full.json').write_text(json.dumps(all_results, indent=2, default=str))
    # Strip per-item before writing summary
    summary_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != '_per_item'}
        summary_rows.append(row)
    (out_dir / 'summary.json').write_text(json.dumps(summary_rows, indent=2, default=str))

    # Markdown table
    md = [f'# Sweep eval — {args.label or Path(args.ckpt).name}\n']
    md.append('| split | temp | n | exec% | iou@1st | max_iou@N |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for r in summary_rows:
        md.append(f'| {r["split_key"]} | {r["temp"]:.2f} | {r["n_samples"]} | '
                  f'{r["exec_rate"]*100:.1f} | {r["iou_first"]} | {r["iou_max_at_N"]} |')
    # feature recall table (benchcad only)
    md.append('\n## BenchCAD feature recall (max-iou sample per item)\n')
    md.append('| split | temp | has_hole | has_fillet | has_chamfer | has_slot | rotational |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in summary_rows:
        if 'feature_recall_at_bestN' not in r:
            continue
        fr = r['feature_recall_at_bestN']
        def _c(f):
            e = fr.get(f, {})
            if e.get('recall') is None:
                return '—'
            return f'{e["recall"]*100:.0f}% ({e["n_hit"]}/{e["n_gt"]})'
        md.append(f'| {r["split_key"]} | {r["temp"]:.2f} | {_c("has_hole")} | {_c("has_fillet")} | '
                  f'{_c("has_chamfer")} | {_c("has_slot")} | {_c("rotational")} |')

    (out_dir / 'summary.md').write_text('\n'.join(md))
    print(f'\nSaved:\n  {out_dir}/full.json\n  {out_dir}/summary.json\n  {out_dir}/summary.md', flush=True)


if __name__ == '__main__':
    main()
