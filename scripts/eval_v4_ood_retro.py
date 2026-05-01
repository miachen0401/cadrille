"""Retroactive 50-OOD eval for v4 (ood_enhance) saved checkpoints.

The currently-running v4-holdout (sft-s50k-lr2e-4-b8a4-img-0430-0828) was
launched before the IID/OOD stratified-sampling refactor (commit 4cf86a1).
Its online predictions JSONL contains a single random n=50 BenchCAD val
sample per step, of which only ~9 land in the held-out families. Per-step
OOD trajectory is noisy.

This script fixes that retroactively: for each saved checkpoint, it loads
the model and runs greedy inference on a fixed stratified 50-OOD sample
(10 holdout families × 5 cases each) drawn deterministically from val.pkl.

Output:
  <out>/step-<N>.jsonl     — per-case greedy preds + IoU + family
  <out>/summary.csv        — step,family,mean_iou,exec_rate,ess_pass

Usage:
    set -a; source .env; set +a
    uv run python scripts/eval_v4_ood_retro.py \
        --ckpts /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828 \
        --base-model Qwen/Qwen3-VL-2B-Instruct --backbone qwen3_vl \
        --out eval_outputs/v4_ood_retro \
        --batch-size 4 --score-workers 8
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import pickle
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.holdout import HOLDOUT_FAMILIES  # noqa: E402
from common.model import Cadrille  # noqa: E402
from eval.bench import run_bench  # noqa: E402


def _build_ood_rows(seed: int = 42, n_per_fam: int = 5) -> list[dict]:
    """50 OOD rows = 10 fams × 5, deterministic via seeded sample of val.pkl."""
    val_pkl = REPO / 'data/benchcad/val.pkl'
    with open(val_pkl, 'rb') as f:
        rows = pickle.load(f)

    rng = random.Random(seed)
    ood_rows = []
    for fam in sorted(HOLDOUT_FAMILIES):
        fam_rows = [r for r in rows if r.get('family') == fam]
        rng.shuffle(fam_rows)
        for r in fam_rows[:n_per_fam]:
            base = REPO / 'data/benchcad'
            png = base / r['png_path']
            py  = base / r['py_path']
            stl = base / r['mesh_path']
            if not (png.exists() and py.exists() and stl.exists()):
                continue
            ood_rows.append({
                'stem':           r['uid'],
                'composite_png':  Image.open(png).convert('RGB'),
                'gt_code':        py.read_text(),
                'family':         fam,
                'difficulty':     r.get('difficulty', '?'),
                'base_plane':     r.get('base_plane', '?'),
                'split':          'ood',
                'feature_tags':   r.get('feature_tags', '{}'),
                'feature_count':  r.get('feature_count', 0),
                'gt_mesh_path':   str(stl),
            })
    return ood_rows


def _list_ckpts(ckpt_root: Path) -> list[Path]:
    out = []
    for p in sorted(ckpt_root.glob('checkpoint-*')):
        if not p.is_dir():
            continue
        try:
            int(p.name.split('-', 1)[1])
            out.append(p)
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts',         required=True,
                    help='Either a ckpt dir OR a ckpt-root containing checkpoint-* subdirs.')
    ap.add_argument('--base-model',    default='Qwen/Qwen3-VL-2B-Instruct')
    ap.add_argument('--backbone',      default='qwen3_vl', choices=['qwen2_vl', 'qwen3_vl'])
    ap.add_argument('--out',           required=True)
    ap.add_argument('--batch-size',    type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers', type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = Path(args.ckpts)
    if ckpt_in.name.startswith('checkpoint-'):
        ckpts = [ckpt_in]
    else:
        ckpts = _list_ckpts(ckpt_in)
    if not ckpts:
        print(f'no checkpoints found under {ckpt_in}', file=sys.stderr)
        sys.exit(2)
    print(f'[1/3] {len(ckpts)} checkpoint(s) found:', flush=True)
    for p in ckpts:
        print(f'   - {p.name}', flush=True)

    print(f'[2/3] building stratified 50-OOD sample (10 fams × 5) ...', flush=True)
    rows = _build_ood_rows(seed=42, n_per_fam=5)
    print(f'   got {len(rows)} OOD rows from {len(set(r["family"] for r in rows))} families', flush=True)

    print(f'[3/3] loading processor + iterating checkpoints', flush=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        token=os.environ.get('HF_TOKEN'),
        min_pixels=200_704, max_pixels=1_003_520,
    )

    summary_rows = []
    for ckpt in ckpts:
        step = int(ckpt.name.split('-', 1)[1])
        ckpt_out = out_dir / f'step-{step:06d}'
        if (ckpt_out / 'metadata.jsonl').exists():
            print(f'  step={step} → already done, skipping', flush=True)
            continue
        print(f'\n  step={step} loading model from {ckpt}', flush=True)
        model = Cadrille.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            backbone=args.backbone,
        ).eval().to('cuda')
        run_bench(
            rows=rows, model=model, processor=processor,
            out_dir=ckpt_out,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            score_workers=args.score_workers,
            save_code=True,
        )
        del model
        torch.cuda.empty_cache()

    for ckpt in ckpts:
        step = int(ckpt.name.split('-', 1)[1])
        meta = out_dir / f'step-{step:06d}' / 'metadata.jsonl'
        if not meta.exists():
            continue
        rows_meta = [json.loads(ln) for ln in meta.open()]
        from collections import defaultdict
        fams = defaultdict(list)
        for r in rows_meta:
            fams[r.get('family') or '?'].append(r)
        for fam, recs in sorted(fams.items()):
            valid = [r for r in recs if r.get('iou') is not None and r['iou'] >= 0]
            iou_mean = (sum(r['iou'] for r in valid) / len(valid)) if valid else None
            exec_rate = len(valid) / max(1, len(recs))
            summary_rows.append({
                'step': step, 'family': fam,
                'n': len(recs), 'iou_mean': iou_mean, 'exec_rate': exec_rate,
            })

    csv_path = out_dir / 'summary.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step', 'family', 'n', 'iou_mean', 'exec_rate'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nsummary written → {csv_path}')


if __name__ == '__main__':
    main()
