"""Run v3 SFT checkpoint over the full BenchCAD/cad_bench_722 dataset.

Saves per-case predictions + IoU + family metadata as JSONL. v3 saw all 106
families during training, so this is the IID upper-bound for §7 figures.

Output layout (--out <dir>):
  metadata.jsonl   — one row per case (stem, family, gt_code preview,
                     pred_code preview, iou, cd, error_type, …)
  *.py             — full generated code per case (no preview clipping)
  summary.json     — per-family aggregates (mean iou, exec rate, count)

Usage:
    set -a; source .env; set +a
    uv run python scripts/eval_v3_cad_bench_722.py \
        --ckpt /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/checkpoint-46000 \
        --out  eval_outputs/v3_cad_bench_722 \
        --batch-size 4 --score-workers 8

Resumes from a partial run by reading existing metadata.jsonl.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.model import get_cadrille_class  # noqa: E402
from eval.bench import run_bench  # noqa: E402


def _normalize_rows(ds) -> list[dict]:
    """cad_bench_722 → rows compatible with eval.bench.run_bench."""
    rows: list[dict] = []
    for ex in ds:
        rows.append({
            'stem':           ex['substitute_bench_stem'],
            'composite_png':  ex['composite_png'],
            'gt_code':        ex['gt_code'],
            'family':         ex['family'],
            'difficulty':     ex['difficulty'],
            'base_plane':     ex['base_plane'],
            'split':          'cad_bench_722',
            'feature_tags':   ex['feature_tags'],
            'feature_count':  ex['feature_count'],
        })
    return rows


def _summarize_per_family(meta_path: Path) -> dict:
    """Aggregate per-family stats from metadata.jsonl."""
    fams: dict[str, list[dict]] = defaultdict(list)
    rows = []
    with open(meta_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rows.append(rec)
            fams[rec.get('family') or 'UNKNOWN'].append(rec)

    overall = {
        'n_total':  len(rows),
        'n_exec':   sum(1 for r in rows if r.get('iou') is not None and r['iou'] >= 0),
        'iou_mean': (sum(r['iou'] for r in rows if r.get('iou') is not None and r['iou'] >= 0)
                     / max(1, sum(1 for r in rows if r.get('iou') is not None and r['iou'] >= 0))),
    }

    by_fam = {}
    for fam, recs in sorted(fams.items()):
        valid = [r for r in recs if r.get('iou') is not None and r['iou'] >= 0]
        by_fam[fam] = {
            'n':         len(recs),
            'n_exec':    len(valid),
            'exec_rate': len(valid) / max(1, len(recs)),
            'iou_mean':  (sum(r['iou'] for r in valid) / max(1, len(valid))) if valid else None,
            'iou_max':   max((r['iou'] for r in valid), default=None),
        }
    return {'overall': overall, 'per_family': by_fam}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',          required=True)
    ap.add_argument('--base-model',    default='Qwen/Qwen3-VL-2B-Instruct')
    ap.add_argument('--backbone',      default='qwen3_vl', choices=['qwen2_vl', 'qwen3_vl'])
    ap.add_argument('--out',           required=True)
    ap.add_argument('--batch-size',    type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers', type=int, default=8)
    ap.add_argument('--limit',         type=int, default=0,
                    help='Optional cap on number of samples (0 = all 722).')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/3] loading BenchCAD/cad_bench_722 ...', flush=True)
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('BenchCAD_HF_TOKEN') or os.environ.get('HF_TOKEN'))
    rows = _normalize_rows(ds)
    if args.limit:
        rows = rows[: args.limit]
    print(f'  {len(rows)} rows from {len(set(r["family"] for r in rows))} families', flush=True)

    print(f'[2/3] loading model from {args.ckpt} ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        token=os.environ.get('HF_TOKEN'),
        min_pixels=200_704, max_pixels=1_003_520,  # matches train/eval convention
    )
    cadrille_cls = get_cadrille_class(args.backbone)
    model = cadrille_cls.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ).eval().to('cuda')

    print(f'[3/3] running inference + scoring → {out_dir}', flush=True)
    summary = run_bench(
        rows=rows,
        model=model,
        processor=processor,
        out_dir=out_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        score_workers=args.score_workers,
        save_code=True,
    )

    meta_path = out_dir / 'metadata.jsonl'
    fam_summary = _summarize_per_family(meta_path)
    (out_dir / 'summary.json').write_text(json.dumps(fam_summary, indent=2))
    print('---')
    print(f'  overall: n={fam_summary["overall"]["n_total"]} '
          f'exec={fam_summary["overall"]["n_exec"]} '
          f'iou_mean={fam_summary["overall"]["iou_mean"]:.3f}')
    for fam, s in sorted(fam_summary['per_family'].items(),
                         key=lambda kv: -(kv[1]['iou_mean'] or -1))[:10]:
        print(f'  {fam:30s} n={s["n"]:3d} exec={s["exec_rate"]:.2f} '
              f'iou_mean={s["iou_mean"] if s["iou_mean"] is not None else float("nan"):.3f}')
    print(f'\nfull breakdown → {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
