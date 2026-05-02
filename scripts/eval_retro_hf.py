"""HF-driven retro 50-OOD eval for any cadrille run.

Looks up checkpoint-N subdirs on a HF model repo, downloads each one in turn
to a /ephemeral cache, runs scripts/eval_v4_ood_retro._build_ood_rows + run_bench
on it, then deletes the cached weights so disk stays under control.

Useful when local save_total_limit rotated out earlier ckpts but the HF
upload kept them (uploaded by train/sft/hf_uploader during training).

Output: same shape as eval_v4_ood_retro.py — eval_outputs/<out>/step-NNNNNN/
        with metadata.jsonl + per-case .py files + summary.csv.

Usage:
    set -a; source .env; set +a
    uv run python scripts/eval_retro_hf.py \
        --repo Hula0401/cadrille-qwen3vl-2b-v3-clean-50k \
        --steps 2000,4000,6000,...,46000 \
        --out eval_outputs/v3_ood_retro_hf
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.model import get_cadrille_class  # noqa: E402
from eval.bench import run_bench  # noqa: E402
# Reuse the corrected (online_eval-aligned) OOD picker.
from scripts.eval_v4_ood_retro import _build_ood_rows  # noqa: E402


def _list_hf_steps(repo: str, token: str | None) -> list[int]:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = api.list_repo_files(repo)
    steps = sorted({int(f.split('/')[0].split('-')[1])
                    for f in files
                    if f.startswith('checkpoint-') and '/' in f})
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo',          required=True, help='HF model repo id')
    ap.add_argument('--steps',         default='all',
                    help='comma-separated step list, or "all" to eval every uploaded ckpt')
    ap.add_argument('--out',           required=True)
    ap.add_argument('--cache-dir',     type=Path, default=Path('/ephemeral/_hf_ckpt_cache'),
                    help='Where each downloaded ckpt is staged (deleted after eval).')
    ap.add_argument('--keep-cache',    action='store_true',
                    help='Skip the post-eval cache cleanup.')
    ap.add_argument('--base-model',    default='Qwen/Qwen3-VL-2B-Instruct')
    ap.add_argument('--backbone',      default='qwen3_vl', choices=['qwen2_vl', 'qwen3_vl'])
    ap.add_argument('--batch-size',    type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers', type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get('HF_TOKEN')

    if args.steps == 'all':
        steps = _list_hf_steps(args.repo, token)
        if not steps:
            print(f'no checkpoint-* dirs on {args.repo}', file=sys.stderr); sys.exit(2)
        print(f'[hf] {len(steps)} ckpts on {args.repo}: {steps[0]} … {steps[-1]}')
    else:
        steps = sorted({int(s.strip()) for s in args.steps.split(',') if s.strip()})
        print(f'[hf] {len(steps)} requested steps: {steps}')

    print(f'[1/3] building 50-OOD sample (online_eval-aligned, seed=42) ...')
    rows = _build_ood_rows(seed=42, n_per_fam=5)
    print(f'   {len(rows)} OOD rows from {len(set(r["family"] for r in rows))} families')

    print(f'[2/3] processor ...')
    processor = AutoProcessor.from_pretrained(
        args.base_model, token=token,
        min_pixels=200_704, max_pixels=1_003_520)

    cadrille_cls = get_cadrille_class(args.backbone)

    print(f'[3/3] iterating ckpts ...')
    for step in steps:
        ckpt_out = out_dir / f'step-{step:06d}'
        if (ckpt_out / 'metadata.jsonl').exists():
            print(f'  step={step} → already done, skipping')
            continue

        print(f'\n--- step={step} ---')
        # Download ONLY this ckpt subdir to the staging cache.
        local = snapshot_download(
            repo_id=args.repo, repo_type='model', token=token,
            cache_dir=str(args.cache_dir),
            allow_patterns=[f'checkpoint-{step}/*'],
        )
        ckpt_dir = Path(local) / f'checkpoint-{step}'
        if not ckpt_dir.is_dir():
            print(f'  ERROR: {ckpt_dir} not found post-download', file=sys.stderr)
            continue

        print(f'  loading model ...')
        model = cadrille_cls.from_pretrained(
            ckpt_dir, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
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

        if not args.keep_cache:
            shutil.rmtree(local, ignore_errors=True)
            # Also try to drop the corresponding HF blob refs (harmless if absent)
            blobs = args.cache_dir / f'models--{args.repo.replace("/", "--")}'
            shutil.rmtree(blobs, ignore_errors=True)
            print(f'  cleaned cache')

    # Aggregate summary
    summary_rows = []
    for step in steps:
        meta = out_dir / f'step-{step:06d}' / 'metadata.jsonl'
        if not meta.exists():
            continue
        rows_meta = [json.loads(ln) for ln in meta.open()]
        fams = defaultdict(list)
        for r in rows_meta:
            fams[r.get('family') or '?'].append(r)
        for fam, recs in sorted(fams.items()):
            valid = [r for r in recs if r.get('iou') is not None and r['iou'] >= 0]
            iou_mean = (sum(r['iou'] for r in valid) / len(valid)) if valid else None
            summary_rows.append({
                'step': step, 'family': fam,
                'n': len(recs), 'iou_mean': iou_mean,
                'exec_rate': len(valid) / max(1, len(recs)),
            })
    csv_path = out_dir / 'summary.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step', 'family', 'n', 'iou_mean', 'exec_rate'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nsummary → {csv_path}')


if __name__ == '__main__':
    main()
