"""Pick the best ood_enhance ckpt to seed RL from.

For each candidate ckpt, runs greedy + temperature=1.0 sampling (n=16) on the
stratified BC val OOD bucket (10 holdout fams × 5 = 50 rows). Reports for
each ckpt:

  greedy_iou       greedy IoU (mean)
  greedy_ess       greedy essential_pass rate (binary, BenchCAD spec)
  max_iou_at_16    best IoU among 16 t=1.0 samples (mean over rows)
  max_ess_at_16    best ess_score (fractional) among 16 samples (mean over rows)

The pick rule is **highest of the joint score**:
  joint = 0.5 * max_iou@16 + 0.5 * max_ess@16

You can override with --pick-by greedy_iou | max_iou_at_16 | max_ess_at_16.

Usage:
    set -a; source .env; set +a
    uv run python scripts/pick_rl_start_ckpt.py \
        --ckpts /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828 \
        --steps 24000,26000,28000,30000 \
        --out eval_outputs/pick_rl_start

Output:
  <out>/summary.json           per-ckpt metrics
  <out>/step-NNNNNN/{*.py, metadata.jsonl}   per-case predictions kept for audit
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.holdout import HOLDOUT_FAMILIES  # noqa: E402
from common.essential_ops import (  # noqa: E402
    ESSENTIAL_BY_FAMILY, find_ops, essential_score, essential_pass,
)
from common.model import get_cadrille_class  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from eval.bench import _exec_gt_code  # noqa: E402


def _build_ood_rows(seed: int = 42, n_per_fam: int = 5) -> list[dict]:
    rows = pickle.load(open(REPO / 'data/benchcad/val.pkl', 'rb'))
    rng = random.Random(seed)
    out = []
    for fam in sorted(HOLDOUT_FAMILIES):
        fam_rows = [r for r in rows if r.get('family') == fam]
        rng.shuffle(fam_rows)
        for r in fam_rows[:n_per_fam]:
            base = REPO / 'data/benchcad'
            png, py, stl = base / r['png_path'], base / r['py_path'], base / r['mesh_path']
            if not (png.exists() and py.exists() and stl.exists()):
                continue
            out.append({
                'uid':      r['uid'],
                'family':   fam,
                'png_path': str(png),
                'gt_code':  py.read_text(),
                'gt_stl':   str(stl),
            })
    return out


@torch.inference_mode()
def _run_ckpt(ckpt_dir: Path, processor, backbone: str, rows: list[dict],
              n_max_iou_samples: int = 16, max_new_tokens: int = 768,
              batch_size: int = 4) -> list[dict]:
    cadrille_cls = get_cadrille_class(backbone)
    model = cadrille_cls.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ).eval().to('cuda')

    from common.model import collate

    def _gen(temp: float, n_per_row: int) -> list[list[str]]:
        """For each row, return list of n_per_row decoded strings."""
        out: list[list[str]] = [[] for _ in rows]
        for shot in range(n_per_row):
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                items = [{
                    'video':       [Image.open(r['png_path']).convert('RGB')],
                    'description': 'Generate cadquery code',
                    'file_name':   r['uid'],
                } for r in chunk]
                b = collate(items, processor, 256, eval=True)
                out_ids = model.generate(
                    input_ids=b['input_ids'].to('cuda'),
                    attention_mask=b['attention_mask'].to('cuda'),
                    point_clouds=b['point_clouds'].to('cuda'),
                    is_pc=b['is_pc'].to('cuda'),
                    is_img=b['is_img'].to('cuda'),
                    pixel_values_videos=(
                        b['pixel_values_videos'].to('cuda') if b.get('pixel_values_videos') is not None else None),
                    video_grid_thw=(
                        b['video_grid_thw'].to('cuda') if b.get('video_grid_thw') is not None else None),
                    max_new_tokens=max_new_tokens,
                    do_sample=(temp > 0),
                    temperature=(None if temp == 0 else temp),
                    top_p=None, top_k=None,
                    eos_token_id=model.config.eos_token_id,
                )
                pl = b['input_ids'].shape[1]
                for j in range(len(chunk)):
                    code = processor.decode(out_ids[j, pl:], skip_special_tokens=True)
                    out[i + j].append(code)
        return out

    print(f'  greedy ...', flush=True)
    greedy_codes = _gen(temp=0.0, n_per_row=1)
    print(f'  t=1.0 × {n_max_iou_samples} ...', flush=True)
    sampled_codes = _gen(temp=1.0, n_per_row=n_max_iou_samples)

    del model
    torch.cuda.empty_cache()

    out_rows: list[dict] = []
    for r, gcodes, scodes in zip(rows, greedy_codes, sampled_codes):
        # Greedy
        g_iou, _ = compute_metrics(gcodes[0], r['gt_stl'], timeout=30)
        g_iou = max(g_iou, 0)
        g_ops = find_ops(gcodes[0])
        g_ess = essential_pass(r['family'], g_ops)
        g_ess_score = 1.0 if g_ess else 0.0 if g_ess is not None else None

        # max@16
        sample_ious = []
        sample_esses = []
        for code in scodes:
            iou, _ = compute_metrics(code, r['gt_stl'], timeout=30)
            sample_ious.append(max(iou, 0))
            ops = find_ops(code)
            sample_esses.append(essential_score(r['family'], ops) or 0.0)
        out_rows.append({
            'uid': r['uid'], 'family': r['family'],
            'greedy_iou':     g_iou,
            'greedy_ess':     g_ess_score,
            'max_iou_at_16':  max(sample_ious),
            'max_ess_at_16':  max(sample_esses),
            'mean_iou_at_16': float(np.mean(sample_ious)),
            'mean_ess_at_16': float(np.mean(sample_esses)),
        })
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts',         required=True,
                    help='Run dir containing checkpoint-NNNN subdirs')
    ap.add_argument('--steps',         required=True,
                    help='Comma-separated step indices, e.g. 24000,26000,28000,30000')
    ap.add_argument('--base-model',    default='Qwen/Qwen3-VL-2B-Instruct')
    ap.add_argument('--backbone',      default='qwen3_vl')
    ap.add_argument('--out',           required=True)
    ap.add_argument('--n-samples',     type=int, default=16)
    ap.add_argument('--batch-size',    type=int, default=4)
    ap.add_argument('--pick-by',       default='joint',
                    choices=['joint', 'greedy_iou', 'max_iou_at_16', 'max_ess_at_16'])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpts)
    steps = [int(s.strip()) for s in args.steps.split(',') if s.strip()]

    print(f'[1/3] building 50-OOD sample (10 fams × 5) ...', flush=True)
    rows = _build_ood_rows()
    print(f'  {len(rows)} OOD rows from {len(set(r["family"] for r in rows))} families', flush=True)

    print(f'[2/3] processor ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model, token=os.environ.get('HF_TOKEN'),
        min_pixels=200_704, max_pixels=1_003_520)

    summary = []
    for step in steps:
        ckpt = ckpt_root / f'checkpoint-{step}'
        if not ckpt.is_dir():
            print(f'[skip] {ckpt} not found')
            continue
        print(f'[3/3] step={step} ckpt={ckpt}', flush=True)
        per_case = _run_ckpt(ckpt, processor, args.backbone, rows,
                             n_max_iou_samples=args.n_samples,
                             batch_size=args.batch_size)
        agg = {
            'step':            step,
            'greedy_iou':      float(np.mean([r['greedy_iou'] for r in per_case])),
            'greedy_ess':      float(np.mean([r['greedy_ess'] or 0.0 for r in per_case])),
            'max_iou_at_16':   float(np.mean([r['max_iou_at_16'] for r in per_case])),
            'max_ess_at_16':   float(np.mean([r['max_ess_at_16'] for r in per_case])),
            'mean_iou_at_16':  float(np.mean([r['mean_iou_at_16'] for r in per_case])),
            'mean_ess_at_16':  float(np.mean([r['mean_ess_at_16'] for r in per_case])),
        }
        summary.append(agg)
        # Save per-case
        step_dir = out_dir / f'step-{step:06d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        with open(step_dir / 'metadata.jsonl', 'w') as f:
            for r in per_case:
                f.write(json.dumps(r) + '\n')
        print(f'  step={step}: greedy_iou={agg["greedy_iou"]:.3f}  '
              f'max@16 iou={agg["max_iou_at_16"]:.3f}  ess={agg["max_ess_at_16"]:.3f}',
              flush=True)

    # Pick the winner
    if not summary:
        print('no ckpts evaluated', file=sys.stderr); sys.exit(2)
    if args.pick_by == 'joint':
        for s in summary:
            s['_score'] = 0.5 * s['max_iou_at_16'] + 0.5 * s['max_ess_at_16']
    else:
        for s in summary:
            s['_score'] = s[args.pick_by]
    best = max(summary, key=lambda s: s['_score'])

    out = {'summary': summary, 'pick_by': args.pick_by, 'best_step': best['step']}
    (out_dir / 'summary.json').write_text(json.dumps(out, indent=2))
    print('\n=== Per-ckpt summary ===')
    print(f'{"step":>6}  {"greedy_iou":>10}  {"greedy_ess":>10}  '
          f'{"max_iou@16":>10}  {"max_ess@16":>10}  {"score":>7}')
    for s in summary:
        print(f'{s["step"]:>6}  {s["greedy_iou"]:>10.3f}  {s["greedy_ess"]:>10.3f}  '
              f'{s["max_iou_at_16"]:>10.3f}  {s["max_ess_at_16"]:>10.3f}  {s["_score"]:>7.3f}')
    print(f'\nBEST (by {args.pick_by}): step={best["step"]} → '
          f'{ckpt_root}/checkpoint-{best["step"]}')


if __name__ == '__main__':
    main()
