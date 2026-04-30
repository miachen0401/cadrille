"""Render OOD-only trajectory grid from existing predictions JSONL.

Renders predicted meshes for all OOD samples × all eval steps, then assembles
a grid with input image, GT mesh, pred meshes per step.

Used to fill in cells that the eval_to_discord watcher skipped because it
only renders 8 deterministic anchors per bucket.

Usage:
    uv run python -m scripts.analysis.render_ood_trajectory \\
        --pred-dir /ephemeral/checkpoints/sft-.../predictions \\
        --out docs/v4_ood_grid_full.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from scripts.analysis.eval_to_discord import (  # noqa: E402
    render_meshes_parallel, render_stls_parallel,
)

HOLDOUT = {'tapered_boss', 'taper_pin', 'venturi_tube', 'bucket', 'dome_cap',
           'nozzle', 'enclosure', 'waffle_plate', 'bolt', 'duct_elbow'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--cell-size', type=int, default=200)
    args = ap.parse_args()

    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    uid2png = {r['uid']: r['png_path'] for r in bc_val}

    # 1. Collect all OOD uids + per-step pred_code + iou
    pred_files = sorted(args.pred_dir.glob('step-*.jsonl'))
    pred_files = [f for f in pred_files if '.max@' not in f.name]
    ood_uids: list[str] = []
    pred_by_step: dict[int, dict[str, dict]] = {}
    for f in pred_files:
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0:
            continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        ood_rows = [r for r in rows if r.get('bucket') == 'BenchCAD val'
                    and uid2fam.get(r['uid']) in HOLDOUT]
        if not ood_uids:
            ood_uids = [r['uid'] for r in ood_rows]
        pred_by_step[step] = {r['uid']: r for r in ood_rows}

    steps = sorted(pred_by_step)
    print(f'OOD uids: {len(ood_uids)}, steps: {len(steps)}')
    for u in ood_uids:
        print(f'  {u} family={uid2fam.get(u)}')

    # 2. Render predicted meshes (uid × step) in parallel
    pred_tasks = []
    for step in steps:
        for uid in ood_uids:
            r = pred_by_step[step].get(uid)
            if not r:
                continue
            code = r.get('pred_code') or ''
            label = f'pred_{step:06d}_{uid}'
            pred_tasks.append((label, code, (180, 80, 80), args.cell_size))
    print(f'rendering {len(pred_tasks)} pred meshes ...')
    pred_results = render_meshes_parallel(pred_tasks, max_workers=args.workers)
    n_ok = sum(1 for v in pred_results.values() if v[0])
    print(f'  pred ok: {n_ok}/{len(pred_tasks)}')

    # 3. Render GT meshes (one per uid) — load STL from data/benchcad/val/
    gt_tasks = []
    for uid in ood_uids:
        stl = REPO_ROOT / 'data/benchcad/val' / f'{uid}.stl'
        gt_tasks.append((f'gt_{uid}', str(stl), (120, 180, 120), args.cell_size))
    print(f'rendering {len(gt_tasks)} GT meshes ...')
    gt_results = render_stls_parallel(gt_tasks, max_workers=args.workers)
    n_ok = sum(1 for v in gt_results.values() if v[0])
    print(f'  GT ok: {n_ok}/{len(gt_tasks)}')

    # 4. Assemble grid
    cell = args.cell_size
    pad = 6
    iou_h = 22
    label_w = 220  # left label
    top_h = 32

    in_col_x = 10  # input image col
    gt_col_x = label_w + cell + pad
    step_x_base = label_w + 2 * (cell + pad)

    n_rows = len(ood_uids)
    n_step_cols = len(steps)
    W = step_x_base + n_step_cols * (cell + pad) + 10
    H = top_h + n_rows * (cell + iou_h + pad) + 10

    canvas = Image.new('RGB', (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        font_lab = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        font_iou = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
        font_top = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font_lab = font_iou = font_top = ImageFont.load_default()

    # Top header
    draw.text((in_col_x + cell // 2 - 20, 8), 'Input', font=font_top, fill='black')
    draw.text((gt_col_x + cell // 2 - 10, 8), 'GT', font=font_top, fill='black')
    for i, step in enumerate(steps):
        x = step_x_base + i * (cell + pad) + cell // 2 - 30
        draw.text((x, 8), f'step {step // 1000}k', font=font_top, fill='black')

    # Rows
    for row, uid in enumerate(ood_uids):
        y = top_h + row * (cell + iou_h + pad)
        fam = uid2fam.get(uid, '')
        draw.text((in_col_x, y + cell + 4), fam, font=font_lab, fill='darkblue')

        # Input image (composite_png from val.pkl)
        in_path = REPO_ROOT / 'data/benchcad' / uid2png.get(uid, '')
        if in_path.exists():
            img = Image.open(in_path).convert('RGB').resize((cell, cell), Image.BICUBIC)
            canvas.paste(img, (in_col_x, y))

        # GT mesh
        gt_png, _ = gt_results.get(f'gt_{uid}', (None, ''))
        if gt_png:
            img = Image.open(__import__('io').BytesIO(gt_png)).convert('RGB')
            if img.size != (cell, cell):
                img = img.resize((cell, cell), Image.BICUBIC)
            canvas.paste(img, (gt_col_x, y))

        # Pred meshes per step
        for i, step in enumerate(steps):
            x = step_x_base + i * (cell + pad)
            r = pred_by_step[step].get(uid)
            iou = r.get('iou') if r else None
            png, status = pred_results.get(f'pred_{step:06d}_{uid}', (None, 'missing'))
            if png:
                img = Image.open(__import__('io').BytesIO(png)).convert('RGB')
                if img.size != (cell, cell):
                    img = img.resize((cell, cell), Image.BICUBIC)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x + cell, y + cell], fill=(245, 235, 235))
                draw.text((x + cell // 3, y + cell // 2), status[:8],
                          font=font_lab, fill='gray')
            iou_txt = f'IoU={iou:.2f}' if iou is not None and iou >= 0 else 'fail'
            color = ('green' if iou and iou > 0.5
                     else 'red' if iou is None or iou < 0.2
                     else 'orange')
            draw.rectangle([x, y + cell, x + cell, y + cell + iou_h], fill='white')
            draw.text((x + 4, y + cell + 4), iou_txt, font=font_iou, fill=color)

    args.out.parent.mkdir(exist_ok=True, parents=True)
    canvas.save(args.out)
    print(f'wrote {args.out} ({args.out.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
