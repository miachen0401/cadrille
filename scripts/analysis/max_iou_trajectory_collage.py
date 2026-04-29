"""max@K trajectory collage — sister script to eval_to_discord.py greedy posts.

Reads `predictions/step-NNNNNN.max@K.jsonl` files (written by online_eval
when `max_iou_k > 0` is configured) and builds a long-form image:

    LAYOUT (per IoU bucket, one collage per file):
        row = case (sampled subset; same anchors as greedy collage)
        col 0 = GT 4-view input image (the same render the model saw)
        col 1 = GT mesh (1 iso view, rendered from GT code) — fixed reference
        col 2 = best-of-K pred mesh at first available step
        col 3 = best-of-K pred mesh at second step
        ...

Reuses `eval_to_discord.py` helpers for rendering + grid composition.

Usage:
    # On-demand build for a run (writes PNGs locally; optional Discord post):
    uv run python -m scripts.analysis.max_iou_trajectory_collage \\
        --output-dir /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-... \\
        --n-anchors 6 \\
        [--post-discord] [--steps 1000,3000,5000,7000,...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from eval_to_discord
from scripts.analysis.eval_to_discord import (  # noqa: E402
    build_grid_collage,
    render_meshes_parallel,
    render_stls_parallel,
    find_gt_stl,
    find_input_image,
    post_to_discord,
)


def list_max_iou_jsonls(pred_dir: Path) -> dict[tuple[int, int], Path]:
    """Find all step-NNNNNN.max@K.jsonl files in `pred_dir`.

    Returns `{(step, k): path}` — keying by both step and K matters because
    a predictions/ dir can hold artifacts from multiple K configs (e.g.
    step-010000.max@8.jsonl AND step-010000.max@16.jsonl from different runs);
    keying by step alone would silently drop one and let the collage mix
    incompatible candidate sets.
    """
    out: dict[tuple[int, int], Path] = {}
    pat = re.compile(r'step-(\d{6})\.max@(\d+)\.jsonl$')
    for p in pred_dir.glob('step-*.max@*.jsonl'):
        m = pat.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return dict(sorted(out.items()))


def pick_anchors(jsonl_path: Path, n_per_bucket: int, seed: int = 42
                  ) -> dict[str, list[dict]]:
    """Sample N stable anchor uids per bucket, deterministic seed."""
    import random as _random
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    by_bucket: dict[str, list[dict]] = {}
    for r in rows:
        by_bucket.setdefault(r['bucket'], []).append(r)
    rng = _random.Random(seed)
    out: dict[str, list[dict]] = {}
    for bucket, items in by_bucket.items():
        rng.shuffle(items)
        out[bucket] = items[:n_per_bucket]
    return out


def build_max_iou_collage(bucket: str,
                          anchors: list[dict],
                          steps: list[int],
                          step_to_jsonl: dict[tuple[int, int], Path],
                          k: int) -> bytes:
    """Build long collage for one bucket: rows=case, cols=GT_input, GT_mesh,
    best-of-K pred at each step. Returns PNG bytes."""
    # Resolve per-step best codes for each anchor uid
    # (one extra read per step file; cheap)
    best_by_step: dict[int, dict[str, str]] = {}  # step → uid → best_code
    for step in steps:
        path = step_to_jsonl.get((step, k))
        if path is None:
            continue
        d = best_by_step.setdefault(step, {})
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r['bucket'] == bucket:
                d[r['uid']] = r.get('best_code') or ''

    # Gather render tasks
    pred_tasks = []
    for anchor in anchors:
        uid = anchor['uid']
        for step in steps:
            code = best_by_step.get(step, {}).get(uid)
            if code:
                pred_tasks.append((f'{step}_{uid}', code, (1, 1, 1), 192))

    print(f'  [{bucket}] rendering {len(pred_tasks)} pred meshes (max@{k})...',
          flush=True)
    # render_*_parallel return {label: (png_bytes_or_None, status)}; we only
    # need the bytes (status is for diagnostic prints inside eval_to_discord).
    pred_raw = render_meshes_parallel(pred_tasks, max_workers=8)
    pred_pngs = {lbl: t[0] for lbl, t in pred_raw.items()}

    # GT mesh + GT input image
    gt_tasks, gt_imgs = [], {}
    for anchor in anchors:
        uid = anchor['uid']
        stl = find_gt_stl(uid, bucket)
        if stl:
            gt_tasks.append((uid, str(stl), (1, 1, 1), 192))
        img_bytes = find_input_image(uid, bucket)
        if img_bytes:
            gt_imgs[uid] = img_bytes
    gt_raw = render_stls_parallel(gt_tasks, max_workers=4) if gt_tasks else {}
    gt_mesh_pngs = {lbl: t[0] for lbl, t in gt_raw.items()}

    # Compose grid: rows = anchor; col 0 = GT input, col 1 = GT mesh, col 2+ = max@K per step
    rows = []
    for anchor in anchors:
        uid = anchor['uid']
        cells = [gt_imgs.get(uid), gt_mesh_pngs.get(uid)]
        for step in steps:
            cells.append(pred_pngs.get(f'{step}_{uid}'))
        # Label includes the latest max@K IoU for this case
        rows.append({
            'cells': cells,
            'label': f'{uid[:14]}  best@last={anchor.get("best_iou", 0):.2f}',
        })

    cols = ['GT input', 'GT mesh'] + [f'step {s}' for s in steps]
    return build_grid_collage(rows,
                              title=f'[{bucket}] max_iou@{k} trajectory',
                              col_titles=cols,
                              cell=192)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True,
                    help='Run output dir containing predictions/')
    ap.add_argument('--n-anchors', type=int, default=6,
                    help='cases per bucket')
    ap.add_argument('--steps', default='',
                    help='Comma-sep step list (default = all max@K jsonls found)')
    ap.add_argument('--k', type=int, default=0,
                    help='K to render (default 0 = auto-detect; if multiple K '
                         'present in predictions/, pick the highest)')
    ap.add_argument('--post-discord', action='store_true')
    ap.add_argument('--out', default='',
                    help='Local output dir (default = <output-dir>/max_iou_collages)')
    args = ap.parse_args()

    pred_dir = Path(args.output_dir) / 'predictions'
    if not pred_dir.is_dir():
        raise SystemExit(f'predictions/ not found at {pred_dir}')

    step_to_jsonl = list_max_iou_jsonls(pred_dir)   # {(step, k): path}
    if not step_to_jsonl:
        print(f'No max@K jsonls found in {pred_dir}. Run training with '
              f'`max_iou_k > 0` to generate them.')
        return

    # Pick K — explicit --k or auto from highest K present
    ks_present = sorted({k for (_step, k) in step_to_jsonl.keys()})
    if args.k:
        if args.k not in ks_present:
            raise SystemExit(f'--k={args.k} not present. Available K: {ks_present}')
        k = args.k
    else:
        k = ks_present[-1]
        if len(ks_present) > 1:
            print(f'Multiple K values present {ks_present}; using --k={k} '
                  f'(pass --k explicitly to pick another).')

    # Steps available for the selected K
    available_steps = sorted(step for (step, kk) in step_to_jsonl.keys() if kk == k)
    print(f'Found {len(available_steps)} max@{k} jsonls (steps {available_steps})')

    if args.steps:
        requested = [int(s) for s in args.steps.split(',') if s.strip()]
        steps = [s for s in requested if s in available_steps]
        missing = [s for s in requested if s not in available_steps]
        if missing:
            print(f'Warning: steps {missing} not present at K={k}; skipping.')
        if not steps:
            raise SystemExit(
                f'None of the requested steps {requested} have a max@{k} '
                f'artifact. Available steps for K={k}: {available_steps}')
    else:
        steps = available_steps

    # Use latest jsonl as anchor source (so anchors reflect latest case set)
    anchors_per_bucket = pick_anchors(step_to_jsonl[(steps[-1], k)], args.n_anchors)
    print(f'Buckets: {list(anchors_per_bucket.keys())}')

    out_dir = Path(args.out) if args.out else (Path(args.output_dir) / 'max_iou_collages')
    out_dir.mkdir(parents=True, exist_ok=True)

    collage_paths: list[Path] = []
    for bucket, anchors in anchors_per_bucket.items():
        if not anchors:
            continue
        png = build_max_iou_collage(bucket, anchors, steps, step_to_jsonl, k)
        safe = bucket.replace(' ', '_').replace('/', '_')
        out_p = out_dir / f'max@{k}_{safe}_steps{steps[0]}-{steps[-1]}.png'
        out_p.write_bytes(png)
        print(f'  wrote {out_p}  ({len(png)//1024} KB)', flush=True)
        collage_paths.append(out_p)

    if args.post_discord and collage_paths:
        msg = (f'**max_iou@{k} trajectory** — '
               f'steps {steps[0]} → {steps[-1]} '
               f'({len(steps)} ckpts × {args.n_anchors} cases per bucket)')
        post_to_discord(content=msg,
                        files=[(p.name, p.read_bytes()) for p in collage_paths])


if __name__ == '__main__':
    main()
