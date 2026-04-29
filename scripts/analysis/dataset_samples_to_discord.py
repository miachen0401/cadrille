"""One-shot: sample N items from each training data source, tile into a
collage, post to Discord (one attachment per source).

Layout: 10 cells per row × ceil(N/10) rows. Cell size 120 px.

For sources with `png_path` (BenchCAD-shell parquets that ship a 4-view
input render): just load the PNG.
For text-only sources (text2cad_bench): render GT cadquery code → 1 iso PNG
in parallel (reusing the worker from eval_to_discord).

Usage:
    uv run python -m scripts.analysis.dataset_samples_to_discord
        # default: benchcad/iso/simple/text2cad = 50, recode_bench = 100, seed 42

    uv run python -m scripts.analysis.dataset_samples_to_discord \\
        --no-discord    # build PNGs locally only
"""
from __future__ import annotations

import argparse
import io
import pickle
import random
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.eval_to_discord import (
    build_grid_collage, post_to_discord,
    render_meshes_parallel,
)

DATA_ROOT = REPO_ROOT / 'data'

# (root, n_per, has_png)
SOURCES: list[tuple[str, Path, int, bool]] = [
    ('benchcad',         DATA_ROOT / 'benchcad',         50,  True),
    ('cad_iso_106',      DATA_ROOT / 'cad-iso-106',      50,  True),
    ('benchcad_simple',  DATA_ROOT / 'benchcad-simple',  50,  True),
    ('text2cad_bench',   DATA_ROOT / 'text2cad-bench',   50,  False),
    ('recode_bench',     DATA_ROOT / 'cad-recode-bench', 100, True),
]


def sample_rows(pkl_path: Path, n: int, seed: int) -> list[dict]:
    rows = pickle.load(open(pkl_path, 'rb'))
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def png_from_path(root: Path, png_rel: str) -> bytes | None:
    p = root / png_rel
    if not p.exists():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def tile_for_source(name: str, items: list[bytes | None], n_cols: int = 10,
                    cell: int = 120) -> bytes:
    rows = []
    for i in range(0, len(items), n_cols):
        chunk = items[i:i + n_cols]
        rows.append({
            'cells': chunk,
            'label': f'#{i + 1}-{i + len(chunk)}',
        })
    title = f'{name}  —  {len(items)} samples (seed=42)'
    return build_grid_collage(rows, title, cell=cell, label_w=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=6,
                    help='parallel cadquery workers for text2cad_bench renders')
    ap.add_argument('--no-discord', action='store_true')
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'experiments_log' /
                                              'dataset_samples'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attachments: list[tuple[str, bytes]] = []

    for name, root, n_per, has_png in SOURCES:
        pkl = root / 'train.pkl'
        if not pkl.exists():
            print(f'[{name}] {pkl} missing, skipping', flush=True); continue

        sampled = sample_rows(pkl, n_per, args.seed)
        print(f'[{name}] sampled {len(sampled)}', flush=True)

        if has_png:
            pngs = [png_from_path(root, r['png_path']) for r in sampled]
            n_ok = sum(1 for p in pngs if p)
            print(f'  loaded {n_ok}/{len(pngs)} PNGs', flush=True)
        else:
            # Render gt code in parallel
            tasks = [(f'{name}_{i}', r.get('code') or '', (136, 200, 255), 200)
                     for i, r in enumerate(sampled)]
            print(f'  rendering {len(tasks)} cadquery → iso ...', flush=True)
            results = render_meshes_parallel(tasks, max_workers=args.workers)
            pngs = [results.get(label, (None, '?'))[0] for label, *_ in tasks]
            n_ok = sum(1 for p in pngs if p)
            print(f'  rendered {n_ok}/{len(tasks)} ok', flush=True)

        n_cols = 10
        cell = 120
        png = tile_for_source(name, pngs, n_cols=n_cols, cell=cell)
        out_png = out_dir / f'samples_{name}_n{len(pngs)}.png'
        out_png.write_bytes(png)
        print(f'  wrote → {out_png}  ({len(png) // 1024} KB)', flush=True)
        attachments.append((f'samples_{name}_n{len(pngs)}.png', png))

    msg = (f'**📦 Dataset samples — geometry diversity check**\n\n'
           + '\n'.join(f'• `{name}`: {n_per} samples'
                       for name, _, n_per, _ in SOURCES)
           + f'\n\nseed={args.seed}, n_cols=10, cell=120 px')

    if args.no_discord:
        print('---no-discord---')
        print(msg)
        for name, _ in attachments:
            print(f'  attached: {name}')
        return

    statuses = post_to_discord(content=msg, files=attachments)
    print(f'discord POST: {statuses}', flush=True)


if __name__ == '__main__':
    main()
