#!/usr/bin/env python3
"""Pre-render STL meshes to 4-view PNG image grids.

Each .stl is rendered to a 268×268 RGB PNG saved alongside it as
  {stem}_render.png

These cached PNGs are loaded by render_img() at training time, saving ~1 s
per example vs. rendering on-the-fly.  Run this once before training; upload
the PNGs to HuggingFace alongside the STLs for fully cached training.

Usage
-----
    python tools/prerender_dataset.py --dirs data/deepcad_train_mesh data/fusion360_train_mesh
    python tools/prerender_dataset.py --dirs data/deepcad_train_mesh --workers 4
    python tools/prerender_dataset.py --pkl data/mined/combined_hard.pkl
    python tools/prerender_dataset.py --dirs data/deepcad_train_mesh --overwrite
"""

import argparse
import os
import pickle
import sys
from glob import glob
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _render_one(args):
    """Render one STL to PNG.  Returns (stl_path, status) where
    status is 'ok' | 'skip' | error message string.
    """
    stl_path, skip_existing = args
    png_path = stl_path[:-4] + '_render.png'

    if skip_existing and os.path.exists(png_path):
        return stl_path, 'skip'

    try:
        from rl.dataset import render_img
        result = render_img(stl_path)
        img = result['video'][0]
        img.save(png_path)
        return stl_path, 'ok'
    except Exception as exc:
        return stl_path, f'ERROR: {exc}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Pre-render STL meshes to 4-view PNG grids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--dirs', nargs='+', metavar='DIR',
                     help='Directories containing .stl files (recursive)')
    src.add_argument('--pkl', metavar='PATH',
                     help='Mined-examples pkl — reads gt_mesh_path from each entry')

    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel render workers (default: 1; try 4 if RAM allows)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-render even if a PNG already exists')
    args = parser.parse_args()

    # ------------------------------------------------------------------ paths
    if args.dirs:
        stl_paths = []
        for d in args.dirs:
            found = sorted(glob(os.path.join(d, '**', '*.stl'), recursive=True))
            print(f'  {d}: {len(found)} stl files')
            stl_paths.extend(found)
    else:
        with open(args.pkl, 'rb') as f:
            examples = pickle.load(f)
        stl_paths = [ex['gt_mesh_path'] for ex in examples]
        # resolve relative paths from repo root
        stl_paths = [
            p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)
            for p in stl_paths
        ]
        print(f'  {args.pkl}: {len(stl_paths)} examples')

    skip_existing = not args.overwrite
    n_existing = sum(
        1 for p in stl_paths if os.path.exists(p[:-4] + '_render.png')
    )
    if skip_existing and n_existing:
        print(f'  {n_existing}/{len(stl_paths)} already have PNGs — skipping')

    job_args = [(p, skip_existing) for p in stl_paths]

    # ------------------------------------------------------------------ run
    ok = skipped = failed = 0
    errors = []

    if args.workers == 1:
        for a in tqdm(job_args, desc='rendering', unit='mesh'):
            _, status = _render_one(a)
            if status == 'skip':
                skipped += 1
            elif status == 'ok':
                ok += 1
            else:
                failed += 1
                errors.append((a[0], status))
    else:
        ctx = __import__('multiprocessing').get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for _, status in tqdm(
                pool.imap_unordered(_render_one, job_args),
                total=len(job_args), desc='rendering', unit='mesh',
            ):
                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    ok += 1
                else:
                    failed += 1
                    errors.append(('<worker>', status))

    print(f'\nDone: {ok} rendered  |  {skipped} skipped  |  {failed} failed')
    if errors:
        print('First 5 errors:')
        for path, msg in errors[:5]:
            print(f'  {os.path.basename(path)}: {msg}')


if __name__ == '__main__':
    main()
