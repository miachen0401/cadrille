"""Upload hard-mined examples (STLs + render PNGs + pkl/scores) to HuggingFace.

Packages the full-scan mining results so an A100 / Colab session can download
just the hard examples (~2-3 GB) instead of the full 100k-mesh train set.

Target repo structure:
  <repo>/
  ├── combined_hard.pkl
  ├── deepcad_hard.pkl
  ├── fusion360_hard.pkl
  ├── deepcad_hard_scores.jsonl
  ├── fusion360_hard_scores.jsonl
  ├── deepcad/          ← STLs + render PNGs for each hard DeepCAD example
  └── fusion360/        ← STLs + render PNGs for each hard Fusion360 example

Usage:
    python tools/upload_mined.py --repo hula-the-cat/cadrille-mined
    python tools/upload_mined.py --repo hula-the-cat/cadrille-mined --dry-run
    python tools/upload_mined.py --repo hula-the-cat/cadrille-mined --no-meshes  # pkl+scores only
"""

import argparse
import os
import pickle
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Upload mined hard examples to HuggingFace')
    p.add_argument('--repo',       required=True, help='HF dataset repo, e.g. hula-the-cat/cadrille-mined')
    p.add_argument('--mined-dir',  default='./data/mined',      help='Directory with pkl/scores files')
    p.add_argument('--staging-dir',default='./data/mined/upload_staging', help='Temp staging dir (hard-linked)')
    p.add_argument('--token',      default=None,  help='HF token (default: HF_TOKEN env var)')
    p.add_argument('--no-meshes',  action='store_true', help='Upload pkl+scores only, skip STL/PNG files')
    p.add_argument('--dry-run',    action='store_true', help='Print what would be uploaded, do not upload')
    return p.parse_args()


def link_or_copy(src: Path, dst: Path):
    """Hard-link if possible (same filesystem), else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    mined  = Path(args.mined_dir)
    staging = Path(args.staging_dir)

    # ── Load PKL files ──────────────────────────────────────────────────────
    pkl_files = [
        mined / 'combined_hard.pkl',
        mined / 'deepcad_hard.pkl',
        mined / 'fusion360_hard.pkl',
    ]
    scores_files = [
        mined / 'deepcad_hard_scores.jsonl',
        mined / 'fusion360_hard_scores.jsonl',
    ]

    for f in pkl_files + scores_files:
        if not f.exists():
            print(f'[warn] Missing: {f}')

    combined_pkl = mined / 'combined_hard.pkl'
    if not combined_pkl.exists():
        sys.exit(f'ERROR: {combined_pkl} not found — run tools/combine_mined.py first')

    combined = pickle.load(open(combined_pkl, 'rb'))
    print(f'combined_hard.pkl: {len(combined):,} entries')

    dc  = [e for e in combined if 'deepcad'   in e['gt_mesh_path']]
    f3  = [e for e in combined if 'fusion360' in e['gt_mesh_path']]
    print(f'  DeepCAD:   {len(dc):,}')
    print(f'  Fusion360: {len(f3):,}')

    if args.dry_run:
        print('\n[dry-run] Would upload:')
        for f in pkl_files + scores_files:
            print(f'  {f}')
        if not args.no_meshes:
            print(f'  {len(dc)} DeepCAD STLs + PNGs  → staging/deepcad/')
            print(f'  {len(f3)} Fusion360 STLs + PNGs → staging/fusion360/')
        print(f'\nTarget repo: https://huggingface.co/datasets/{args.repo}')
        return

    # ── Stage metadata files ─────────────────────────────────────────────────
    staging.mkdir(parents=True, exist_ok=True)
    for f in pkl_files + scores_files:
        if f.exists():
            link_or_copy(f, staging / f.name)
            print(f'  staged: {f.name}')

    # ── Stage STLs + PNGs ────────────────────────────────────────────────────
    if not args.no_meshes:
        n_stl = n_png = n_missing = 0
        for entry in combined:
            stl = Path(entry['gt_mesh_path'])
            ds  = 'deepcad' if 'deepcad' in str(stl) else 'fusion360'
            dst_dir = staging / ds

            if stl.exists():
                link_or_copy(stl, dst_dir / stl.name)
                n_stl += 1
            else:
                n_missing += 1

            png = stl.with_name(stl.stem + '_render.png')
            if png.exists():
                link_or_copy(png, dst_dir / png.name)
                n_png += 1

        print(f'Staged: {n_stl:,} STLs, {n_png:,} PNGs, {n_missing} missing STLs')

    # ── Upload ────────────────────────────────────────────────────────────────
    print(f'\nUploading to {args.repo} ...')
    from huggingface_hub import HfApi
    token = args.token or os.environ.get('HF_TOKEN')
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo, repo_type='dataset', exist_ok=True)
    except Exception as e:
        print(f'[warn] create_repo: {e}')

    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo,
        repo_type='dataset',
        commit_message=f'Upload {len(combined):,} hard-mined examples (full scan)',
    )
    print(f'\nDone. Dataset: https://huggingface.co/datasets/{args.repo}')


if __name__ == '__main__':
    main()
