"""Write a partial benchcad train.pkl/val.pkl from STLs already on disk.

`fetch_benchcad.py` writes the canonical pkls only after all 20143 STLs
finish (~20 min). This helper lets us kick off SFT while the materializer
is still running by writing a smaller pkl from whatever subset is on disk
*right now*.

Safety: refuses to overwrite an existing pkl that has more rows than what
this run would produce, so a concurrent fetch_benchcad finishing first is
not clobbered. Use --output-suffix .partial to write to a side file
(`{split}.partial.pkl`) instead, which is always safe.

Usage:
    python -m data_prep.make_benchcad_partial_pkl
    python -m data_prep.make_benchcad_partial_pkl --root data/benchcad
    python -m data_prep.make_benchcad_partial_pkl --output-suffix .partial
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def _build_rows(split_dir: Path, split: str, min_stl_bytes: int) -> list[dict]:
    rows: list[dict] = []
    for stl in sorted(split_dir.glob('*.stl')):
        if stl.stat().st_size < min_stl_bytes:
            continue
        stem = stl.stem
        py = split_dir / f'{stem}.py'
        png = split_dir / f'{stem}_render.png'
        if not (py.exists() and png.exists()):
            continue
        rows.append({
            'uid': stem,
            'py_path': f'{split}/{stem}.py',
            'mesh_path': f'{split}/{stem}.stl',
            'png_path': f'{split}/{stem}_render.png',
            'description': 'Generate cadquery code',
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('data/benchcad'),
                    help='Benchcad data root (default: data/benchcad)')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'],
                    help='Splits to write (default: train val)')
    ap.add_argument('--min-stl-bytes', type=int, default=200,
                    help='Skip STL files smaller than this (default: 200)')
    ap.add_argument('--output-suffix', default='',
                    help="Suffix added to output filenames, e.g. '.partial' "
                         'writes train.partial.pkl. Empty (default) writes '
                         'the canonical train.pkl/val.pkl, with overwrite guard.')
    ap.add_argument('--force', action='store_true',
                    help='Allow overwriting an existing pkl that has more '
                         'rows (default: refuse to clobber a fuller pkl).')
    args = ap.parse_args()

    for split in args.splits:
        split_dir = args.root / split
        if not split_dir.is_dir():
            print(f'  skip: {split_dir} not a directory')
            continue
        rows = _build_rows(split_dir, split, args.min_stl_bytes)
        out = args.root / f'{split}{args.output_suffix}.pkl'

        if out.exists() and not args.force and not args.output_suffix:
            try:
                existing = pickle.load(out.open('rb'))
                if isinstance(existing, list) and len(existing) >= len(rows):
                    print(f'  skip: {out} already has {len(existing)} rows '
                          f'(>= our {len(rows)}). Pass --force to overwrite.')
                    continue
            except Exception:
                pass  # unreadable / different format → fall through to overwrite

        with out.open('wb') as fp:
            pickle.dump(rows, fp)
        print(f'  {out}: {len(rows)} rows')


if __name__ == '__main__':
    main()
