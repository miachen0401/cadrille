"""Build 10x10 grids of 100 random renders per v3 training source.

Output PNG per source written to docs/source_grids_2026-04-29/. Sources are the
v3 sft_mix_weights image-modality entries (text2cad_bench_text excluded — no png).
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path('/home/ubuntu/cadrille')
OUT_DIR = ROOT / 'docs/source_grids_2026-04-29'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ('benchcad',           'data/benchcad'),
    ('cad_iso_106',        'data/cad-iso-106'),
    ('benchcad_simple',    'data/benchcad-simple'),
    ('text2cad_bench_img', 'data/text2cad-bench'),
    ('recode_bench',       'data/cad-recode-bench'),
]

TILE = 192          # tile size in px
BORDER = 4          # gap between tiles
GRID = 10           # 10x10 = 100 tiles
SEED = 42


def build_grid(label: str, source_root: Path) -> tuple[Path, int, int]:
    pkl = source_root / 'train.pkl'
    rows = pickle.load(open(pkl, 'rb'))
    n_total = len(rows)

    rng = random.Random(SEED)
    sample = rng.sample(rows, min(GRID * GRID, n_total))

    # Compose canvas
    cell = TILE + BORDER
    side = cell * GRID + BORDER
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    placeholder = Image.new('RGB', (TILE, TILE), (60, 60, 60))
    ph_draw = ImageDraw.Draw(placeholder)
    ph_draw.text((TILE // 2 - 30, TILE // 2 - 6), 'no png', fill=(180, 180, 180))

    n_with_png = 0
    n_missing = 0
    for i, row in enumerate(sample):
        gx, gy = i % GRID, i // GRID
        x0 = BORDER + gx * cell
        y0 = BORDER + gy * cell
        png_path = row.get('png_path')
        tile_img = None
        if png_path:
            p = source_root / png_path
            if p.exists():
                try:
                    img = Image.open(p).convert('RGB')
                    img = img.resize((TILE, TILE), Image.LANCZOS)
                    tile_img = img
                    n_with_png += 1
                except Exception:
                    n_missing += 1
            else:
                n_missing += 1
        else:
            n_missing += 1
        if tile_img is None:
            tile_img = placeholder
        canvas.paste(tile_img, (x0, y0))

    out_path = OUT_DIR / f'{label}_100.png'
    canvas.save(out_path, format='PNG', optimize=True)
    return out_path, n_total, n_with_png


def main():
    results: list[tuple[str, Path, int, int]] = []
    print(f'Building 10x10 grids → {OUT_DIR}\n')
    for label, rel in SOURCES:
        src = ROOT / rel
        path, n_total, n_with_png = build_grid(label, src)
        size_kb = path.stat().st_size // 1024
        print(f'  {label:>20}  n_total={n_total:>7}  n_with_png={n_with_png:>3}/100  → {path.name} ({size_kb} KB)')
        results.append((label, path, n_total, n_with_png))

    print(f'\n{len(results)} grids written.')
    return results


if __name__ == '__main__':
    main()
