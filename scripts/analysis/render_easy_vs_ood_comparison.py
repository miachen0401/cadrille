"""Render side-by-side family grids — benchcad-easy vs OOD holdout families.

Shows what kind of shapes each set teaches the model. Helps decide whether
benchcad-easy is too dissimilar to the OOD families to act as a useful
supplement (the §7 hypothesis after observing ood_enhance < ood).

Output:
  paper/figures/easy_vs_ood_render_grid.png
"""
from __future__ import annotations
import pickle
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _row_grid(families: list[str], pkl_path: str, png_dir: Path,
              n_per_fam: int, cell: int, seed: int = 42) -> Image.Image:
    """One row per family, n_per_fam example renders per row + label."""
    rows = pickle.load(open(pkl_path, 'rb'))
    rng = random.Random(seed)

    # Bucket rows by family
    by_fam: dict[str, list[dict]] = {}
    for r in rows:
        by_fam.setdefault(r.get('family'), []).append(r)

    # Build grid
    label_w = 220
    grid_w = label_w + cell * n_per_fam
    grid_h = cell * len(families)
    canvas = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    for i, fam in enumerate(families):
        y = i * cell
        # Label cell
        draw.rectangle((0, y, label_w - 1, y + cell - 1), outline='black', width=1)
        draw.text((10, y + cell // 2 - 8), fam, fill='black', font=font)

        pool = by_fam.get(fam, [])
        if not pool:
            draw.text((label_w + 10, y + cell // 2),
                      '(no rows)', fill='red', font=font)
            continue

        rng.shuffle(pool)
        for j, r in enumerate(pool[:n_per_fam]):
            png = png_dir / r['png_path']
            x = label_w + j * cell
            if not png.exists():
                draw.rectangle((x, y, x + cell - 1, y + cell - 1),
                               outline='gray', width=1)
                draw.text((x + 10, y + cell // 2),
                          'missing', fill='gray', font=font)
                continue
            img = Image.open(png).convert('RGB').resize((cell, cell), Image.LANCZOS)
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1),
                           outline='black', width=1)

    return canvas


def main() -> None:
    out_path = REPO / 'paper/figures/easy_vs_ood_render_grid.png'

    # OOD: all 10 holdout families (small set, show 4 each)
    import yaml
    holdout = yaml.safe_load(open(REPO / 'configs/sft/holdout_families.yaml'))['holdout_families']

    # Easy: pick 12 representative families (a mix of plate / volumetric)
    # so the grid stays one image. The full 86 simple_* would be too tall.
    easy_pick = [
        'simple_coil_spring',     # only volumetric coil
        'simple_hemisphere',
        'simple_hexagon_block',
        'simple_obelisk',
        'simple_propeller',
        'simple_t_solid',
        'simple_filleted_t_plate',
        'simple_disc_with_boss',
        'simple_disc_with_pegs',
        'simple_dovetail_block',
        'simple_handle_block',
        'simple_v_block',
    ]

    n_per = 4
    cell  = 200

    print(f'rendering OOD grid ({len(holdout)} families × {n_per})...', flush=True)
    g_ood  = _row_grid(holdout, str(REPO / 'data/benchcad/val.pkl'),
                       REPO / 'data/benchcad', n_per, cell, seed=42)
    print(f'rendering easy grid ({len(easy_pick)} families × {n_per})...', flush=True)
    g_easy = _row_grid(easy_pick, str(REPO / 'data/benchcad-easy/train.pkl'),
                       REPO / 'data/benchcad-easy', n_per, cell, seed=42)

    # Stack vertically with section banners
    banner_h = 36
    total_w = max(g_ood.width, g_easy.width)
    total_h = banner_h * 2 + g_ood.height + g_easy.height
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    except OSError:
        font_big = ImageFont.load_default()

    # OOD banner
    draw.rectangle((0, 0, total_w, banner_h), fill='#ffd5c2')
    draw.text((10, 6), f'§7 OOD families (10 held-out, BC val) — '
                       f'mechanical / industrial parts',
              fill='black', font=font_big)
    canvas.paste(g_ood, (0, banner_h))

    # Easy banner
    y0 = banner_h + g_ood.height
    draw.rectangle((0, y0, total_w, y0 + banner_h), fill='#c2dcff')
    draw.text((10, y0 + 6),
              f'benchcad-easy (12/86 simple_* families shown) — '
              f'plates / primitives',
              fill='black', font=font_big)
    canvas.paste(g_easy, (0, y0 + banner_h))

    canvas.save(out_path, 'PNG')
    print(f'wrote {out_path} ({out_path.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
