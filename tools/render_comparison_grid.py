"""Render GT vs pred side-by-side comparison grids for analysis.

For each case, shows [GT render | Pred render] with IoU and error category label.
Cases are sorted by IoU ascending (worst first) by default.

Usage
-----
  # All low-IoU dim_error cases for one combo (default: deepcad_rl_img)
  python3 tools/render_comparison_grid.py

  # Specific combo, all categories, worst 80 cases
  python3 tools/render_comparison_grid.py --combo deepcad_sft_pc --n 80

  # Only show specific categories
  python3 tools/render_comparison_grid.py --combo deepcad_rl_img --category dim_error wrong_primitive

  # IoU range filter
  python3 tools/render_comparison_grid.py --iou-max 0.5 --n 60

  # Compare SFT vs RL for the same cases
  python3 tools/render_comparison_grid.py --combo deepcad_sft_img deepcad_rl_img --n 30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

# ── category classifier (same as analyze_dim_errors.py) ──────────────────────

def _classify_code(code: str, iou: float) -> str:
    nums = [float(x) for x in re.findall(r"\b\d+\.?\d*\b", code)]
    max_num = max(nums) if nums else 0
    extrude_vals = [float(x) for x in re.findall(r"\.extrude\((-?\d+\.?\d*)", code)]
    min_extrude = min(abs(v) for v in extrude_vals) if extrude_vals else 999
    n_union    = code.count(".union(")
    n_push     = len(re.findall(r"\.push\(\[", code))
    n_extrude  = len(re.findall(r"\.extrude\(", code))
    n_segments = code.count(".segment(") + code.count(".arc(")
    n_subtract = code.count("mode='s'")
    has_sketch   = ".sketch()" in code
    has_box_prim = ".box(" in code and not has_sketch
    has_cyl_prim = ".cylinder(" in code and not has_sketch

    if iou < 0.05:                                      return "degenerate"
    if min_extrude <= 1.5 and max_num > 50 and iou < 0.15: return "degenerate"
    if has_box_prim or has_cyl_prim:                    return "wrong_primitive"
    if n_extrude == 1 and n_union == 0:
        if n_segments > 8 and iou < 0.40:              return "partial_geom"
        if n_subtract >= 2 and iou < 0.35:             return "partial_geom"
    workplanes = re.findall(r"cq\.Workplane\('(\w+)'", code)
    non_xy = sum(1 for w in workplanes if w not in ("XY", "xy"))
    if non_xy > 0 and n_union == 0 and iou < 0.40 and n_segments < 8: return "wrong_plane"
    if min_extrude < 20 and max_num > 100 and iou < 0.20 and n_union == 0: return "wrong_plane"
    if n_union >= 4 and iou < 0.45:                    return "feature_count"
    if n_push > 5 and iou < 0.45:                      return "feature_count"
    return "dim_error"


# ── category colours ──────────────────────────────────────────────────────────

_CAT_COLOR = {
    "dim_error":       (70,  130, 180),   # steel blue
    "wrong_primitive": (220, 120,  40),   # orange
    "degenerate":      (180,  50,  50),   # red
    "wrong_plane":     (140,  80, 200),   # purple
    "partial_geom":    (80,  170,  80),   # green
    "feature_count":   (200, 180,  30),   # yellow
    "unknown":         (120, 120, 120),
}


# ── tile rendering ────────────────────────────────────────────────────────────

_TILE_IMG = 268      # render image size
_GAP      = 4        # gap between GT and pred
_LABEL_H  = 52       # label strip height below images
_TILE_W   = _TILE_IMG * 2 + _GAP
_TILE_H   = _TILE_IMG + _LABEL_H
_COLS     = 5        # tiles per row
_BORDER   = 3        # coloured border thickness

_BLANK = Image.new("RGB", (_TILE_IMG, _TILE_IMG), (40, 40, 40))


def _load(path: Path) -> Image.Image:
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return _BLANK.copy()


def _draw_tile(gt_img: Image.Image, pred_img: Image.Image,
               case_id: str, iou: float, category: str,
               combo_label: str = "") -> Image.Image:
    tile = Image.new("RGB", (_TILE_W, _TILE_H), (25, 25, 30))

    # coloured border for category
    color = _CAT_COLOR.get(category, _CAT_COLOR["unknown"])
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, _TILE_W - 1, _TILE_IMG - 1], outline=color, width=_BORDER)

    # GT image (left) with thin GT label
    tile.paste(gt_img.resize((_TILE_IMG, _TILE_IMG)), (0, 0))
    draw.rectangle([0, 0, 22, 12], fill=(0, 0, 0, 180))
    draw.text((2, 1), "GT", fill=(200, 200, 200))

    # Pred image (right)
    tile.paste(pred_img.resize((_TILE_IMG, _TILE_IMG)), (_TILE_IMG + _GAP, 0))
    draw.rectangle([_TILE_IMG + _GAP, 0, _TILE_IMG + _GAP + 34, 12], fill=(0, 0, 0, 180))
    draw.text((_TILE_IMG + _GAP + 2, 1), "Pred", fill=(200, 200, 200))

    # label strip
    y0 = _TILE_IMG
    draw.rectangle([0, y0, _TILE_W, _TILE_H], fill=(18, 18, 22))

    iou_color = (
        (80, 200, 80)   if iou >= 0.90 else
        (200, 200, 80)  if iou >= 0.70 else
        (220, 120, 50)  if iou >= 0.50 else
        (220, 60,  60)
    )

    try:
        from PIL import ImageFont
        font_large = ImageFont.load_default(size=16)
        font_small = ImageFont.load_default(size=13)
    except Exception:
        font_large = font_small = None

    kw_l = {"font": font_large} if font_large else {}
    kw_s = {"font": font_small} if font_small else {}

    label = f"{case_id}   IoU={iou:.3f}"
    draw.text((5, y0 + 4), label, fill=iou_color, **kw_l)

    cat_label = category if not combo_label else f"{category} | {combo_label}"
    draw.text((5, y0 + 26), cat_label, fill=color, **kw_s)

    return tile


# ── grid assembly ─────────────────────────────────────────────────────────────

def _build_grid(tiles: list[Image.Image], cols: int = _COLS) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    pad  = 8
    W = cols * (_TILE_W + pad) + pad
    H = rows * (_TILE_H + pad) + pad
    grid = Image.new("RGB", (W, H), (12, 12, 16))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (_TILE_W + pad)
        y = pad + r * (_TILE_H + pad)
        grid.paste(tile, (x, y))
    return grid


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combo", nargs="+",
                        default=["deepcad_rl_img"],
                        help="Combo(s) to visualise (default: deepcad_rl_img)")
    parser.add_argument("--category", nargs="*",
                        help="Filter by category (default: all). "
                             "E.g. --category dim_error wrong_primitive")
    parser.add_argument("--n", type=int, default=50,
                        help="Max cases to show per combo (default: 50)")
    parser.add_argument("--iou-max", type=float, default=0.70,
                        help="Only show cases with IoU ≤ this (default: 0.70)")
    parser.add_argument("--iou-min", type=float, default=0.0,
                        help="Only show cases with IoU ≥ this (default: 0)")
    parser.add_argument("--sort", choices=["asc", "desc", "none"], default="asc",
                        help="Sort by IoU: asc=worst first (default), desc=best first")
    parser.add_argument("--cols", type=int, default=_COLS,
                        help=f"Tiles per row (default: {_COLS})")
    parser.add_argument("--out", default=None,
                        help="Output path (default: data/analysis/<combo>/comparison_grid.png)")
    args = parser.parse_args()

    analysis_dir = _REPO / "data" / "analysis"

    # dataset → mesh dir mapping
    _MESH_DIR = {
        "deepcad":   "deepcad_test_mesh",
        "fusion360": "fusion360_test_mesh",
    }

    multi_combo = len(args.combo) > 1

    for combo in args.combo:
        combo_dir = analysis_dir / combo
        meta_path = combo_dir / "metadata.jsonl"
        dataset   = "deepcad" if "deepcad" in combo else "fusion360"
        gt_dir    = _REPO / "data" / _MESH_DIR[dataset]

        if not meta_path.exists():
            print(f"SKIP {combo}: no metadata.jsonl")
            continue

        # load and filter
        rows = []
        with open(meta_path) as f:
            for line in f:
                rows.append(json.loads(line))

        rows = [r for r in rows
                if r["error_type"] == "success"
                and r["iou"] is not None
                and args.iou_min <= r["iou"] <= args.iou_max]

        # classify
        classified = []
        for r in rows:
            case_id = r["case_id"]
            iou     = float(r["iou"])
            py_path = combo_dir / f"{case_id}_pred.py"
            code    = py_path.read_text() if py_path.exists() else ""
            cat     = _classify_code(code, iou) if code else "unknown"
            classified.append((case_id, iou, cat))

        # category filter
        if args.category:
            classified = [(c, i, k) for c, i, k in classified if k in args.category]

        # sort
        if args.sort == "asc":
            classified.sort(key=lambda x: x[1])
        elif args.sort == "desc":
            classified.sort(key=lambda x: -x[1])

        # cap
        classified = classified[:args.n]

        print(f"{combo}: {len(classified)} tiles to render")

        # build tiles
        tiles = []
        missing = 0
        for case_id, iou, cat in classified:
            gt_path   = gt_dir    / f"{case_id}_render.png"
            pred_path = combo_dir / f"{case_id}_pred_render.png"
            if not gt_path.exists() or not pred_path.exists():
                missing += 1
            gt_img   = _load(gt_path)
            pred_img = _load(pred_path)
            combo_label = combo.replace("deepcad_", "").replace("fusion360_", "") if multi_combo else ""
            tile = _draw_tile(gt_img, pred_img, case_id, iou, cat, combo_label)
            tiles.append(tile)

        if missing:
            print(f"  ({missing} cases missing a render — shown as dark placeholder)")

        if not tiles:
            print(f"  No tiles to render for {combo}")
            continue

        grid = _build_grid(tiles, cols=args.cols)

        # output path
        if args.out:
            out_path = Path(args.out)
        elif multi_combo:
            out_path = analysis_dir / f"comparison_{'_vs_'.join(args.combo)}.png"
        else:
            out_path = combo_dir / "comparison_grid.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(str(out_path))
        print(f"  Saved → {out_path}  ({grid.width}×{grid.height})")


if __name__ == "__main__":
    main()
