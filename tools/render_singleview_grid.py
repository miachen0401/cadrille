"""Build a single-view contact sheet from 4-view DeepCAD/Fusion360 renders.

Typical usage:
  python3 tools/render_singleview_grid.py \
    --data-dir data/deepcad_test_mesh \
    --n 100 \
    --cols 10 \
    --view top_left

The input render is assumed to be the existing 2x2 grid used by Cadrille
(268x268 by default, four 134x134 tiles). This tool crops one tile from each
sample and packs the chosen view into a large contact sheet.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from rl.dataset import render_img


_VIEW_TO_INDEX = {
    "top_left": 0,
    "top_right": 1,
    "bottom_left": 2,
    "bottom_right": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}


def _find_render_path(mesh_path: Path) -> Path:
    return mesh_path.with_name(f"{mesh_path.stem}_render.png")


def _load_render(mesh_path: Path) -> Image.Image:
    render_path = _find_render_path(mesh_path)
    if render_path.exists():
        return Image.open(render_path).convert("RGB")
    return render_img(str(mesh_path))["video"][0].convert("RGB")


def _crop_view(img: Image.Image, view_index: int) -> Image.Image:
    w, h = img.size
    if w % 2 != 0 or h % 2 != 0:
        raise ValueError(f"expected 2x2 render with even dimensions, got {w}x{h}")
    tile_w = w // 2
    tile_h = h // 2
    col = view_index % 2
    row = view_index // 2
    left = col * tile_w
    top = row * tile_h
    return img.crop((left, top, left + tile_w, top + tile_h))


def _sample_meshes(data_dir: Path, n: int, seed: int, shuffle: bool) -> list[Path]:
    meshes = sorted(data_dir.glob("*.stl"))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(meshes)
    return meshes[:n]


def _resolve_mesh_paths(paths: list[str]) -> list[Path]:
    meshes = [Path(p) for p in paths]
    missing = [str(p) for p in meshes if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing mesh paths: {missing[:5]}")
    return meshes


def _build_grid(
    tiles: list[tuple[str, Image.Image]],
    cols: int,
    tile_size: int,
    label: bool,
    label_overlay: bool,
    pad: int,
    bg: tuple[int, int, int],
) -> Image.Image:
    if not tiles:
        raise ValueError("no tiles to render")
    label_h = 18 if label and not label_overlay else 0
    rows = math.ceil(len(tiles) / cols)
    width = pad + cols * (tile_size + pad)
    height = pad + rows * (tile_size + label_h + pad)
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    for idx, (stem, tile) in enumerate(tiles):
        row, col = divmod(idx, cols)
        x = pad + col * (tile_size + pad)
        y = pad + row * (tile_size + label_h + pad)
        tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        if label and label_overlay:
            tile = tile.copy()
            tile_draw = ImageDraw.Draw(tile)
            text_x = 4
            text_y = max(2, tile_size - 14)
            text_w = min(tile_size - 4, 7 * len(stem) + 6)
            tile_draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_w, tile_size - 2],
                fill=(0, 0, 0),
            )
            tile_draw.text((text_x, text_y), stem, fill=(230, 230, 230))
        canvas.paste(tile, (x, y))
        if label and not label_overlay:
            draw.text((x + 2, y + tile_size + 2), stem, fill=(210, 210, 210))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/deepcad_test_mesh"),
        help="Directory containing .stl files and optional *_render.png cache",
    )
    parser.add_argument(
        "--mesh-path",
        action="append",
        default=[],
        help="Explicit mesh path to include; repeat to build a custom mixed grid",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of samples to include")
    parser.add_argument("--cols", type=int, default=10, help="Grid columns")
    parser.add_argument(
        "--view",
        default="top_left",
        choices=sorted(_VIEW_TO_INDEX.keys()),
        help="Which quadrant of the 2x2 render to extract",
    )
    parser.add_argument("--tile-size", type=int, default=128, help="Output tile size in pixels")
    parser.add_argument("--pad", type=int, default=8, help="Padding between tiles in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before taking the first n",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Draw mesh stem on each tile",
    )
    parser.add_argument(
        "--label-overlay",
        action="store_true",
        help="Place the label inside the tile at bottom-left instead of below it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: data/analysis/<dataset>_singleview_<view>_<n>.png)",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.cols <= 0:
        raise SystemExit("--cols must be > 0")

    if args.mesh_path:
        meshes = _resolve_mesh_paths(args.mesh_path)
        dataset_name = "custom_mix"
    else:
        meshes = _sample_meshes(args.data_dir, args.n, args.seed, args.shuffle)
        if not meshes:
            raise SystemExit(f"no .stl files found in {args.data_dir}")
        dataset_name = args.data_dir.name
    out_path = args.out
    if out_path is None:
        out_path = (
            _REPO
            / "data"
            / "analysis"
            / f"{dataset_name}_singleview_{args.view}_{len(meshes)}.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    view_index = _VIEW_TO_INDEX[args.view]
    tiles: list[tuple[str, Image.Image]] = []
    failures: list[str] = []
    for mesh_path in meshes:
        try:
            render = _load_render(mesh_path)
            tiles.append((mesh_path.stem, _crop_view(render, view_index)))
        except Exception as exc:
            failures.append(f"{mesh_path.name}: {exc}")

    if not tiles:
        raise SystemExit("all renders failed")

    grid = _build_grid(
        tiles=tiles,
        cols=args.cols,
        tile_size=args.tile_size,
        label=args.label,
        label_overlay=args.label_overlay,
        pad=args.pad,
        bg=(12, 12, 16),
    )
    grid.save(out_path)

    print(f"saved {len(tiles)} tiles -> {out_path} ({grid.width}x{grid.height})")
    if failures:
        print(f"failed renders: {len(failures)}")
        for msg in failures[:10]:
            print(f"  {msg}")


if __name__ == "__main__":
    main()
