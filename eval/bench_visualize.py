"""Visualize Cadrille bench eval results: GT vs one or more checkpoints.

For each selected sample, shows:
  - Column 0: GT composite render (4-view, from HF dataset)
  - Column 1+: Generated shape render per checkpoint (executed from .py code)

Each row = one sample. Titles show IoU and error_type.

Usage:
    # GT vs SFT vs RL, 20 rows, random sample
    python3 tools/bench_visualize.py \\
        --eval-dirs eval_outputs/bench/sft_all:SFT eval_outputs/bench/rl_all:RL \\
        --n 20 \\
        --out eval_outputs/bench/compare_sft_rl_n20.png

    # Top-20 by SFT IoU (first eval-dir is ranking reference)
    python3 tools/bench_visualize.py \\
        --eval-dirs eval_outputs/bench/sft_all:SFT eval_outputs/bench/rl_all:RL \\
        --n 20 --select top \\
        --out eval_outputs/bench/compare_top20.png

    # Only SFT, bottom-20 failures
    python3 tools/bench_visualize.py \\
        --eval-dirs eval_outputs/bench/sft_all:SFT \\
        --n 20 --select bottom \\
        --out eval_outputs/bench/sft_bottom20.png
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_LD = os.environ.get("LD_LIBRARY_PATH", "/workspace/.local/lib")

# Cell size in output image
_CELL = 128
_FONT_SIZE = 11
_TITLE_H = 30   # pixels above each cell for label


# ---------------------------------------------------------------------------
# CQ code → STL (subprocess, isolated)
# ---------------------------------------------------------------------------

_EXEC_TMPL = textwrap.dedent("""\
    import sys, io, warnings
    import cadquery as cq
    import trimesh
    import numpy as np
    show_object = lambda *a, **kw: None

    {code}

    _res = locals().get('r') or locals().get('result')
    if _res is None:
        raise ValueError('no result/r variable')
    compound = _res.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
    buf = trimesh.exchange.stl.export_stl(mesh)
    open(sys.argv[1], 'wb').write(buf)
""")


def _exec_to_stl(py_path: Path, timeout: float = 60.0) -> str | None:
    """Execute generated CadQuery code → STL. Returns temp path or None."""
    code = py_path.read_text()
    script = _EXEC_TMPL.format(code=code)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        stl_path = f.name
    env = {**os.environ, "LD_LIBRARY_PATH": _LD}
    try:
        r = subprocess.run(
            [sys.executable, "-c", script, stl_path],
            capture_output=True, timeout=timeout, env=env,
        )
        if r.returncode == 0 and Path(stl_path).stat().st_size > 100:
            return stl_path
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STL → 4-view composite PIL image (matches Cadrille render pipeline)
# ---------------------------------------------------------------------------

def _render_stl(stl_path: str) -> Image.Image | None:
    """Render STL → 4-view 2×2 composite via open3d (same as rl/dataset.py)."""
    try:
        import trimesh
        import open3d
        from PIL import ImageOps
        from common.datasets import mesh_to_image  # cadrille/dataset.py

        mesh = trimesh.load(stl_path, force="mesh")
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        mesh.apply_scale(2.0 / max(mesh.extents))
        mesh.apply_scale(0.5)
        mesh.apply_translation([0.5, 0.5, 0.5])

        o3d_mesh = open3d.geometry.TriangleMesh()
        o3d_mesh.vertices = open3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = open3d.utility.Vector3iVector(np.asarray(mesh.faces))
        o3d_mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d_mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        imgs = [
            ImageOps.expand(
                mesh_to_image(o3d_mesh, camera_distance=-0.9, front=f, img_size=_CELL // 2),
                border=2, fill="black",
            )
            for f in fronts
        ]
        combined = Image.fromarray(np.vstack((
            np.hstack((np.array(imgs[0]), np.array(imgs[1]))),
            np.hstack((np.array(imgs[2]), np.array(imgs[3]))),
        )))
        return combined.resize((_CELL, _CELL), Image.LANCZOS)
    except Exception:
        return None


def _error_cell(msg: str) -> Image.Image:
    """Gray cell with error text."""
    img = Image.new("RGB", (_CELL, _CELL), (60, 60, 60))
    draw = ImageDraw.Draw(img)
    draw.text((4, _CELL // 2 - 8), msg[:18], fill=(200, 80, 80))
    return img


# ---------------------------------------------------------------------------
# Load eval metadata
# ---------------------------------------------------------------------------

def _load_meta(eval_dir: Path) -> dict[str, dict]:
    """stem → record dict."""
    meta: dict[str, dict] = {}
    meta_path = eval_dir / "metadata.jsonl"
    if not meta_path.exists():
        return meta
    with open(meta_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                meta[r["stem"]] = r
            except Exception:
                pass
    return meta


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _select_stems(
    metas: list[dict[str, dict]],
    n: int,
    select: str,
    seed: int,
) -> list[str]:
    """Choose n stems present in ALL eval dirs."""
    # Intersection of stems across all dirs
    common = set(metas[0].keys())
    for m in metas[1:]:
        common &= set(m.keys())
    stems = sorted(common)

    ref = metas[0]  # rank by first eval-dir IoU

    if select == "random":
        random.seed(seed)
        random.shuffle(stems)
        return stems[:n]
    elif select == "top":
        stems.sort(key=lambda s: ref[s].get("iou") or -1.0, reverse=True)
        return stems[:n]
    elif select == "bottom":
        stems.sort(key=lambda s: ref[s].get("iou") or -1.0)
        return stems[:n]
    elif select == "mid":
        stems.sort(key=lambda s: ref[s].get("iou") or -1.0)
        mid = len(stems) // 2
        half = n // 2
        return stems[max(0, mid - half): mid - half + n]
    else:
        raise ValueError(f"Unknown select mode: {select}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _CELL
    ap = argparse.ArgumentParser(
        description="Visualize bench eval: GT vs checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--eval-dirs", nargs="+", required=True,
        metavar="DIR[:LABEL]",
        help="Eval output dirs, optionally with :LABEL suffix (e.g. eval_outputs/bench/sft_all:SFT)",
    )
    ap.add_argument("--n", type=int, default=20, help="Number of rows (samples)")
    ap.add_argument(
        "--select", default="random",
        choices=["random", "top", "bottom", "mid"],
        help="How to select samples (default: random)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument(
        "--hf-repo", default="Hula0401/test_bench",
        help="HF dataset repo to load GT composite PNGs",
    )
    ap.add_argument(
        "--no-hf", action="store_true",
        help="Skip GT column (don't load HF dataset)",
    )
    ap.add_argument("--cell", type=int, default=_CELL, help="Cell size in pixels")
    args = ap.parse_args()
    _CELL = args.cell

    # Parse eval dirs

    eval_entries: list[tuple[Path, str]] = []
    for entry in args.eval_dirs:
        if ":" in entry:
            d, label = entry.rsplit(":", 1)
        else:
            d, label = entry, Path(entry).name
        eval_entries.append((Path(d), label))

    # Load metadata
    metas = [_load_meta(d) for d, _ in eval_entries]
    for (d, label), meta in zip(eval_entries, metas):
        print(f"  {label}: {len(meta)} records from {d}")

    # Select stems
    selected = _select_stems(metas, args.n, args.select, args.seed)
    print(f"Selected {len(selected)} stems (mode={args.select})")

    # Load GT composite PNGs from HF
    gt_imgs: dict[str, Image.Image] = {}
    if not args.no_hf:
        print(f"Loading GT images from {args.hf_repo} ...")
        from datasets import load_dataset
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        ds = load_dataset(args.hf_repo, token=token)
        for split_name in ds:
            for row in ds[split_name]:
                if row["stem"] in selected:
                    gt_imgs[row["stem"]] = row["composite_png"].resize((_CELL, _CELL), Image.LANCZOS)
        print(f"  {len(gt_imgs)} GT images loaded")

    # Build image grid
    n_cols = (0 if args.no_hf else 1) + len(eval_entries)
    col_labels = ([] if args.no_hf else ["GT"]) + [label for _, label in eval_entries]
    n_rows = len(selected)

    title_h = _TITLE_H
    pad = 4
    total_w = n_cols * (_CELL + pad) + pad
    total_h = title_h + n_rows * (_CELL + title_h + pad) + pad
    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", _FONT_SIZE + 1)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE - 1)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for ci, lbl in enumerate(col_labels):
        x = pad + ci * (_CELL + pad) + _CELL // 2
        draw.text((x, 6), lbl, fill=(220, 220, 220), font=font, anchor="mm")

    for ri, stem in enumerate(selected):
        y_top = title_h + ri * (_CELL + title_h + pad)

        # Row label (stem + IoU from first eval dir)
        ref_rec = metas[0].get(stem, {})
        iou_str = f"IoU={ref_rec['iou']:.3f}" if ref_rec.get("iou") is not None else ref_rec.get("error_type", "?")[:12]
        short_stem = stem[-24:] if len(stem) > 24 else stem
        draw.text(
            (pad + 2, y_top + title_h // 2),
            f"{short_stem}  {iou_str}",
            fill=(180, 180, 180), font=font_sm, anchor="lm",
        )

        ci = 0

        # GT column
        if not args.no_hf:
            x = pad + ci * (_CELL + pad)
            y = y_top + title_h
            gt_img = gt_imgs.get(stem)
            if gt_img:
                canvas.paste(gt_img, (x, y))
            else:
                canvas.paste(_error_cell("GT N/A"), (x, y))
            ci += 1

        # Checkpoint columns
        for (eval_dir, label), meta in zip(eval_entries, metas):
            x = pad + ci * (_CELL + pad)
            y = y_top + title_h
            rec = meta.get(stem, {})
            py_path = eval_dir / f"{stem}.py"

            cell_img: Image.Image | None = None
            if py_path.exists() and rec.get("error_type") != "gt_exec_fail":
                stl_path = _exec_to_stl(py_path)
                if stl_path:
                    cell_img = _render_stl(stl_path)
                    try:
                        Path(stl_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            if cell_img is None:
                err = rec.get("error_type", "no_code")[:12]
                cell_img = _error_cell(err)
            else:
                cell_img = cell_img.resize((_CELL, _CELL), Image.LANCZOS)

            # Overlay IoU
            iou_val = rec.get("iou")
            if iou_val is not None:
                overlay_draw = ImageDraw.Draw(cell_img)
                overlay_draw.text(
                    (3, _CELL - 14),
                    f"{iou_val:.3f}",
                    fill=(255, 255, 100),
                    font=font_sm,
                )

            canvas.paste(cell_img, (x, y))
            ci += 1

        if (ri + 1) % 5 == 0 or ri == n_rows - 1:
            print(f"  row {ri+1}/{n_rows} done", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    print(f"\n保存到: {out_path}  ({total_w}×{total_h}px)")


if __name__ == "__main__":
    main()
