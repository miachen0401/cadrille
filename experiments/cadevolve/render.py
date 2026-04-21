"""CadEvolve rendering pipeline — replicates visualization_norm.py from zhemdi/CADEvolve.

8-view coordinate-colored collage (476×952 px):
  Row 0: -Z | +Z     (orthographic, parallel projection, zoom 1.7)
  Row 1: +Y | -Y
  Row 2: +X | -X
  Row 3: Iso | -Iso  (perspective, zoom 1.1)

Mesh is pre-normalized to [0,1]^3. Each orthographic view encodes the coordinate
along the viewing axis as green intensity (pure green = high, black = low).
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt

TILE = 14 * 17  # 238 px per tile
COLS, ROWS = 2, 4
W = TILE * COLS   # 476
H = TILE * ROWS   # 952

# Pure-green colormap: G-channel = coordinate in [0,1]
_greens = np.zeros((256, 3))
_greens[:, 1] = np.linspace(0, 1, 256)
CMAP_GT = plt.matplotlib.colors.ListedColormap(_greens)


def render_stl(stl_path: str) -> Image.Image:
    """Render STL file to 476×952 CadEvolve-style collage (PIL Image, RGB)."""
    mesh = pv.read(stl_path)

    # Normalize to [0,1]^3
    b = mesh.bounds
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    ext = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
    if ext < 1e-7:
        ext = 1.0
    mesh = mesh.translate([-cx, -cy, -cz]).scale(1.0 / ext).translate([0.5, 0.5, 0.5])

    def _ortho(view_fn, axis: int, negate: bool, flip: bool) -> Image.Image:
        pl = pv.Plotter(off_screen=True, window_size=(TILE * 2, TILE * 2))
        pl.background_color = 'black'
        m = mesh.copy()
        coords = m.points[:, axis]
        if negate:
            coords = -coords + 1.0
        m.point_data['s'] = coords
        pl.add_mesh(m, scalars='s', cmap=CMAP_GT, clim=[0, 1],
                    show_scalar_bar=False, lighting=False)
        view_fn(pl)
        pl.enable_parallel_projection()
        pl.camera.zoom(1.7)
        img = pl.screenshot(None, return_img=True)
        pl.close()
        pil = Image.fromarray(img).resize((TILE, TILE), Image.LANCZOS)
        return pil.transpose(Image.FLIP_LEFT_RIGHT) if flip else pil

    def _iso(negative: bool, flip: bool) -> Image.Image:
        pl = pv.Plotter(off_screen=True, window_size=(TILE * 2, TILE * 2))
        pl.background_color = 'black'
        pl.add_mesh(mesh, color='green', lighting=True)
        pl.view_isometric(negative=negative)
        pl.camera.zoom(1.1)
        img = pl.screenshot(None, return_img=True)
        pl.close()
        pil = Image.fromarray(img).resize((TILE, TILE), Image.LANCZOS)
        return pil.transpose(Image.FLIP_LEFT_RIGHT) if flip else pil

    # 8 views: -Z, +Z, +Y, -Y, +X, -X, Iso, -Iso
    views = [
        _ortho(lambda pl: pl.view_xy(negative=True),  2, True,  True),
        _ortho(lambda pl: pl.view_xy(negative=False), 2, False, False),
        _ortho(lambda pl: pl.view_xz(negative=True),  1, False, True),
        _ortho(lambda pl: pl.view_xz(negative=False), 1, True,  False),
        _ortho(lambda pl: pl.view_yz(negative=False), 0, False, True),
        _ortho(lambda pl: pl.view_yz(negative=True),  0, True,  False),
        _iso(negative=False, flip=True),
        _iso(negative=True,  flip=False),
    ]

    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    for i, img in enumerate(views):
        r, c = divmod(i, COLS)
        canvas.paste(img, (c * TILE, r * TILE))
    return canvas
