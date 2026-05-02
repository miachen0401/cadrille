"""Cadrille-rl on cad_bench_722 deepdive: why is mean IoU only 0.068?

Renders for 6 illustrative cases:
- input composite_png (from HF) — what the model saw
- GT mesh 4-view (pyvista)
- Cadrille-rl pred mesh 4-view (pyvista)
- annotated with IoU (from metadata.jsonl)

Cases mix:
  • washers / spacer_ring — geometry correct, plane choice wrong → very low IoU
  • hex_nut — geometry correct + extra-features bloat the bbox
  • lathe_turned_part / dome_cap — rotation-symmetric "best cases" still only ~0.5

Output: /tmp/cadrille_rl_cad_bench_deepdive.png
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

REPO = Path('/home/hula0401/Projects/cadrille')
INPUTS_DIR = Path('/tmp/cad_bench_inputs')
INPUTS_DIR.mkdir(exist_ok=True)

CASES = [
    # (stem, family, note)
    ('dvsub_synth_washer_000043_s4420',  'washer',          'plane wrong (ZX vs XY); washer disc itself is correct'),
    ('dvsub_synth_washer_000110_s4420',  'washer',          'same plane bug; circle().circle(s).extrude() is right'),
    ('dvsub_synth_hex_nut_000143_s4420', 'hex_nut',         'hex polygon via segments + extra junk → bbox bloat'),
    ('dvsub_synth_spacer_ring_000199_s4420','spacer_ring',  'ring geometry but wrong axis'),
    ('dvsub_synth_dome_cap_000103_s4420','dome_cap',        'rotation-symmetric → best-case IoU ~0.48'),
    ('synth_lathe_turned_part_000111_s4420', 'lathe',       'top Cadrille-rl pred, IoU=0.59'),
]


def _fetch_input_png(stem):
    """Return PIL of composite_png; download from HF if not cached."""
    p = INPUTS_DIR / f'{stem}.png'
    if p.exists():
        return Image.open(p).convert('RGB')
    print(f'  fetching {stem} from HF…', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train')
    for r in ds:
        if r['stem'] == stem:
            with open(p, 'wb') as f:
                f.write(r['composite_png']['bytes'])
            with open(INPUTS_DIR / f'{stem}_gt.py', 'w') as f:
                f.write(r['gt_code'])
            return Image.open(p).convert('RGB')
    raise RuntimeError(f'stem {stem} not found in cad_bench_722')


def _exec_to_stl(code, out_stl):
    """Run cadquery in subprocess, write tessellated STL. Returns False on failure."""
    script = textwrap.dedent('''
        import sys, trimesh, cadquery as cq
        ns = {{'cq': cq, '__name__': '__main__'}}
        code = {code!r}
        code = code.replace("show_object(result)", "").replace("show_object(r)", "")
        exec(code, ns)
        obj = ns.get("result", ns.get("r"))
        if hasattr(obj, "val"): obj = obj.val()
        v, f = obj.tessellate(0.001, 0.1)
        m = trimesh.Trimesh([(p.x, p.y, p.z) for p in v], f)
        m.export(sys.argv[1])
    ''').format(code=code)
    try:
        r = subprocess.run([sys.executable, '-c', script, str(out_stl)],
                           capture_output=True, timeout=30)
        return r.returncode == 0 and Path(out_stl).stat().st_size > 100
    except Exception:
        return False


def _render_4view_pyvista(stl_path, side=268):
    import pyvista as pv
    mesh = pv.read(str(stl_path))
    b = mesh.bounds
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    ext = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
    if ext < 1e-7: ext = 1.0
    mesh = mesh.translate([-cx, -cy, -cz]).scale(1.0 / ext)
    tile = side // 2
    color = (255/255, 255/255, 136/255)
    fronts = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]
    tiles = []
    for fx, fy, fz in fronts:
        pl = pv.Plotter(off_screen=True, window_size=(tile, tile))
        pl.background_color = (0.07, 0.07, 0.07)
        pl.add_mesh(mesh, color=color, lighting=True, smooth_shading=True)
        pl.camera_position = [(fx*1.6, fy*1.6, fz*1.6), (0, 0, 0), (0, 0, 1)]
        pl.enable_parallel_projection(); pl.camera.zoom(1.4)
        arr = pl.screenshot(None, return_img=True); pl.close()
        tiles.append(arr)
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    canvas = np.vstack([top, bot])
    return Image.fromarray(canvas).convert('RGB').resize((side, side), Image.LANCZOS)


def _render_or_blank(stl_path):
    if not Path(stl_path).exists():
        return Image.new('RGB', (268, 268), 'black')
    try:
        return _render_4view_pyvista(stl_path)
    except Exception as e:
        print(f'  pyvista render failed: {e}', flush=True)
        return Image.new('RGB', (268, 268), 'black')


def main():
    # Load IoU metadata
    meta = {}
    for line in open(REPO / 'eval_outputs/repro_official/cad_bench_722_full/metadata.jsonl'):
        r = json.loads(line); meta[r['stem']] = r

    # Build per-case panels
    rows = []
    for stem, fam, note in CASES:
        print(f'rendering {stem}', flush=True)
        # 1. input composite (what the model saw)
        try:
            inp = _fetch_input_png(stem)
        except Exception as e:
            print(f'  input fetch failed: {e}'); inp = Image.new('RGB', (268,268), 'gray')
        # 2. GT mesh render
        gt_py = INPUTS_DIR / f'{stem}_gt.py'
        gt_stl = Path(tempfile.mkstemp(suffix='.stl')[1])
        if gt_py.exists():
            _exec_to_stl(gt_py.read_text(), gt_stl)
        gt_img = _render_or_blank(gt_stl)
        gt_stl.unlink(missing_ok=True)
        # 3. Cadrille-rl pred mesh render
        pred_py = REPO / 'eval_outputs/repro_official/cad_bench_722_full/py' / f'{stem}.py'
        pred_stl = Path(tempfile.mkstemp(suffix='.stl')[1])
        if pred_py.exists():
            _exec_to_stl(pred_py.read_text(), pred_stl)
        pred_img = _render_or_blank(pred_stl)
        pred_stl.unlink(missing_ok=True)
        # 4. IoU
        iou = (meta.get(stem) or {}).get('iou')
        rows.append((stem, fam, note, inp, gt_img, pred_img, iou))

    # Compose figure
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.0*n + 1.0),
                              gridspec_kw={'wspace': 0.05, 'hspace': 0.30})
    for i, (stem, fam, note, inp, gt_img, pred_img, iou) in enumerate(rows):
        for j, (img, title) in enumerate([
            (inp,      'INPUT (composite_png from HF)'),
            (gt_img,   'GT mesh (rendered)'),
            (pred_img, f'Cadrille-rl pred  IoU={iou:.3f}' if iou is not None else 'pred (failed)'),
        ]):
            ax = axes[i][j] if n > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0: ax.set_title(title, fontsize=10, pad=4)
        # Row label on the left
        left_ax = axes[i][0] if n > 1 else axes[0]
        left_ax.text(-0.08, 0.5, f'{fam}\n{stem.replace("dvsub_synth_","").replace("synth_","")[:18]}',
                     transform=left_ax.transAxes, fontsize=9,
                     ha='right', va='center', family='monospace')
        # Note below the pred
        right_ax = axes[i][2] if n > 1 else axes[2]
        right_ax.text(1.02, 0.5, note, transform=right_ax.transAxes,
                      fontsize=8.5, ha='left', va='center', wrap=True,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff8d0',
                                edgecolor='#bbb', linewidth=0.5))

    fig.suptitle('Cadrille-rl on cad_bench_722 — why is mean IoU 0.068?\n'
                 '(metric pipeline: paper-repro 4.50.3, identical to the one giving 0.915 on DeepCAD-300)',
                 fontsize=12, y=0.995)
    out = '/tmp/cadrille_rl_cad_bench_deepdive.png'
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nWrote {out}')
    return out


if __name__ == '__main__':
    main()
