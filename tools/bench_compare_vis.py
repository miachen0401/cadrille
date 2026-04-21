"""Generate comparison visualization: GT | SFT | RL | CadEvolve for bench samples.

Usage:
    python3 tools/bench_compare_vis.py \
        --sft-dir eval_outputs/bench/sft_n300 \
        --rl-dir  eval_outputs/bench/rl_n300 \
        --cade-dir eval_outputs/bench/cadevolve_n300 \
        --n 20 --seed 42 \
        --out eval_outputs/bench/compare_4model_n20.png
"""
from __future__ import annotations
import argparse, json, os, random, sys, tempfile, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_LD = os.environ.get('LD_LIBRARY_PATH', '/workspace/.local/lib')

# --------------------------------------------------------------------------
# Render generated CadQuery code → PIL image via Cadrille render_img
# --------------------------------------------------------------------------
_RENDER_SCRIPT = """
import sys, io, warnings
import cadquery as cq
import trimesh, numpy as np
show_object = lambda *a, **kw: None

{code}

_r = locals().get('r') or locals().get('result')
if _r is None:
    raise ValueError('no r/result')
compound = _r.val()
verts, faces = compound.tessellate(0.001, 0.1)
mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
buf = trimesh.exchange.stl.export_stl(mesh)
open(sys.argv[1],'wb').write(buf)
"""

def _code_to_stl(code: str, timeout: float = 30.0) -> str | None:
    script = _RENDER_SCRIPT.format(code=code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    env = {**os.environ, 'LD_LIBRARY_PATH': _LD}
    try:
        r = subprocess.run([sys.executable, '-c', script, stl],
                           capture_output=True, timeout=timeout, env=env)
        if r.returncode == 0 and Path(stl).stat().st_size > 100:
            return stl
    except Exception:
        pass
    return None

def _render_stl_cadrille(stl_path: str) -> Image.Image | None:
    """Render STL with Cadrille's 4-view pipeline."""
    try:
        from eval.render import render_img  # type: ignore
        return render_img(stl_path)
    except Exception:
        return None

def _render_stl_cadevolve(stl_path: str) -> Image.Image | None:
    """Render STL with CadEvolve's 8-view pipeline."""
    try:
        from experiments.cadevolve.render import render_stl
        return render_stl(stl_path)
    except Exception:
        return None

def _code_to_img(code: str) -> Image.Image | None:
    """Render any generated CadQuery code → PIL image via PyVista (works without EGL)."""
    stl = _code_to_stl(code)
    if stl is None:
        return None
    try:
        img = _render_stl_cadevolve(stl)
    finally:
        Path(stl).unlink(missing_ok=True)
    return img


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
THUMB = 224   # thumbnail size per cell
LABEL_H = 22
BG = (30, 30, 30)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
GREEN = (80, 200, 80)
RED = (220, 80, 80)
YELLOW = (220, 200, 60)

try:
    _FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    _FONT_SM = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except Exception:
    _FONT = _FONT_SM = ImageFont.load_default()


def _thumb(img: Image.Image | None, size: int = THUMB) -> Image.Image:
    if img is None:
        t = Image.new('RGB', (size, size), (60, 60, 60))
        d = ImageDraw.Draw(t)
        d.text((size//2 - 20, size//2 - 8), 'FAIL', fill=RED, font=_FONT)
        return t
    img = img.convert('RGB')
    # Fit into size×size
    img.thumbnail((size, size), Image.LANCZOS)
    out = Image.new('RGB', (size, size), BG)
    ox = (size - img.width) // 2
    oy = (size - img.height) // 2
    out.paste(img, (ox, oy))
    return out


def _label_bar(text: str, iou: float | None, width: int) -> Image.Image:
    bar = Image.new('RGB', (width, LABEL_H), (50, 50, 50))
    d = ImageDraw.Draw(bar)
    if iou is not None:
        col = GREEN if iou >= 0.5 else (YELLOW if iou >= 0.2 else RED)
        label = f'{text}  IoU={iou:.3f}'
    else:
        col = GRAY
        label = f'{text}  FAIL'
    d.text((4, 4), label, fill=col, font=_FONT_SM)
    return bar


def make_grid(rows_data: list[dict], n_cols: int = 4) -> Image.Image:
    """
    rows_data: list of dicts with keys:
        stem, gt_img, sft_img, sft_iou, rl_img, rl_iou, cade_img, cade_iou
    """
    col_labels = ['GT (bench)', 'Cadrille-SFT', 'Cadrille-RL', 'CadEvolve']
    cell_w = THUMB
    cell_h = THUMB + LABEL_H
    header_h = 28
    total_w = cell_w * n_cols
    total_h = header_h + cell_h * len(rows_data)

    canvas = Image.new('RGB', (total_w, total_h), BG)
    d = ImageDraw.Draw(canvas)

    # Column headers
    for ci, lbl in enumerate(col_labels):
        d.text((ci * cell_w + 4, 6), lbl, fill=WHITE, font=_FONT)

    for ri, row in enumerate(rows_data):
        y0 = header_h + ri * cell_h
        cells = [
            (row['gt_img'],   None,          'GT'),
            (row['sft_img'],  row['sft_iou'], 'SFT'),
            (row['rl_img'],   row['rl_iou'],  'RL'),
            (row['cade_img'], row['cade_iou'],'CadEvolve'),
        ]
        for ci, (img, iou, tag) in enumerate(cells):
            x0 = ci * cell_w
            thumb = _thumb(img)
            canvas.paste(thumb, (x0, y0))
            bar = _label_bar(tag, iou, cell_w)
            canvas.paste(bar, (x0, y0 + THUMB))

    return canvas


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='4-model comparison visualization')
    ap.add_argument('--sft-dir',  required=True)
    ap.add_argument('--rl-dir',   required=True)
    ap.add_argument('--cade-dir', required=True)
    ap.add_argument('--n',        type=int, default=20)
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--out',      required=True)
    ap.add_argument('--hf-repo',  default='Hula0401/test_bench')
    args = ap.parse_args()

    sft_dir  = Path(args.sft_dir)
    rl_dir   = Path(args.rl_dir)
    cade_dir = Path(args.cade_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    sft_meta  = {json.loads(l)['stem']: json.loads(l) for l in open(sft_dir/'metadata.jsonl')}
    rl_meta   = {json.loads(l)['stem']: json.loads(l) for l in open(rl_dir/'metadata.jsonl')}
    cade_meta = {json.loads(l)['stem']: json.loads(l) for l in open(cade_dir/'metadata.jsonl')}

    # Find stems present in all three
    common = sorted(set(sft_meta) & set(rl_meta) & set(cade_meta))
    rng = random.Random(args.seed)
    rng.shuffle(common)
    stems = common[:args.n]
    print(f'Visualizing {len(stems)} samples from {len(common)} common stems')

    # Load HF dataset for GT images and GT code
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    ds_all = load_dataset(args.hf_repo, token=token)
    gt_lookup = {}
    for split in ds_all:
        for row in ds_all[split]:
            gt_lookup[row['stem']] = row

    rows_data = []
    for i, stem in enumerate(stems):
        print(f'  [{i+1}/{len(stems)}] {stem}', flush=True)
        row_hf = gt_lookup.get(stem)

        # GT image from HF composite_png
        gt_img = row_hf['composite_png'].convert('RGB') if row_hf else None

        # SFT: render generated code
        sft_code_path = sft_dir / f'{stem}.py'
        sft_code = sft_code_path.read_text() if sft_code_path.exists() else ''
        sft_iou  = sft_meta[stem].get('iou')
        sft_img  = _code_to_img(sft_code) if sft_code else None

        # RL: render generated code
        rl_code_path = rl_dir / f'{stem}.py'
        rl_code = rl_code_path.read_text() if rl_code_path.exists() else ''
        rl_iou  = rl_meta[stem].get('iou')
        rl_img  = _code_to_img(rl_code) if rl_code else None

        # CadEvolve: render generated code
        cade_code_path = cade_dir / f'{stem}.py'
        cade_code = cade_code_path.read_text() if cade_code_path.exists() else ''
        cade_iou  = cade_meta[stem].get('iou')
        cade_img  = _code_to_img(cade_code) if cade_code else None

        rows_data.append(dict(
            stem=stem,
            gt_img=gt_img,
            sft_img=sft_img,  sft_iou=sft_iou,
            rl_img=rl_img,    rl_iou=rl_iou,
            cade_img=cade_img, cade_iou=cade_iou,
        ))

    print('Building grid...', flush=True)
    grid = make_grid(rows_data)
    grid.save(str(out_path))
    print(f'Saved → {out_path}  ({grid.width}×{grid.height})')


if __name__ == '__main__':
    main()
