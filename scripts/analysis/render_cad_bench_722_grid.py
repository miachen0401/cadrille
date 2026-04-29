"""Build a 4×N comparison grid for the cad_bench_722 baselines.

Rows  : GT, Cadrille-rl, CADEvolve-rl1, Qwen2.5-VL-3B-zs
Cols  : N samples where all three models exec'd successfully (so each cell
        has a real mesh to render).

Each cell is a 4-view 268×268 PIL Image rendered via `common.meshio.render_img`
(the same renderer used during data prep, so styles match across rows).
The GT row uses the upstream `composite_png` directly when available.

Output:
  eval_outputs/cad_bench_722/grid.png  — long image suitable for Discord upload

Optional: --discord posts to $DISCORD_WEBHOOK_URL via multipart upload.

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
    uv run python scripts/analysis/render_cad_bench_722_grid.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
MODELS = [
    ('cadrille_rl',     'Cadrille-rl'),
    ('cadevolve_rl1',   'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',  'Qwen2.5-VL-3B-zs'),
]

# Subprocess script: exec arbitrary cadquery code → tessellate → STL on disk
_EXEC_TMPL = textwrap.dedent('''\
    import sys, io
    import cadquery as cq
    import trimesh
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get("result") or locals().get("r")
    if _r is None:
        raise ValueError("no result/r variable")
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
    if len(verts) < 4 or len(faces) < 4:
        raise ValueError("degenerate mesh")
    mesh.export(sys.argv[1])
''')

_LD = os.environ.get('LD_LIBRARY_PATH', '/workspace/.local/lib')


def code_to_stl(code: str, timeout: float = 30.0) -> Optional[str]:
    """Exec a cadquery code string in a fresh subprocess → STL path. None on fail."""
    script = _EXEC_TMPL.format(code=code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    env = {**os.environ, 'LD_LIBRARY_PATH': _LD}
    try:
        r = subprocess.run([sys.executable, '-c', script, stl],
                           capture_output=True, timeout=timeout, env=env)
        if r.returncode == 0 and Path(stl).stat().st_size > 100:
            return stl
        return None
    except Exception:
        return None


def render_4view_from_code(code: str) -> Optional['Image.Image']:
    """Pred code → 4-view 268×268 PIL Image (or None if exec fails)."""
    stl = code_to_stl(code)
    if stl is None:
        return None
    try:
        from common.meshio import render_img
        out = render_img(stl)
        return out['video'][0]
    finally:
        Path(stl).unlink(missing_ok=True)


def fail_tile(side: int = 268, msg: str = 'EXEC FAIL') -> 'Image.Image':
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (side, side), color=(40, 40, 40))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), msg, font=font)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    d.text(((side - w) / 2, (side - h) / 2), msg, fill=(255, 90, 90), font=font)
    return img


def annotate_cell(img: 'Image.Image', label: str, color=(255, 255, 255)) -> 'Image.Image':
    """Add a single-line annotation to the bottom of a cell."""
    from PIL import Image, ImageDraw, ImageFont
    side = img.width
    LABEL_H = 30
    canvas = Image.new('RGB', (side, side + LABEL_H), color=(20, 20, 20))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), label, font=font)
    w = bbox[2] - bbox[0]
    d.text(((side - w) / 2, side + 4), label, fill=color, font=font)
    return canvas


def header_strip(side: int, header_h: int, text_lines: list[str],
                 bg=(20, 20, 20), fg=(220, 220, 220)) -> 'Image.Image':
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (side, header_h), color=bg)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    y = 4
    for t in text_lines:
        bbox = d.textbbox((0, 0), t, font=font)
        w = bbox[2] - bbox[0]
        d.text(((side - w) / 2, y), t, fill=fg, font=font)
        y += 18
    return img


def left_label(width: int, height: int, text: str) -> 'Image.Image':
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (width, height), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    d.text(((width - w) / 2, (height - h) / 2), text, fill=(240, 240, 240), font=font)
    return img


def post_image_to_discord(path: Path, content: str) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  no DISCORD_WEBHOOK_URL — skipping ping'); return
    import urllib.request, mimetypes, uuid
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s: str): body.write(s.encode())
    def b(s: bytes): body.write(s)
    # text payload field
    w(f'--{boundary}\r\n')
    w('Content-Disposition: form-data; name="payload_json"\r\n')
    w('Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    # file field
    w(f'--{boundary}\r\n')
    w(f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n')
    w('Content-Type: image/png\r\n\r\n')
    b(path.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    data = body.getvalue()
    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-bench-722-grid/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=30).read()
        print('  posted to Discord ✓')
    except Exception as e:
        print(f'  Discord post failed: {e}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(EVAL_ROOT / 'grid.png'))
    ap.add_argument('--n-cases', type=int, default=12,
                    help='Cap on # of cases (cols). Default 12 = all where all 3 exec OK.')
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    print('Loading metadata …', flush=True)
    metas = {}
    for slug, _ in MODELS:
        d = {}
        with open(EVAL_ROOT / slug / 'metadata.jsonl') as f:
            for line in f:
                try: r = json.loads(line); d[r['stem']] = r
                except: pass
        metas[slug] = d

    common = set(metas[MODELS[0][0]])
    for slug, _ in MODELS[1:]:
        common &= set(metas[slug])
    cases = sorted(s for s in common
                   if all(metas[slug][s].get('error_type') == 'success'
                          for slug, _ in MODELS))
    cases = cases[:args.n_cases]
    print(f'  {len(cases)} cases (all 3 models exec\'d):')
    for s in cases:
        ious = ' '.join(f'{slug[:3]}={metas[slug][s]["iou"]:.2f}'
                        for slug, _ in MODELS)
        print(f'    {s}  fam={metas[MODELS[0][0]][s]["family"]:<22} {ious}')

    # Pre-load GT images from upstream (one HF request)
    print('Fetching GT composite_png from BenchCAD/cad_bench_722 …', flush=True)
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset('BenchCAD/cad_bench_722', split='train', token=token)
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds
                  if row['stem'] in set(cases)}
    print(f'  got {len(gt_by_stem)} GT images')

    # Render every cell
    SIDE = 268
    cells: dict[tuple[str, str], 'Image.Image'] = {}  # (row_slug_or_GT, stem) → image
    for j, stem in enumerate(cases):
        # GT
        gt = gt_by_stem.get(stem)
        if gt is not None:
            cells[('GT', stem)] = annotate_cell(gt.convert('RGB'), 'GT')
        else:
            cells[('GT', stem)] = annotate_cell(fail_tile(SIDE, 'NO GT'), 'GT')
        # Each model
        for slug, label in MODELS:
            iou = metas[slug][stem]['iou']
            py = EVAL_ROOT / slug / f'{stem}.py'
            print(f'  rendering [{slug}] {stem} (iou={iou:.3f}) ...', flush=True)
            img = None
            if py.exists():
                code = py.read_text()
                img = render_4view_from_code(code)
            if img is None:
                img = fail_tile(SIDE, 'RENDER FAIL')
            cells[(slug, stem)] = annotate_cell(img, f'iou={iou:.3f}')

    # Layout
    LEFT_W   = 130
    HEADER_H = 70
    PAD      = 4
    rows = ['GT'] + [slug for slug, _ in MODELS]
    n_cols = len(cases)
    n_rows = len(rows)
    cell_h = SIDE + 30  # render side + label
    cell_w = SIDE
    full_w = LEFT_W + n_cols * (cell_w + PAD)
    full_h = HEADER_H + n_rows * (cell_h + PAD)

    from PIL import Image
    canvas = Image.new('RGB', (full_w, full_h), color=(15, 15, 15))

    # Column headers
    for j, stem in enumerate(cases):
        m0 = metas[MODELS[0][0]][stem]
        short_stem = stem.replace('synth_', '').replace('dvsub_', 'dv:')[:30]
        head = header_strip(cell_w, HEADER_H,
                            [short_stem,
                             f'{m0["family"]} / {m0["difficulty"]}'])
        canvas.paste(head, (LEFT_W + j * (cell_w + PAD), 0))

    # Row labels + cells
    row_label_map = {'GT': 'GT'} | {slug: label for slug, label in MODELS}
    for i, slug in enumerate(rows):
        y = HEADER_H + i * (cell_h + PAD)
        canvas.paste(left_label(LEFT_W, cell_h, row_label_map[slug]), (0, y))
        for j, stem in enumerate(cases):
            x = LEFT_W + j * (cell_w + PAD)
            canvas.paste(cells[(slug, stem)], (x, y))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    sz = out_path.stat().st_size / 1024
    print(f'\nSaved {out_path}  ({canvas.size[0]}×{canvas.size[1]}, {sz:.1f} kB)')

    if args.discord:
        # Build a one-line summary for the message body
        n = len(cases)
        msg = (f'🖼️ **cad_bench_722 cross-model 4-view grid** — {n} cases where '
               f'cadrille-rl + cadevolve-rl1 + qwen2.5-vl-3b-zs all exec\'d. '
               f'rows: GT / Cadrille / CADEvolve / Qwen-zs. each cell labelled '
               f'with that model\'s IoU on the case.')
        post_image_to_discord(out_path, msg)


if __name__ == '__main__':
    main()
