"""Score the 12 cad_bench_722 cases × 3 models with the alternative metrics
defined in this folder, then build an extended figure that takes the existing
visual grid (`eval_outputs/cad_bench_722/grid.png`) and appends a per-case
score table below it.

Metrics computed per (case, model):
  - iou        : volumetric IoU (re-uses the value from the original metadata.jsonl)
  - fscore@0.05: F-score at τ=0.05 on normalised meshes  (geom_metrics.fscore_at_tau)
  - dino_cos   : DINOv2-S image-image cosine on the 4-view collage
  - clip_cos   : CLIP ViT-B/32 image-image cosine on the 4-view collage
  - lpips      : LPIPS-AlexNet distance (perceptual; lower = better)
  - ssim       : SSIM on the 4-view collage

Only the IoU, F-score, DINO, LPIPS columns end up in the rendered table to
keep it readable; all six are saved in the JSON sidecar.

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
    uv run python research/3d_similarity/score_grid_cases.py --discord
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
import uuid
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
GRID_PNG  = EVAL_ROOT / 'grid.png'
OUT_PNG   = EVAL_ROOT / 'grid_with_metrics.png'
SCORES_JSON = EVAL_ROOT / 'metrics_per_case.json'

MODELS = [
    ('cadrille_rl',     'Cadrille-rl'),
    ('cadevolve_rl1',   'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',  'Qwen-zs'),
]
# Columns to show in the rendered table per model (kept compact).
TABLE_METRICS = [
    ('iou',         'IoU',  '{:.3f}', 'high'),
    ('fscore_05',   'Fs',   '{:.3f}', 'high'),
    ('dino_cos',    'DINO', '{:.3f}', 'high'),
    ('lpips',       'LP',   '{:.3f}', 'low'),
]

# Subprocess code-exec → STL on disk (mirrors render_cad_bench_722_grid.py)
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


def render_4view_pil(stl_path: str):
    from common.meshio import render_img
    out = render_img(stl_path)
    return out['video'][0]


def normalised_mesh(stl_path: str):
    """Load STL and normalise to [-1, 1]^3 — mirrors compute_iou's convention."""
    import numpy as np
    import trimesh
    m = trimesh.load(stl_path, force='mesh')
    m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
    ext = float(np.max(m.extents))
    if ext > 1e-7:
        m.apply_scale(2.0 / ext)
    return m


# ---------------------------------------------------------------------------
# Discord upload (multipart)
# ---------------------------------------------------------------------------

def post_image_to_discord(path: Path, content: str) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  no DISCORD_WEBHOOK_URL — skipping ping'); return
    import urllib.request
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s: str): body.write(s.encode())
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
      f'Content-Type: application/json\r\n\r\n{json.dumps({"content": content})}\r\n')
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
      f'Content-Type: image/png\r\n\r\n')
    body.write(path.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-bench-722-metrics/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=30).read()
        print('  posted to Discord ✓')
    except Exception as e:
        print(f'  Discord post failed: {e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-cases', type=int, default=12)
    ap.add_argument('--out',     default=str(OUT_PNG))
    ap.add_argument('--scores',  default=str(SCORES_JSON))
    ap.add_argument('--tau',     type=float, default=0.05)
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    print('Loading metadata …', flush=True)
    metas = {}
    for slug, _ in MODELS:
        with open(EVAL_ROOT / slug / 'metadata.jsonl') as f:
            metas[slug] = {(r := json.loads(line))['stem']: r for line in f
                           if line.strip()}

    common = set(metas[MODELS[0][0]])
    for slug, _ in MODELS[1:]:
        common &= set(metas[slug])
    cases = sorted(s for s in common
                   if all(metas[slug][s].get('error_type') == 'success'
                          for slug, _ in MODELS))
    cases = cases[:args.n_cases]
    print(f'  {len(cases)} cases (all 3 models exec\'d)')

    # Get GT codes
    print('Fetching GT codes from BenchCAD/cad_bench_722 …', flush=True)
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset('BenchCAD/cad_bench_722', split='train', token=token)
    gt_by_stem: dict = {}
    for row in ds:
        if row['stem'] in set(cases):
            gt_by_stem[row['stem']] = {
                'gt_code': row['gt_code'],
                'composite_png': row['composite_png'],
            }
    print(f'  got {len(gt_by_stem)} GT entries')

    # Lazy imports for metrics — geom_metrics + image_metrics live alongside
    # this script. Add this dir to sys.path so they import as top-level modules.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from geom_metrics  import fscore_at_tau
    from image_metrics import lpips_distance, ssim_score, dino_cos, clip_cos, psnr

    scores: dict = {'tau': args.tau, 'cases': {}}

    for stem in cases:
        gt_code   = gt_by_stem[stem]['gt_code']
        gt_img    = gt_by_stem[stem]['composite_png']
        gt_stl    = code_to_stl(gt_code, timeout=60)
        if gt_stl is None:
            print(f'  SKIP {stem}: GT exec failed')
            continue
        gt_mesh   = normalised_mesh(gt_stl)
        per_model = {}
        for slug, _ in MODELS:
            iou_orig = metas[slug][stem]['iou']
            py = EVAL_ROOT / slug / f'{stem}.py'
            pred_stl = code_to_stl(py.read_text())
            if pred_stl is None:
                per_model[slug] = {'iou': iou_orig, 'fscore_05': None,
                                   'dino_cos': None, 'clip_cos': None,
                                   'lpips': None, 'ssim': None, 'psnr': None,
                                   'error': 'pred exec failed'}
                continue
            pred_mesh = normalised_mesh(pred_stl)
            pred_img  = render_4view_pil(pred_stl)
            f, p, r   = fscore_at_tau(gt_mesh, pred_mesh, tau=args.tau)
            per_model[slug] = {
                'iou':       iou_orig,
                'fscore_05': f,
                'dino_cos':  dino_cos(gt_img, pred_img),
                'clip_cos':  clip_cos(gt_img, pred_img),
                'lpips':     lpips_distance(gt_img, pred_img),
                'ssim':      ssim_score(gt_img, pred_img),
                'psnr':      psnr(gt_img, pred_img),
            }
            Path(pred_stl).unlink(missing_ok=True)
            print(f'  {stem:<55} {slug:<16}  '
                  f'iou={iou_orig:.3f} fs={per_model[slug]["fscore_05"]:.3f} '
                  f'dino={per_model[slug]["dino_cos"]:.3f} '
                  f'lpips={per_model[slug]["lpips"]:.3f}', flush=True)
        Path(gt_stl).unlink(missing_ok=True)
        scores['cases'][stem] = {
            'family':     metas[MODELS[0][0]][stem]['family'],
            'difficulty': metas[MODELS[0][0]][stem]['difficulty'],
            'models':     per_model,
        }

    Path(args.scores).write_text(json.dumps(scores, indent=2))
    print(f'\nWrote {args.scores}')

    # ── Build extended figure ────────────────────────────────────────────
    print('Building extended figure …', flush=True)
    from PIL import Image, ImageDraw, ImageFont

    grid = Image.open(GRID_PNG).convert('RGB')
    GRID_W = grid.width
    LEFT_W = 130   # matches the visual grid's left label column
    PAD    = 4

    # Score table dims
    ROW_H        = 36
    COL_HEADER_H = 38
    SUB_HEADER_H = 24
    n_metrics_per_model = len(TABLE_METRICS)
    n_score_cols = len(MODELS) * n_metrics_per_model
    cell_w = (GRID_W - LEFT_W) // n_score_cols   # uniform width
    table_w = LEFT_W + cell_w * n_score_cols
    table_h = COL_HEADER_H + SUB_HEADER_H + ROW_H * len(scores['cases']) + 60  # +tail

    table = Image.new('RGB', (table_w, table_h), color=(15, 15, 15))
    draw = ImageDraw.Draw(table)
    try:
        font_big   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
        font_med   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception:
        font_big = font_med = font_small = ImageFont.load_default()

    # Top row: model header bands
    for mi, (slug, label) in enumerate(MODELS):
        x0 = LEFT_W + mi * n_metrics_per_model * cell_w
        x1 = x0 + n_metrics_per_model * cell_w
        # alternating band
        bg = (50, 50, 80) if mi % 2 == 0 else (80, 50, 50)
        draw.rectangle([x0, 0, x1, COL_HEADER_H], fill=bg)
        bbox = draw.textbbox((0, 0), label, font=font_big)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        draw.text((x0 + (n_metrics_per_model * cell_w - w) / 2,
                   (COL_HEADER_H - h) / 2 - 2), label, fill=(230, 230, 230), font=font_big)
    # Sub-header: metric short names
    for mi in range(len(MODELS)):
        for ki, (_, short, _, _) in enumerate(TABLE_METRICS):
            x0 = LEFT_W + (mi * n_metrics_per_model + ki) * cell_w
            draw.rectangle([x0, COL_HEADER_H, x0 + cell_w, COL_HEADER_H + SUB_HEADER_H],
                           fill=(30, 30, 30))
            bbox = draw.textbbox((0, 0), short, font=font_med)
            w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
            draw.text((x0 + (cell_w - w) / 2,
                       COL_HEADER_H + (SUB_HEADER_H - h) / 2 - 2),
                      short, fill=(180, 180, 180), font=font_med)

    # "case" header
    draw.rectangle([0, 0, LEFT_W, COL_HEADER_H + SUB_HEADER_H], fill=(35, 35, 35))
    draw.text((10, 10), 'case', fill=(220, 220, 220), font=font_big)

    # Per-case rows: best score per metric per row gets highlighted
    case_stems = list(scores['cases'].keys())
    for ri, stem in enumerate(case_stems):
        y0 = COL_HEADER_H + SUB_HEADER_H + ri * ROW_H
        y1 = y0 + ROW_H
        # alternating row background
        rowbg = (22, 22, 22) if ri % 2 == 0 else (28, 28, 28)
        draw.rectangle([0, y0, table_w, y1], fill=rowbg)
        # case label (left)
        short_stem = stem.replace('synth_', '').replace('dvsub_', 'dv:')[:30]
        fam_diff = f'{scores["cases"][stem]["family"][:14]}/{scores["cases"][stem]["difficulty"][:3]}'
        draw.text((6, y0 + 2),  short_stem, fill=(220, 220, 220), font=font_small)
        draw.text((6, y0 + 18), fam_diff,   fill=(140, 140, 140), font=font_small)
        # gather best per metric across models
        for ki, (key, _, fmt, direction) in enumerate(TABLE_METRICS):
            vals = [scores['cases'][stem]['models'][slug].get(key) for slug, _ in MODELS]
            valid = [v for v in vals if v is not None]
            if valid:
                best = max(valid) if direction == 'high' else min(valid)
            else:
                best = None
            for mi, (slug, _) in enumerate(MODELS):
                v = vals[mi]
                x0 = LEFT_W + (mi * n_metrics_per_model + ki) * cell_w
                if v is None:
                    s = '—'; col = (130, 130, 130); bgcol = rowbg
                else:
                    s = fmt.format(v)
                    is_best = (best is not None and abs(v - best) < 1e-9)
                    if is_best:
                        col = (255, 255, 255); bgcol = (40, 90, 40)
                    else:
                        col = (200, 200, 200); bgcol = rowbg
                draw.rectangle([x0, y0, x0 + cell_w, y1], fill=bgcol)
                bbox = draw.textbbox((0, 0), s, font=font_med)
                w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
                draw.text((x0 + (cell_w - w) / 2, y0 + (ROW_H - h) / 2 - 1),
                          s, fill=col, font=font_med)

    # Footer
    foot_y = COL_HEADER_H + SUB_HEADER_H + len(case_stems) * ROW_H + 8
    foot_text = (f'IoU ↑   Fs (F-score @ τ={args.tau}) ↑   '
                 f'DINO (DINOv2-S cosine) ↑   LP (LPIPS-AlexNet, ↓ = better)   '
                 f'green = best in row')
    draw.text((10, foot_y), foot_text, fill=(170, 170, 170), font=font_small)

    # Stack grid + table
    full_h = grid.height + table.height
    full = Image.new('RGB', (max(grid.width, table.width), full_h), color=(10, 10, 10))
    full.paste(grid,  (0, 0))
    full.paste(table, (0, grid.height))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.save(out_path, optimize=True)
    sz = out_path.stat().st_size / 1024
    print(f'\nSaved {out_path}  ({full.size[0]}×{full.size[1]}, {sz:.1f} kB)')

    if args.discord:
        msg = (f'📐 **cad_bench_722 cross-model + alternative metrics** — '
               f'12 cases × 3 models. Top: 4-view rendered comparison. '
               f'Bottom: per-case scores with **F-score@τ={args.tau}** (geom, tolerant), '
               f'**DINOv2 cos** (visual semantic), **LPIPS** (perceptual). '
               f'Green = best in row. Full report: research/3d_similarity/README.md')
        post_image_to_discord(out_path, msg)


if __name__ == '__main__':
    main()
