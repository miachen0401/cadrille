"""Code-level deep-dive: show CADEvolve v3's output format/template alongside GT
and our Q3VL-v3, on representative cases.

Picks 8 cases spanning:
  1. Simple win-all (hex_nut)
  2. CADEvolve dominates prismatic (cam / clevis-style)
  3. Ours dominates rotational (ball_knob)
  4. Ours dominates curved (capsule)
  5. Both excellent on a common part (gusseted_bracket / connecting_rod)
  6. CADEvolve uses loft where we miss (bevel_gear)
  7. Both struggle (impeller)
  8. Hard test (helical_gear, everyone 0%)

For each, dumps the GT + CADEvolve + Q3VL code + line counts + ops used + IoU.
Writes a long markdown and posts to Discord.

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_code_deepdive.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

CASES = [
    ('dvsub_synth_hex_nut_001573_s4420',  'simple — both near-perfect (CADEvolve 0.99, ours 0.87)'),
    ('dvsub_synth_cam_000195_s4420',      'prismatic — CADEvolve wins (0.99 vs 0.68)'),
    ('dvsub_synth_ball_knob_000825_s4420','rotational — ours wins (1.00 vs 0.27 — CADEvolve has no `.sphere()`)'),
    ('synth_capsule_002407_s4252',        'curved — ours wins (0.97 vs 0.83 — sphere/revolve gap)'),
    ('synth_clevis_003212_s4252',         'cut-and-carve — CADEvolve dominant (0.99 vs 0.41)'),
    ('synth_bevel_gear_000514_s4420',     'loft case — CADEvolve uses loft, we miss (0.85 vs 0.56)'),
    ('synth_helical_gear_000051_s4420',   'helical_gear — CADEvolve 0.87 (!), we 0.10 — sweep+helix gap'),
    ('synth_propeller_000837_s4420',      'both struggle (0.65 vs 0.25) — radial pattern is hard'),
]

EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
PRED_DIR = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3',
    'Cadrille-rl':  REPO / 'eval_outputs/repro_official/cad_bench_722_full/py',
}


def _post(content, attachment):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url: return False
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
    w('Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    w(f'--{boundary}\r\nContent-Disposition: form-data; '
      f'name="file"; filename="{attachment.name}"\r\n')
    w('Content-Type: text/markdown\r\n\r\n')
    body.write(attachment.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-deepdive/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=30).read()
        return True
    except Exception as e:
        print(f'Discord failed: {e}'); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(EVAL_ROOT / 'code_deepdive.md'))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    print('Loading metadata + essential_ops + GT …', flush=True)
    metas = {}
    for label, pdir in PRED_DIR.items():
        meta = pdir.parent / pdir.name / 'metadata.jsonl'
        if not meta.exists():
            # Cadrille-rl repro path
            meta = REPO / 'eval_outputs/repro_official/cad_bench_722_full/metadata.jsonl'
        metas[label] = {}
        with open(meta) as f:
            for line in f:
                try:
                    r = json.loads(line); metas[label][r['stem']] = r
                except Exception:
                    pass

    ess = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_per_case = {slug: {c['stem']: c for c in m['per_case']}
                    for slug, m in ess['models'].items()}
    SLUG_TO_LABEL = {
        'cadevolve_rl1':       'CADEvolve v3',
        'cadrille_qwen3vl_v3': 'Q3VL (ours)',
        'cadrille_rl_repro':   'Cadrille-rl',
    }

    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    gt_by_stem = {row['stem']: row for row in ds}

    L = ['# cad_bench_722 — code-level deep dive (CADEvolve v3 vs ours vs GT)',
         '',
         "What does CADEvolve actually output? Here are 8 representative cases "
         "showing GT (BenchCAD shell style) alongside CADEvolve v3 and our "
         "Cadrille-Q3VL-v3 — same prompt input, different code styles.",
         '',
         "**Annotations per pred:** `IoU` (geometric), `ESS` (✓/✗/—, op-spec match), "
         "`F1` (chamfer/fillet/hole), and the **ops used** color-coded "
         "(✓ matched essential, ✗ missing essential, • extra non-essential).",
         '']

    for stem, blurb in CASES:
        gt_row = gt_by_stem.get(stem)
        if gt_row is None:
            L.append(f'## ⚠️ {stem} — not found in dataset'); continue
        family = gt_row.get('family', '?')
        diff = gt_row.get('difficulty', '?')
        gt_code = gt_row.get('gt_code', '')
        L.append(f'## {stem}')
        L.append(f'**Family**: `{family}` · **Difficulty**: `{diff}` · '
                 f'**Why this case**: {blurb}')
        L.append('')

        # GT code
        L.append('### GT (BenchCAD shell style)')
        L.append('```python')
        L.append(gt_code.strip())
        L.append('```')
        L.append('')

        # Per-model
        for label, pdir in PRED_DIR.items():
            slug = {v: k for k, v in SLUG_TO_LABEL.items()}.get(label)
            slug_meta = {'CADEvolve v3': 'cadevolve_rl1',
                         'Q3VL (ours)':  'cadrille_qwen3vl_v3',
                         'Cadrille-rl':  'cadrille_rl_repro'}.get(label)
            rec = metas[label].get(stem) or {}
            ec  = ess_per_case.get(slug_meta, {}).get(stem, {}) or {}
            iou = rec.get('iou')
            cd  = rec.get('cd')
            err = rec.get('error_type', 'no pred')
            ep  = ec.get('essential_pass')
            f1  = ec.get('feature_f1')
            gen_ops = ec.get('gen_ops') or []
            gt_ops  = ec.get('gt_ops') or []

            # build ops annotation
            ess_set = set()
            from sys import path as _p
            _p.insert(0, str(REPO / 'research/essential_ops'))
            from common.essential_ops import ESSENTIAL_BY_FAMILY
            spec = ESSENTIAL_BY_FAMILY.get(family) or []
            for elem in spec:
                if isinstance(elem, str):
                    ess_set.add(elem)
                else:
                    ess_set.update(elem)
            used = set(gen_ops)
            matched = [op for op in sorted(used) if op in ess_set]
            extra   = [op for op in sorted(used) if op not in ess_set]
            missing = [op for op in sorted(ess_set - used)]

            iou_s = f'{iou:.3f}' if iou is not None else '—'
            cd_s  = f'{cd:.4f}' if cd is not None else '—'
            ep_s = '✓' if ep is True else ('✗' if ep is False else '—')
            f1_s = f'{f1:.2f}' if f1 is not None else '—'

            ops_line_parts = []
            for op in matched:
                ops_line_parts.append(f'`✓ {op}`')
            for op in extra:
                ops_line_parts.append(f'`• {op}`')
            for op in missing:
                ops_line_parts.append(f'`✗ {op}`')

            L.append(f'### {label} — IoU={iou_s} · CD={cd_s} · ESS={ep_s} · F1={f1_s} · err={err}')
            if ops_line_parts:
                L.append(f'**ops**: ' + ' '.join(ops_line_parts))
                L.append('')

            py = pdir / f'{stem}.py'
            if py.exists():
                code = py.read_text().strip()
                if not code:
                    L.append('*(empty pred)*'); L.append('')
                else:
                    L.append('```python')
                    L.append(code)
                    L.append('```')
                    L.append('')
            else:
                L.append('*(no .py file)*'); L.append('')

        L.append('---')
        L.append('')

    L.append('## Summary observations')
    L.append('')
    L.append('1. **CADEvolve\'s style: long sequential numbered Workplane chains** '
             '(`wp1`, `wp2`, …), heavy on `box`/`cylinder`/`cut`/`union`. Almost '
             'never uses `revolve`, `loft`, `polarArray`, `lineTo`, `threePointArc`. '
             'Its op vocabulary is narrower than GT but its geometry is precise.')
    L.append('')
    L.append('2. **Our (Q3VL) style: chained method calls** (`cq.Workplane().sphere().union(...).cut(...)`). '
             'Uses canonical ops (`sphere`, `revolve`, `polarArray`, `chamfer`, `fillet`) '
             'when the part needs them. Op vocabulary aligns with GT.')
    L.append('')
    L.append('3. **GT (BenchCAD shell style): hybrid** — sketches with `segment`/`arc`/`finalize`, '
             'placed via `push`/`assemble`/`face`, with explicit Workplane offsets. '
             'CADEvolve sometimes mimics this, but more often falls back to Workplane chains.')

    Path(args.out).write_text('\n'.join(L))
    print(f'Wrote {args.out}')

    if args.discord:
        # Discord post — message + .md file
        msg = ('🔬 **CADEvolve v3 — output format deep-dive (8 representative cases)**\n'
               '\n'
               'For each case: GT (BenchCAD shell style) + CADEvolve v3 + '
               'Cadrille-Q3VL-v3 (ours) + Cadrille-rl, with ops color-coded.\n'
               '\n'
               '**Spoilers**:\n'
               '• CADEvolve writes long *numbered Workplane chains* (`wp1`, `wp2`, …) — '
               'box/cylinder/cut/union dominant, almost no revolve/loft/polarArray.\n'
               '• Ours uses *chained method calls* with canonical ops '
               '(`sphere`, `revolve`, `chamfer`) when the part needs them.\n'
               '• On rotational/curved (ball_knob, capsule) we win because '
               'CADEvolve\'s vocabulary lacks `.sphere()` and `.revolve()`.\n'
               '• On prismatic/cuts (cam, clevis, bevel_gear via loft) CADEvolve\'s '
               'longer cut-and-carve chains beat us.\n'
               '\n'
               'Full code samples in attached `.md`.')
        ok = _post(msg, Path(args.out))
        print(f'  Discord: {"sent" if ok else "FAILED"}')


if __name__ == '__main__':
    main()
