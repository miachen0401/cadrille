"""Code deep-dive on the 'high-IoU, no-ESS' pattern — where CADEvolve nails
the geometry (IoU ≥ 0.99) but produces ZERO canonical ops because it builds
the shape from raw primitives (cylinder/box/cut) instead of the BenchCAD-spec
ops (hole/polyline/chamfer/lineTo/revolve).

Posts the resulting markdown to Discord.

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_high_iou_no_ess_deepdive.py --discord
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

# Cases picked: CADEvolve IoU ≥ 0.99 AND CADEvolve ESS=False.
# Diversified across 8 distinct families.
CASES = [
    'dvsub_synth_rect_frame_000471_s4420',
    'synth_taper_pin_002522_s4420',
    'synth_standoff_004177_s4420',
    'synth_gusseted_bracket_002805_s4420',
    'synth_cruciform_001042_s4420',
    'synth_pipe_flange_001020_s4420',
    'synth_star_blank_000092_s4420',
    'synth_fan_shroud_000087_s4420',
]
EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
PRED_DIR = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3',
}
META = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
}
SLUG = {'CADEvolve v3': 'cadevolve_rl1', 'Q3VL (ours)': 'cadrille_qwen3vl_v3'}


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
        'User-Agent': 'cad-highiou-noess/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=30).read()
        return True
    except Exception as e:
        print(f'Discord failed: {e}'); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(EVAL_ROOT / 'high_iou_no_ess_deepdive.md'))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    print('Loading metadata + essential_ops + GT …', flush=True)
    metas = {}
    for label, p in META.items():
        metas[label] = {}
        for line in open(p):
            try:
                r = json.loads(line); metas[label][r['stem']] = r
            except Exception: pass

    ess = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_per_case = {slug: {c['stem']: c for c in m['per_case']}
                    for slug, m in ess['models'].items()}

    sys.path.insert(0, str(REPO / 'research/essential_ops'))
    from canonical_ops import ESSENTIAL_BY_FAMILY

    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    gt_by_stem = {row['stem']: row for row in ds}

    L = ['# High-IoU but ESS-fail — CADEvolve\'s "geometry-right, ops-wrong" pattern',
         '',
         "Eight cad_bench_722 cases where **CADEvolve v3 hits IoU ≥ 0.99 yet "
         "fails the essential-ops check** because it builds the shape with "
         "primitive cylinder/box/cut chains instead of the BenchCAD canonical "
         "ops (`hole`, `polyline`, `chamfer`, `lineTo`, `revolve`, etc).",
         '',
         "This is the head-to-head failure mode that drives the divergence "
         "between *geometric* metrics (IoU, CD) and *semantic* metrics "
         "(essential_pass, F1). The geometry is identical to GT — but the "
         "code uses an entirely different vocabulary.",
         '']

    for stem in CASES:
        gt_row = gt_by_stem.get(stem)
        if gt_row is None:
            L.append(f'## ⚠️ {stem} — not in dataset'); continue
        family = gt_row.get('family', '?')
        diff = gt_row.get('difficulty', '?')
        gt_code = gt_row.get('gt_code', '')
        spec = ESSENTIAL_BY_FAMILY.get(family) or []
        ess_set = set()
        for elem in spec:
            if isinstance(elem, str): ess_set.add(elem)
            else: ess_set.update(elem)
        spec_str = ' AND '.join(
            (e if isinstance(e, str) else '(' + ' | '.join(e) + ')') for e in spec
        )

        L.append(f'## {stem}')
        L.append(f'**Family**: `{family}` · **Difficulty**: `{diff}` · '
                 f'**Essential spec**: `{spec_str}`')
        L.append('')

        # GT
        L.append('### GT')
        L.append('```python')
        L.append(gt_code.strip())
        L.append('```')
        L.append('')

        for label in ('CADEvolve v3', 'Q3VL (ours)'):
            slug = SLUG[label]
            rec = metas[label].get(stem) or {}
            ec  = ess_per_case.get(slug, {}).get(stem, {}) or {}
            iou = rec.get('iou'); cd = rec.get('cd')
            ep = ec.get('essential_pass'); f1 = ec.get('feature_f1')
            gen_ops = ec.get('gen_ops') or []
            used = set(gen_ops)
            matched = sorted(used & ess_set)
            extra   = sorted(used - ess_set)
            missing = sorted(ess_set - used)

            iou_s = f'{iou:.3f}' if iou is not None else '—'
            cd_s  = f'{cd:.4f}' if cd is not None else '—'
            ep_s  = '✓' if ep is True else ('✗' if ep is False else '—')
            f1_s  = f'{f1:.2f}' if f1 is not None else '—'

            line = []
            for op in matched: line.append(f'`✓ {op}`')
            for op in extra:   line.append(f'`• {op}`')
            for op in missing: line.append(f'`✗ {op}`')

            L.append(f'### {label} — IoU={iou_s} · CD={cd_s} · ESS={ep_s} · F1={f1_s}')
            if line:
                L.append('**ops**: ' + ' '.join(line))
            else:
                L.append('**ops**: *(none of the canonical patterns matched)*')
            L.append('')
            py = PRED_DIR[label] / f'{stem}.py'
            if py.exists():
                code = py.read_text().strip() or '*(empty pred)*'
                L.append('```python')
                L.append(code)
                L.append('```')
            else:
                L.append('*(no .py file)*')
            L.append('')

        L.append('---')
        L.append('')

    L.append('## Pattern')
    L.append('')
    L.append('Across these 8 cases:')
    L.append('')
    L.append('- **CADEvolve uses primitives, not canonical ops.** When the spec '
             'asks for `hole`, CADEvolve writes `cylinder().cut()`. When it asks '
             'for `polyline`, CADEvolve writes a chain of `box().union(box())`. '
             'When it asks for `revolve`, CADEvolve constructs a stack of '
             '`cylinder()` discs. The geometry matches GT (IoU 0.99+) but '
             '`gen_ops ∩ essential_ops = ∅`.')
    L.append('')
    L.append('- **Implication for op-level metrics.** Essential-pass and '
             'feature-F1 are *style* metrics — they reward using the canonical '
             'op vocabulary, not just hitting the geometry. Whether that\'s '
             '"fair" depends on the downstream use case:')
    L.append('  - For *parametric editability* (the BenchCAD framing), '
             'op-level metrics matter — a designer who sees `cylinder().cut()` '
             'instead of `hole()` cannot tweak the hole diameter via '
             '`.hole(d)` without rewriting.')
    L.append('  - For *visual / manufacturing reproduction*, geometry is what '
             'matters and CADEvolve wins.')
    L.append('')
    L.append('- **Why we score op-level higher.** Cadrille-Q3VL-v3 was SFT\'d '
             'on a 100k mix containing native BenchCAD shell-style code, so it '
             'learned the canonical vocabulary. CADEvolve was trained on '
             '`kulibinai/cadquery-dataset` (private), which appears to bias '
             'toward primitive-based construction.')

    Path(args.out).write_text('\n'.join(L))
    print(f'Wrote {args.out}')

    if args.discord:
        msg = ('🧩 **High-IoU, no-ESS deep-dive** — 8 cases where CADEvolve v3 '
               'hits IoU ≥ 0.99 but fails essential-ops because it never uses '
               'the canonical ops the spec asks for.\n'
               '\n'
               'Pattern: CADEvolve writes `cylinder().cut()` where GT uses '
               '`.hole()`. Stacks of `cylinder` discs where GT uses `revolve`. '
               'Chained `box.union(box)` where GT uses `polyline`. Geometry '
               'matches; vocabulary doesn\'t.\n'
               '\n'
               'Tradeoff: visual/manufacturing reproduction → CADEvolve wins. '
               'Parametric editability → ours wins (`.hole(d)` is editable; '
               'a 12-cylinder stack is not).\n'
               '\n'
               'Full code samples in attached `.md`.')
        ok = _post(msg, Path(args.out))
        print(f'  Discord: {"sent" if ok else "FAILED"}')


if __name__ == '__main__':
    main()
