"""Score 4 baselines on `cad_bench_722` with the essential-ops + feature-F1
metrics from https://github.com/HaozheZhang6/Cadance/pull/7.

Per case (stem, family, gt_ops):
  - gen_ops      : `find_ops(pred_code)` — regex match over OP_PATTERNS
  - essential_pass : True / False / None  (None = N/A family, no canonical spec)
  - feature_f1   : F1 over {chamfer, fillet, hole}

Per model aggregate:
  - pct_essential_pass      : True / (True + False), excludes None
  - n_essential_applicable  : True + False
  - mean_feature_f1         : over all paired cases

Sources:
  - canonical_ops.py / canonical_ops.yaml: vendored from upstream PR
  - GT ops: BenchCAD/cad_bench_722 row['ops_used'] (JSON list of strings)
  - Pred code: eval_outputs/cad_bench_722/<model>/<stem>.py

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
    uv run python research/essential_ops/score_models.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from canonical_ops import (
    ESSENTIAL_BY_FAMILY, OP_PATTERNS, FEATURE_CLASS,
    find_ops, essential_pass, feature_f1, fmt_spec,
)

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'

MODELS = [
    ('cadrille_rl',          'Cadrille-rl (broken 5.x)'),
    ('cadrille_rl_repro',    'Cadrille-rl (paper repro 4.50.3)'),
    ('cadevolve_rl1',        'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',       'Qwen2.5-VL-3B-zs'),
    ('cadrille_qwen3vl_v3',  'Cadrille-Q3VL-v3'),
]
# Override pred dir for the special "_repro" entry (it's stored under
# eval_outputs/repro_official/cad_bench_722_full/py/, not the standard layout).
MODEL_PRED_DIR = {
    'cadrille_rl_repro': REPO / 'eval_outputs' / 'repro_official' /
                         'cad_bench_722_full' / 'py',
}


def gt_ops_from_row(row: dict) -> set[str]:
    """Convert dataset's `ops_used` field (JSON-stringified list of str) +
    the GT cadquery code into a recognised-op set. We use the same
    `find_ops` over `gt_code` for symmetry — the bare ops_used list uses
    different naming (e.g. 'circle', 'extrude') than the regex namespace.
    """
    return find_ops(row.get('gt_code') or '')


def post_to_discord(content: str, attachment: Path | None = None) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  (no DISCORD_WEBHOOK_URL — skipping ping)')
        return
    if attachment is None:
        data = json.dumps({'content': content}).encode()
        req = urllib.request.Request(url, data=data,
            headers={'Content-Type': 'application/json',
                     'User-Agent': 'cad-essential-ops/1.0'})
    else:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        def w(s): body.write(s.encode())
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
        w('Content-Type: application/json\r\n\r\n')
        w(json.dumps({'content': content}) + '\r\n')
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{attachment.name}"\r\n')
        w('Content-Type: text/markdown\r\n\r\n')
        body.write(attachment.read_bytes()); w('\r\n')
        w(f'--{boundary}--\r\n')
        req = urllib.request.Request(url, data=body.getvalue(),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}',
                     'User-Agent': 'cad-essential-ops/1.0'})
    try:
        urllib.request.urlopen(req, timeout=20).read()
        print('  posted to Discord ✓')
    except Exception as e:
        print(f'  Discord post failed: {e}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',     default=str(EVAL_ROOT / 'essential_ops.json'))
    ap.add_argument('--report',  default=str(EVAL_ROOT / 'essential_ops_report.md'))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    print('Loading BenchCAD/cad_bench_722 (for gt_code + family) …', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    by_stem = {row['stem']: row for row in ds}
    print(f'  {len(by_stem)} rows', flush=True)
    n_with_essentials = sum(1 for r in ds if r['family'] in ESSENTIAL_BY_FAMILY)
    print(f'  {n_with_essentials}/{len(ds)} cases belong to families with '
          f'an essential spec; the rest are N/A', flush=True)

    out = {'n_total': len(ds),
           'n_essentials_applicable': n_with_essentials,
           'op_patterns': sorted(OP_PATTERNS.keys()),
           'feature_class': sorted(FEATURE_CLASS),
           'models': {}}

    for slug, label in MODELS:
        pred_dir = MODEL_PRED_DIR.get(slug, EVAL_ROOT / slug)
        if not pred_dir.exists():
            print(f'  {label}: dir not found at {pred_dir}, skipping')
            continue
        n_pass = 0; n_fail = 0; n_na = 0
        n_no_pred = 0
        feat_f1s = []
        per_family: dict[str, list[bool]] = defaultdict(list)
        per_case = []   # for output
        for stem, row in by_stem.items():
            py = pred_dir / f'{stem}.py'
            if not py.exists():
                n_no_pred += 1
                continue
            gen_code = py.read_text()
            gen_ops = find_ops(gen_code)
            gt_ops  = gt_ops_from_row(row)
            ep = essential_pass(row['family'], gen_ops)
            ff1 = feature_f1(gen_ops, gt_ops)
            if ep is True:  n_pass += 1; per_family[row['family']].append(True)
            elif ep is False: n_fail += 1; per_family[row['family']].append(False)
            else: n_na += 1
            feat_f1s.append(ff1)
            per_case.append({'stem': stem, 'family': row['family'],
                             'difficulty': row.get('difficulty'),
                             'gen_ops': sorted(gen_ops),
                             'gt_ops':  sorted(gt_ops),
                             'essential_pass': ep,
                             'feature_f1': round(ff1, 4)})

        n_app = n_pass + n_fail
        out['models'][slug] = {
            'label':          label,
            'n_with_pred':    len(per_case),
            'n_no_pred':      n_no_pred,
            'n_pass':         n_pass,
            'n_fail':         n_fail,
            'n_na':           n_na,
            'pct_essential_pass': (n_pass / n_app) if n_app else None,
            'mean_feature_f1':    (sum(feat_f1s) / len(feat_f1s)) if feat_f1s else None,
            'per_family_pass_rate': {
                f: round(sum(b for b in v) / len(v), 4)
                for f, v in per_family.items() if len(v) >= 4
            },
            'per_case': per_case,
        }
        print(f'  {label:<22}  n_pred={len(per_case):4}  '
              f'pass={n_pass:4} fail={n_fail:4} na={n_na:4}  '
              f'essential_pass={n_pass/n_app*100 if n_app else 0:5.1f}%  '
              f'mean_F1={sum(feat_f1s)/len(feat_f1s):.3f}',
              flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'\nWrote raw scores to {args.out}', flush=True)

    # ── Markdown report ──────────────────────────────────────────────────
    lines = ['# `cad_bench_722` — essential-ops + feature-F1 scoring',
             '',
             f'Source: per-model `<stem>.py` predictions in '
             f'`eval_outputs/cad_bench_722/<model>/`. Methodology vendored '
             f'from [HaozheZhang6/Cadance#7]'
             f'(https://github.com/HaozheZhang6/Cadance/pull/7).',
             '',
             f'- **N total**: {len(ds)}',
             f'- **N with essential spec** (essential_pass applicable): '
             f'{n_with_essentials} / {len(ds)} = {n_with_essentials/len(ds)*100:.0f}%',
             f'- **N families with spec**: {sum(1 for f in ESSENTIAL_BY_FAMILY)}',
             f'- **Op vocabulary**: {len(OP_PATTERNS)} patterns; '
             f'feature class = {{{", ".join(sorted(FEATURE_CLASS))}}}',
             '',
             '## Per-model summary',
             '',
             f'| {"model":<24} | {"n_pred":>6} | {"essential_pass":>15} | '
             f'{"feature_f1":>11} |',
             f'|{"-"*26}|{"-"*8}|{"-"*17}|{"-"*13}|']
    for slug, label in MODELS:
        d = out['models'].get(slug)
        if not d: continue
        ep = (f'{d["pct_essential_pass"]*100:.1f}%' if d['pct_essential_pass'] is not None
              else '—')
        ff = (f'{d["mean_feature_f1"]:.4f}' if d['mean_feature_f1'] is not None
              else '—')
        n_app = d['n_pass'] + d['n_fail']
        lines.append(f'| {label:<24} | {d["n_with_pred"]:>6} | '
                     f'{ep:>9} ({d["n_pass"]:>3}/{n_app:>3}) | {ff:>11} |')

    # Per-difficulty breakdown
    lines.extend(['', '## Per-difficulty (essential_pass × feature_f1)', ''])
    for slug, label in MODELS:
        d = out['models'].get(slug)
        if not d: continue
        by_diff = defaultdict(lambda: {'pass': 0, 'fail': 0, 'f1s': []})
        for c in d['per_case']:
            slot = by_diff[c['difficulty']]
            if c['essential_pass'] is True: slot['pass'] += 1
            elif c['essential_pass'] is False: slot['fail'] += 1
            slot['f1s'].append(c['feature_f1'])
        lines.append(f'\n### {label}\n')
        lines.append(f'| {"difficulty":<10} | {"n_app":>6} | {"essential":>10} | {"mean_F1":>8} |')
        lines.append(f'|{"-"*12}|{"-"*8}|{"-"*12}|{"-"*10}|')
        for diff in ('easy', 'medium', 'hard'):
            s = by_diff.get(diff)
            if not s: continue
            n_app = s['pass'] + s['fail']
            ep = f'{s["pass"]/n_app*100:.1f}%' if n_app else '—'
            f1 = f'{sum(s["f1s"])/len(s["f1s"]):.3f}' if s['f1s'] else '—'
            lines.append(f'| {diff:<10} | {n_app:>6} | {ep:>10} | {f1:>8} |')

    # Top families where Cadrille-Q3VL-v3 vs Cadrille-rl differ most
    lines.extend(['', '## Per-family pass rate — top 20 families by sample count',
                  ''])
    fam_count = defaultdict(int)
    for r in ds:
        if r['family'] in ESSENTIAL_BY_FAMILY:
            fam_count[r['family']] += 1
    top = sorted(fam_count.items(), key=lambda kv: -kv[1])[:20]
    header = '| family                     | n  | spec'
    sub_h  = '|----------------------------|----|------'
    for slug, label in MODELS:
        if slug in out['models']:
            header += f' | {label[:14]:>14}'
            sub_h  += f'|{"-"*16}'
    header += ' |'; sub_h += '|'
    lines.append(header); lines.append(sub_h)
    for fam, cnt in top:
        spec = fmt_spec(ESSENTIAL_BY_FAMILY[fam])[:60]
        row = f'| {fam:<26} | {cnt:>2} | {spec[:5]:<5}'
        for slug, label in MODELS:
            d = out['models'].get(slug, {})
            r = d.get('per_family_pass_rate', {}).get(fam)
            row += f' | {r*100:>13.1f}%' if r is not None else f' | {"—":>14}'
        row += ' |'
        lines.append(row)

    Path(args.report).write_text('\n'.join(lines))
    print(f'Wrote report to {args.report}', flush=True)

    if args.discord:
        # short Discord summary
        msg = ['📐 **cad_bench_722 — essential-ops + feature-F1**',
               '',
               '```',
               f'{"model":<24} {"n_pred":>6} {"essential":>10} {"feat_F1":>9}',
               '-' * 56]
        for slug, label in MODELS:
            d = out['models'].get(slug)
            if not d: continue
            ep = (f'{d["pct_essential_pass"]*100:.1f}%'
                  if d['pct_essential_pass'] is not None else '—')
            ff = (f'{d["mean_feature_f1"]:.4f}'
                  if d['mean_feature_f1'] is not None else '—')
            msg.append(f'{label:<24} {d["n_with_pred"]:>6} {ep:>10} {ff:>9}')
        msg.append('```')
        msg.append('')
        msg.append('**essential**: % of cases (with applicable family spec) where '
                   'the predicted code uses the canonical AND-of-OR-tuples ops. '
                   '**feat_F1**: F1 over {chamfer, fillet, hole} indicators.')
        msg.append(f'Full table at `{args.report}`.')
        post_to_discord('\n'.join(msg), attachment=Path(args.report))


if __name__ == '__main__':
    main()
