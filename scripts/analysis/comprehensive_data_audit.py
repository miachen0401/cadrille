"""Comprehensive per-source data audit + cleaning recommendation.

Covers all 6 active training sources:
  benchcad, cad_iso_106, benchcad_simple, text2cad_bench, text2cad_legacy, recode_bench

Per source produces:
  - n_total / n_unique_code (code-hash dedup)
  - lines p10/p50/p90, unique-ops p10/p50/p90
  - has_family? if yes, family count + family-level recommendation (DROP/CUT/KEEP)
  - has_image? (png_path exists)
  - per-source cleaning rule (concrete code-level criteria)
  - estimated keep count after cleaning

Output: experiments_log/comprehensive_audit/
  - summary.csv             — per-source row, all stats + recommendation
  - source_{name}.csv       — per-item flag (keep/drop with reason) for each source
  - summary.png             — plots: complexity histograms, dedup ratios, recommendation breakdown
  - decision_report.md      — readable per-source rules + final mix proposal

Usage:
    uv run python -m scripts.analysis.comprehensive_data_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT / 'data'

CADQUERY_OP_RE = re.compile(
    r'\.(box|circle|rect|polygon|polyline|ellipse|arc|line|sketch|workplane|'
    r'extrude|cut|union|intersect|loft|sweep|revolve|fillet|chamfer|shell|'
    r'mirror|rotate|translate|transformed|edges|faces|vertices|tag|sphere|'
    r'cylinder|wedge|center|moveTo|lineTo|threePointArc|spline|close|'
    r'polarArray|rarray|hLine|vLine|push)\b'
)


def fam_from_uid(uid: str) -> str:
    parts = uid.split('_')
    while parts and (parts[-1].isdigit() or re.fullmatch(r's\d+', parts[-1])):
        parts.pop()
    return '_'.join(parts) if parts else 'UNK'


def code_stats(code: str) -> tuple[int, int, str]:
    n_lines = sum(1 for l in code.split('\n') if l.strip())
    ops = set(CADQUERY_OP_RE.findall(code))
    h = hashlib.md5(code.encode()).hexdigest()[:16]
    return n_lines, len(ops), h


def load_code(root: Path, item: dict) -> str | None:
    if item.get('code'):
        return item['code']
    if 'py_path' in item:
        p = root / item['py_path']
        if p.exists():
            try: return p.read_text()
            except Exception: return None
    p = root / 'cadquery' / f"{item['uid']}.py"
    if p.exists():
        try: return p.read_text()
        except Exception: return None
    return None


def has_image(root: Path, item: dict) -> bool:
    if item.get('png_path'):
        return (root / item['png_path']).exists()
    # text2cad legacy: check {root}/train/{uid}_render.png
    return (root / 'train' / f"{item['uid']}_render.png").exists()


# ─── per-source cleaning rules ────────────────────────────────────────────────

def rule_benchcad(item, code, fam, fams_grid_rec) -> tuple[str, str]:
    """benchcad: family-grid (P3 audit) + universal trivial filter."""
    if fam in fams_grid_rec.get('benchcad', {}).get('DROP', []):
        return 'DROP', 'family_drop'
    nl, no, _ = code_stats(code) if code else (0, 0, '')
    if nl < 5 or no < 3:
        return 'DROP', 'trivial_code'
    if fam in fams_grid_rec.get('benchcad', {}).get('CUT', []):
        return 'CUT', 'family_cut'
    return 'KEEP', 'ok'


def rule_iso(item, code, fam, fams_grid_rec) -> tuple[str, str]:
    if fam in fams_grid_rec.get('cad_iso_106', {}).get('DROP', []):
        return 'DROP', 'family_drop'
    nl, no, _ = code_stats(code) if code else (0, 0, '')
    if nl < 5 or no < 3:
        return 'DROP', 'trivial_code'
    if fam in fams_grid_rec.get('cad_iso_106', {}).get('CUT', []):
        return 'CUT', 'family_cut'
    return 'KEEP', 'ok'


def rule_simple(item, code, fam, fams_grid_rec) -> tuple[str, str]:
    if fam in fams_grid_rec.get('benchcad_simple', {}).get('DROP', []):
        return 'DROP', 'family_drop'
    nl, no, _ = code_stats(code) if code else (0, 0, '')
    if nl < 5 or no < 3:
        return 'DROP', 'trivial_code'
    if fam in fams_grid_rec.get('benchcad_simple', {}).get('CUT', []):
        return 'CUT', 'family_cut'
    return 'KEEP', 'ok'


def rule_text2cad(item, code, fam, fams_grid_rec) -> tuple[str, str]:
    """text2cad (no family): code-level filter only + dedup handled at source level."""
    nl, no, _ = code_stats(code) if code else (0, 0, '')
    if nl < 5 or no < 3:
        return 'DROP', 'trivial_code'
    if no < 4:
        return 'CUT', 'low_op_variety'
    return 'KEEP', 'ok'


def rule_recode(item, code, fam, fams_grid_rec) -> tuple[str, str]:
    """recode_bench (no family, most diverse): code-level only."""
    nl, no, _ = code_stats(code) if code else (0, 0, '')
    if nl < 5 or no < 3:
        return 'DROP', 'trivial_code'
    return 'KEEP', 'ok'


SOURCES = [
    ('benchcad',         DATA_ROOT / 'benchcad',         True,  rule_benchcad),
    ('cad_iso_106',      DATA_ROOT / 'cad-iso-106',      False, rule_iso),
    ('benchcad_simple',  DATA_ROOT / 'benchcad-simple',  False, rule_simple),
    ('text2cad_bench',   DATA_ROOT / 'text2cad-bench',   None,  rule_text2cad),  # NO family
    ('text2cad_legacy',  DATA_ROOT / 'text2cad',         None,  rule_text2cad),
    ('recode_bench',     DATA_ROOT / 'cad-recode-bench', None,  rule_recode),
]


def load_family_grid_recommendations(csv_path: Path) -> dict[str, dict[str, set]]:
    """{source: {DROP: {fam, ...}, CUT: {fam, ...}}}"""
    out: dict[str, dict[str, set]] = defaultdict(lambda: {'DROP': set(), 'CUT': set()})
    if not csv_path.exists(): return out
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rec = r.get('recommendation', '')
            if rec in ('DROP', 'CUT'):
                out[r['source']][rec].add(r['family'])
    return out


def audit_source(name: str, root: Path, has_family_field: bool | None,
                 rule_fn, fams_grid_rec: dict, out_dir: Path) -> dict:
    pkl = root / 'train.pkl'
    if not pkl.exists():
        return {'source': name, 'error': f'{pkl} missing'}
    rows = pickle.load(open(pkl, 'rb'))
    n_total = len(rows)

    # Per-item analysis
    per_item: list[dict] = []
    n_lines_list, n_ops_list = [], []
    code_hashes = Counter()
    n_no_code = 0; n_with_image = 0
    fam_counts = Counter()
    rec_counts = Counter()
    reason_counts = Counter()

    for r in rows:
        code = load_code(root, r)
        if code:
            nl, no, h = code_stats(code)
            n_lines_list.append(nl); n_ops_list.append(no)
            code_hashes[h] += 1
        else:
            n_no_code += 1
            nl, no, h = 0, 0, ''

        if has_image(root, r):
            n_with_image += 1

        # family
        if has_family_field is True and r.get('family'):
            fam = r['family']
        elif has_family_field is False:
            fam = fam_from_uid(r['uid'])
        else:
            fam = ''
        if fam: fam_counts[fam] += 1

        rec, reason = rule_fn(r, code, fam, fams_grid_rec)
        rec_counts[rec] += 1
        reason_counts[reason] += 1
        per_item.append({
            'uid': r['uid'], 'family': fam, 'lines': nl, 'ops': no,
            'code_hash': h, 'recommendation': rec, 'reason': reason,
        })

    # Save per-item CSV
    pi_csv = out_dir / f'source_{name}.csv'
    with pi_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['uid', 'family', 'lines', 'ops',
                                           'code_hash', 'recommendation', 'reason'])
        w.writeheader()
        for r in per_item: w.writerow(r)

    n_unique_codes = len(code_hashes)
    n_dup = n_total - n_unique_codes
    most_common_dup = code_hashes.most_common(5)

    return {
        'source':        name,
        'n_total':       n_total,
        'n_unique_code': n_unique_codes,
        'n_dup':         n_dup,
        'dup_rate_pct':  round(100 * n_dup / n_total, 1) if n_total else 0,
        'n_no_code':     n_no_code,
        'n_with_image':  n_with_image,
        'has_family':    has_family_field,
        'n_families':    len(fam_counts),
        'lines_p10':     percentile(n_lines_list, 10),
        'lines_p50':     percentile(n_lines_list, 50),
        'lines_p90':     percentile(n_lines_list, 90),
        'ops_p10':       percentile(n_ops_list, 10),
        'ops_p50':       percentile(n_ops_list, 50),
        'ops_p90':       percentile(n_ops_list, 90),
        'rec_DROP':      rec_counts.get('DROP', 0),
        'rec_CUT':       rec_counts.get('CUT', 0),
        'rec_KEEP':      rec_counts.get('KEEP', 0),
        'reason_breakdown': dict(reason_counts),
        'top5_duplicates': most_common_dup,
    }


def percentile(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(REPO_ROOT / 'experiments_log' /
                                          'comprehensive_audit'))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fams_grid_rec = load_family_grid_recommendations(
        REPO_ROOT / 'experiments_log/family_grid_summary.csv')

    rows = []
    for name, root, has_fam, rule in SOURCES:
        print(f'\n=== {name} ===', flush=True)
        r = audit_source(name, root, has_fam, rule, fams_grid_rec, out_dir)
        if 'error' not in r:
            n = r['n_total']
            kp = r['rec_KEEP']; ct = r['rec_CUT']; dp = r['rec_DROP']
            print(f'  n={n}  unique_code={r["n_unique_code"]} ({100-r["dup_rate_pct"]:.1f}% unique)')
            print(f'  lines p50={r["lines_p50"]:.0f}  ops p50={r["ops_p50"]:.0f}')
            print(f'  KEEP={kp} ({100*kp/n:.1f}%)  CUT={ct} ({100*ct/n:.1f}%)  '
                  f'DROP={dp} ({100*dp/n:.1f}%)')
            print(f'  reasons: {r["reason_breakdown"]}')
        rows.append(r)

    # Summary CSV
    fields = ['source', 'n_total', 'n_unique_code', 'n_dup', 'dup_rate_pct',
              'n_no_code', 'n_with_image', 'has_family', 'n_families',
              'lines_p10', 'lines_p50', 'lines_p90',
              'ops_p10', 'ops_p50', 'ops_p90',
              'rec_DROP', 'rec_CUT', 'rec_KEEP']
    with (out_dir / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n→ summary.csv at {out_dir/"summary.csv"}')


if __name__ == '__main__':
    main()
