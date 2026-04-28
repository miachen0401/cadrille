"""Per-source family grid: stats + 5 sample renders per family.

For each source with a `family` (or extractable from uid prefix), build:
  rows = families sorted by code complexity (median lines × ops desc)
  each row = [5 sample renders | family name + n_items + lines/ops stats | recommendation badge]

Sources:
  benchcad, cad_iso_106, benchcad_simple   — all 3 have family info

Output: experiments_log/family_grid_{source}.png  (and a summary table CSV)

Usage:
    uv run python -m scripts.analysis.family_grid_audit
"""
from __future__ import annotations

import argparse
import io
import pickle
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

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


def code_stats(code: str) -> tuple[int, int]:
    n_lines = sum(1 for l in code.split('\n') if l.strip())
    ops = set(CADQUERY_OP_RE.findall(code))
    return n_lines, len(ops)


def family_for(item: dict, has_family_field: bool) -> str:
    if has_family_field and item.get('family'):
        return item['family']
    return fam_from_uid(item['uid'])


def load_code(root: Path, item: dict) -> str | None:
    if 'py_path' in item:
        p = root / item['py_path']
        if p.exists():
            try: return p.read_text()
            except Exception: return None
    return None


def audit_source(src_name: str, root: Path, has_family_field: bool,
                 samples_per_fam: int = 50) -> dict:
    """Return: {family: {'n_items': int, 'lines_p50', 'lines_p90',
                         'ops_p50', 'ops_p90', 'sample_uids': [5]}}."""
    rows = pickle.load(open(root / 'train.pkl', 'rb'))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[family_for(r, has_family_field)].append(r)

    rng = random.Random(42)
    out = {}
    for fam, items in by_family.items():
        # Stats: sample N codes for stats
        sample_for_stats = rng.sample(items, min(samples_per_fam, len(items)))
        line_counts, op_counts = [], []
        for it in sample_for_stats:
            code = load_code(root, it)
            if code:
                l, o = code_stats(code)
                line_counts.append(l); op_counts.append(o)
        # 5 sample uids for visual
        sample_for_vis = rng.sample(items, min(5, len(items)))
        out[fam] = {
            'n_items': len(items),
            'lines_p50': statistics.median(line_counts) if line_counts else 0,
            'lines_p90': float(np.percentile(line_counts, 90)) if line_counts else 0,
            'ops_p50':   statistics.median(op_counts) if op_counts else 0,
            'ops_p90':   float(np.percentile(op_counts, 90)) if op_counts else 0,
            'sample_items': sample_for_vis,
        }
    return out


def png_thumb(p: Path, size: int) -> Image.Image | None:
    if not p.exists(): return None
    try:
        return Image.open(p).convert('RGB').resize((size, size), Image.LANCZOS)
    except Exception:
        return None


def build_grid(src_name: str, root: Path, fams_data: dict,
               recommendation_fn) -> bytes:
    """Layout per row:
        [5 thumbs (80px each) | label-text panel | recommendation badge]
    Sort: recommended-drop on top, then complexity ascending → easy to scan.
    """
    cell = 90
    label_w = 380
    n_thumbs = 5

    # Sort: drop first, then by ops_p50 ascending (so simplest within each group is at top)
    def sort_key(fam):
        d = fams_data[fam]
        rec = recommendation_fn(fam, d)
        rec_rank = {'DROP': 0, 'CUT': 1, 'KEEP': 2}.get(rec, 3)
        return (rec_rank, d['ops_p50'], d['lines_p50'])

    fams_sorted = sorted(fams_data.keys(), key=sort_key)
    n_rows = len(fams_sorted)

    header_h = 30
    W = n_thumbs * cell + label_w
    H = n_rows * cell + header_h

    img = Image.new('RGB', (W, H), (245, 245, 245))
    drw = ImageDraw.Draw(img)

    # Title
    total_items = sum(d['n_items'] for d in fams_data.values())
    drw.text((10, 6),
             f'{src_name}  —  {n_rows} families, {total_items} total items   '
             f'(sorted: DROP top → KEEP bottom, then complexity asc)',
             fill=(20, 20, 20))

    for r, fam in enumerate(fams_sorted):
        d = fams_data[fam]
        y = header_h + r * cell
        rec = recommendation_fn(fam, d)
        # Recommendation badge color
        bg = {'DROP': (255, 220, 220), 'CUT': (255, 240, 200),
              'KEEP': (220, 240, 220)}.get(rec, (240, 240, 240))
        drw.rectangle([0, y, W, y + cell - 1], fill=bg, outline=(180, 180, 180))

        # 5 thumbs
        for c, item in enumerate(d['sample_items'][:n_thumbs]):
            x = c * cell
            png_path = root / item.get('png_path', '')
            tmb = png_thumb(png_path, cell - 4)
            if tmb is not None:
                img.paste(tmb, (x + 2, y + 2))
            else:
                drw.rectangle([x + 2, y + 2, x + cell - 2, y + cell - 2],
                              fill=(220, 220, 220))

        # Label panel
        lx = n_thumbs * cell + 8
        drw.text((lx, y + 4), f'[{rec}]  {fam}', fill=(20, 20, 20))
        drw.text((lx, y + 22),
                 f'n={d["n_items"]:>5}   lines p50/p90={d["lines_p50"]:.0f}/{d["lines_p90"]:.0f}',
                 fill=(50, 50, 50))
        drw.text((lx, y + 38),
                 f'ops p50/p90 = {d["ops_p50"]:.0f}/{d["ops_p90"]:.0f}',
                 fill=(50, 50, 50))
        # Recommendation reason
        reason = recommendation_reason(fam, d)
        drw.text((lx, y + 56), reason, fill=(80, 80, 80))

    buf = io.BytesIO(); img.save(buf, format='PNG'); return buf.getvalue()


def recommendation_for(fam: str, d: dict) -> str:
    """DROP / CUT / KEEP based on per-family stats."""
    ops_p50 = d['ops_p50']; lines_p50 = d['lines_p50']
    # Pure-primitive families with ops_p50 ≤ 2 AND lines_p50 ≤ 8 → DROP
    if ops_p50 <= 2 and lines_p50 <= 8:
        return 'DROP'
    # Marginal: ops_p50 ≤ 3 → CUT (reduce weight)
    if ops_p50 <= 3:
        return 'CUT'
    return 'KEEP'


def recommendation_reason(fam: str, d: dict) -> str:
    rec = recommendation_for(fam, d)
    if rec == 'DROP':
        return 'reason: trivial code (≤2 unique ops, ≤8 lines)'
    if rec == 'CUT':
        return 'reason: low op variety (≤3 unique ops)'
    return 'reason: ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-per-fam', type=int, default=50)
    args = ap.parse_args()

    sources = [
        ('benchcad',         REPO_ROOT / 'data/benchcad',         True),
        ('cad_iso_106',      REPO_ROOT / 'data/cad-iso-106',      False),
        ('benchcad_simple',  REPO_ROOT / 'data/benchcad-simple',  False),
    ]
    out_dir = REPO_ROOT / 'experiments_log'
    out_dir.mkdir(exist_ok=True)

    summary_lines = ['source,family,n_items,lines_p50,lines_p90,ops_p50,ops_p90,recommendation']

    for src_name, root, has_fam in sources:
        print(f'\n=== {src_name} ===', flush=True)
        fams = audit_source(src_name, root, has_fam, args.samples_per_fam)
        png = build_grid(src_name, root, fams, recommendation_for)
        out = out_dir / f'family_grid_{src_name}.png'
        out.write_bytes(png)
        print(f'wrote {out}  ({len(png)//1024} KB, {len(fams)} families)',
              flush=True)
        # Tally recommendations
        n_drop = sum(1 for f, d in fams.items()
                     if recommendation_for(f, d) == 'DROP')
        n_cut = sum(1 for f, d in fams.items()
                    if recommendation_for(f, d) == 'CUT')
        items_drop = sum(d['n_items'] for f, d in fams.items()
                         if recommendation_for(f, d) == 'DROP')
        items_cut = sum(d['n_items'] for f, d in fams.items()
                        if recommendation_for(f, d) == 'CUT')
        items_total = sum(d['n_items'] for d in fams.values())
        print(f'  DROP: {n_drop} families ({items_drop}/{items_total} items, '
              f'{100*items_drop/items_total:.1f}%)')
        print(f'  CUT:  {n_cut} families ({items_cut}/{items_total} items, '
              f'{100*items_cut/items_total:.1f}%)')
        for fam, d in fams.items():
            rec = recommendation_for(fam, d)
            summary_lines.append(
                f'{src_name},{fam},{d["n_items"]},{d["lines_p50"]:.0f},'
                f'{d["lines_p90"]:.0f},{d["ops_p50"]:.0f},'
                f'{d["ops_p90"]:.0f},{rec}')

    csv_path = out_dir / 'family_grid_summary.csv'
    csv_path.write_text('\n'.join(summary_lines) + '\n')
    print(f'\nwrote summary CSV → {csv_path}')

    # ── Master summary image ───────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv as _csv
    rows = []
    with csv_path.open() as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    sources_order = ['benchcad', 'cad_iso_106', 'benchcad_simple']
    rec_colors = {'DROP': '#e63946', 'CUT': '#f4a261', 'KEEP': '#2a9d8f'}

    # Top row: per-source recommendation breakdown (stacked bars)
    ax = fig.add_subplot(gs[0, :])
    src_data = {s: {'DROP': 0, 'CUT': 0, 'KEEP': 0,
                    'DROP_items': 0, 'CUT_items': 0, 'KEEP_items': 0}
                for s in sources_order}
    for r in rows:
        s = r['source']; rec = r['recommendation']; n = int(r['n_items'])
        src_data[s][rec] += 1
        src_data[s][f'{rec}_items'] += n

    x = np.arange(len(sources_order))
    width = 0.35
    keep_items = [src_data[s]['KEEP_items'] for s in sources_order]
    cut_items  = [src_data[s]['CUT_items'] for s in sources_order]
    drop_items = [src_data[s]['DROP_items'] for s in sources_order]
    ax.bar(x, keep_items, width, label='KEEP',
           color=rec_colors['KEEP'])
    ax.bar(x, cut_items, width, bottom=keep_items, label='CUT',
           color=rec_colors['CUT'])
    ax.bar(x, drop_items, width,
           bottom=[k+c for k, c in zip(keep_items, cut_items)],
           label='DROP', color=rec_colors['DROP'])
    ax.set_xticks(x); ax.set_xticklabels(sources_order, fontsize=11)
    ax.set_ylabel('item count', fontsize=11)
    ax.set_title('Per-source filter recommendation (item-level breakdown)',
                 fontsize=13)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25, axis='y')
    for i, s in enumerate(sources_order):
        total = keep_items[i] + cut_items[i] + drop_items[i]
        kept_pct = 100 * keep_items[i] / total
        ax.text(i, total + total*0.02,
                f'{total} total\n{int(kept_pct)}% keep, '
                f'{int(100*cut_items[i]/total)}% cut, '
                f'{int(100*drop_items[i]/total)}% drop',
                ha='center', fontsize=9)

    # Middle row: median ops + median lines histograms per source
    for i, src in enumerate(sources_order):
        ax2 = fig.add_subplot(gs[1, i])
        src_rows = [r for r in rows if r['source'] == src]
        ops_data = [int(r['ops_p50']) for r in src_rows]
        lines_data = [int(r['lines_p50']) for r in src_rows]
        ax2.hist(ops_data, bins=range(0, 14), color='#3a7ca5',
                 alpha=0.65, label='ops p50')
        ax2.hist(lines_data, bins=range(0, 50, 3), color='#e85c1e',
                 alpha=0.45, label='lines p50')
        ax2.set_title(f'{src}', fontsize=11)
        ax2.set_xlabel('value', fontsize=10)
        ax2.set_ylabel('# families', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.25)

    # Bottom row: family-count and per-family-item-count distribution per source
    ax3 = fig.add_subplot(gs[2, :])
    table_text = []
    for s in sources_order:
        d = src_data[s]
        n_total = d['DROP'] + d['CUT'] + d['KEEP']
        items_total = d['DROP_items'] + d['CUT_items'] + d['KEEP_items']
        table_text.append([
            s,
            f'{n_total}',
            f"{d['DROP']}",
            f"{d['CUT']}",
            f"{d['KEEP']}",
            f'{items_total:,}',
            f"{d['DROP_items']:,} ({100*d['DROP_items']/max(items_total,1):.1f}%)",
            f"{d['CUT_items']:,} ({100*d['CUT_items']/max(items_total,1):.1f}%)",
            f"{d['KEEP_items']:,} ({100*d['KEEP_items']/max(items_total,1):.1f}%)",
        ])
    ax3.axis('off')
    tbl = ax3.table(cellText=table_text,
                     colLabels=['source', 'n_fam', 'DROP fam', 'CUT fam',
                                'KEEP fam', 'total items',
                                'DROP items', 'CUT items', 'KEEP items'],
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    # color recommendation columns
    for r_idx in range(1, len(table_text) + 1):
        tbl[(r_idx, 2)].set_facecolor(rec_colors['DROP'])
        tbl[(r_idx, 3)].set_facecolor(rec_colors['CUT'])
        tbl[(r_idx, 4)].set_facecolor(rec_colors['KEEP'])
        tbl[(r_idx, 6)].set_facecolor(rec_colors['DROP'])
        tbl[(r_idx, 7)].set_facecolor(rec_colors['CUT'])
        tbl[(r_idx, 8)].set_facecolor(rec_colors['KEEP'])
    ax3.set_title('Filter summary table', fontsize=12, pad=12)

    fig.suptitle('Dataset filter recommendation — overview', fontsize=15,
                 y=0.995)

    out_summary = out_dir / 'family_grid_summary.png'
    fig.savefig(out_summary, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote summary plot → {out_summary}')


if __name__ == '__main__':
    main()
