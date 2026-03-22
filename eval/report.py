"""Generate markdown eval report from results collected by runner.py."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_report(run_dir: Path, cfg_yaml: str) -> str:
    results = _collect_results(run_dir)
    report = _render_report(results, run_dir, cfg_yaml)
    (run_dir / 'report.md').write_text(report)
    return report


def _collect_results(run_dir: Path) -> dict:
    results = {}

    for ckpt_dir in sorted(run_dir.iterdir()):
        if not ckpt_dir.is_dir() or ckpt_dir.name in ('renders',):
            continue

        ckpt_label = ckpt_dir.name

        for combo_dir in sorted(ckpt_dir.iterdir()):
            if not combo_dir.is_dir():
                continue

            meta_path = combo_dir / 'metadata.jsonl'
            if not meta_path.exists():
                continue

            parts = combo_dir.name.rsplit('_', 1)
            if len(parts) != 2:
                continue

            ds_name, modality = parts

            summary = _summarize(meta_path)
            key = (ckpt_label, ds_name, modality)
            results[key] = summary

            passk_path = combo_dir / 'passk.json'
            if passk_path.exists():
                with open(passk_path) as f:
                    results[key]['passk'] = json.load(f)

    return results


def _summarize(meta_path: Path) -> dict:
    import numpy as np

    records = []
    with open(meta_path) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        return {'n': 0}

    ious = [r['iou'] for r in records if r.get('iou') is not None and r['iou'] > 0]
    cds = [r['cd'] for r in records if r.get('cd') is not None]
    fail = sum(1 for r in records if r['error_type'] != 'success')

    err_types = {}
    for r in records:
        err_types[r['error_type']] = err_types.get(r['error_type'], 0) + 1

    iou_all = [r['iou'] if r.get('iou') is not None else 0.0 for r in records]

    buckets = {
        '>0.95':    sum(1 for v in iou_all if v > 0.95),
        '0.90-0.95': sum(1 for v in iou_all if 0.9 <= v <= 0.95),
        '0.70-0.90': sum(1 for v in iou_all if 0.7 <= v < 0.9),
        '0.50-0.70': sum(1 for v in iou_all if 0.5 <= v < 0.7),
        '<0.50':    sum(1 for v in iou_all if 0 < v < 0.5),
        'failure':  sum(1 for v in iou_all if v <= 0),
    }

    return {
        'n':            len(records),
        'n_success':    len(records) - fail,
        'failure_rate': fail / len(records) if records else 0.0,
        'mean_iou':     float(np.mean(ious)) if ious else 0.0,
        'median_iou':   float(np.median(ious)) if ious else 0.0,
        'mean_cd':      float(np.mean(cds)) if cds else None,
        'error_types':  err_types,
        'iou_buckets':  buckets,
    }


def _pct(n: int, total: int) -> str:
    if total == 0:
        return '—'
    return f'{n / total * 100:.1f}%'


def _render_report(results: dict, run_dir: Path, cfg_yaml: str) -> str:
    date = datetime.now().strftime('%Y-%m-%d')

    lines = [
        f'# Eval Report — {run_dir.name}',
        f'**Date:** {date}  |  **Dir:** `{run_dir}`',
        '',
        '---',
        '',
    ]

    ckpt_labels = sorted({k[0] for k in results})
    combos = sorted({(k[1], k[2]) for k in results})

    lines += [
        '## 1. IoU Summary',
        '',
        '| Checkpoint | ' + ' | '.join(f'{ds}/{mod}' for ds, mod in combos) + ' |',
        '|---' * (len(combos) + 1) + '|',
    ]

    for ckpt in ckpt_labels:
        cells = []
        for ds, mod in combos:
            res = results.get((ckpt, ds, mod))
            if res and res['n'] > 0:
                iou = res['mean_iou'] * 100
                fail = res['failure_rate'] * 100
                cells.append(f'{iou:.2f}% (fail {fail:.1f}%)')
            else:
                cells.append('—')
        lines.append(f'| {ckpt} | ' + ' | '.join(cells) + ' |')

    lines += ['']

    lines += [
        '## 2. Failure Breakdown',
        '',
        '| Checkpoint | Dataset/Mod | n | success | zero_iou | runtime_err | syntax_err | timeout |',
        '|---|---|---|---|---|---|---|---|',
    ]

    for ckpt in ckpt_labels:
        for ds, mod in combos:
            res = results.get((ckpt, ds, mod))
            if not res or res['n'] == 0:
                continue
            et = res.get('error_types', {})
            n = res['n']
            lines.append(
                f'| {ckpt} | {ds}/{mod} | {n} | '
                f'{et.get("success", 0)} ({_pct(et.get("success", 0), n)}) | '
                f'{et.get("zero_iou", 0)} | '
                f'{et.get("runtime_error", 0)} | '
                f'{et.get("syntax_error", 0)} | '
                f'{et.get("timeout", 0)} |'
            )

    lines += ['']

    lines += [
        '## 3. IoU Distribution',
        '',
        '| Checkpoint | Dataset/Mod | >0.95 | 0.90-0.95 | 0.70-0.90 | 0.50-0.70 | <0.50 | failure |',
        '|---|---|---|---|---|---|---|---|',
    ]

    for ckpt in ckpt_labels:
        for ds, mod in combos:
            res = results.get((ckpt, ds, mod))
            if not res or res['n'] == 0:
                continue
            bk = res.get('iou_buckets', {})
            n = res['n']
            lines.append(''.join([
                f'| {ckpt} | {ds}/{mod} | ',
                f'{bk.get(">0.95", 0)} ({_pct(bk.get(">0.95", 0), n)}) | ',
                f'{bk.get("0.90-0.95", 0)} ({_pct(bk.get("0.90-0.95", 0), n)}) | ',
                f'{bk.get("0.70-0.90", 0)} ({_pct(bk.get("0.70-0.90", 0), n)}) | ',
                f'{bk.get("0.50-0.70", 0)} ({_pct(bk.get("0.50-0.70", 0), n)}) | ',
                f'{bk.get("<0.50", 0)} ({_pct(bk.get("<0.50", 0), n)}) | ',
                f'{bk.get("failure", 0)} ({_pct(bk.get("failure", 0), n)}) |',
            ]))

    lines += ['']

    passk_results = {k: v for k, v in results.items() if v.get('passk')}
    if passk_results:
        lines += ['## 4. Pass@k', '']

        k_vals = set()
        for res in passk_results.values():
            k_vals.update(int(k) for k in res['passk'].get('pass_at_k', {}).keys())
        k_vals = sorted(k_vals)

        lines += [
            '| Checkpoint | Dataset/Mod | threshold | n_samples | ' + ' | '.join(f'pass@{k}' for k in k_vals) + ' | mean_IoU |',
            '|---' * (len(k_vals) + 4) + '|',
        ]

        for ckpt in ckpt_labels:
            for ds, mod in combos:
                res = results.get((ckpt, ds, mod))
                if not res or not res.get('passk'):
                    continue
                pk = res['passk']
                thr = pk.get('threshold', '?')
                ns = pk.get('n_samples', '?')
                k_cells = []
                for k in k_vals:
                    v = pk.get('pass_at_k', {}).get(str(k))
                    k_cells.append(f'{v:.3f}' if v is not None else '—')
                mean_iou = pk.get('mean_iou')
                if mean_iou:
                    lines.append(
                        f'| {ckpt} | {ds}/{mod} | {thr} | {ns} | '
                        + ' | '.join(k_cells)
                        + f' | {mean_iou:.3f} |'
                    )
                else:
                    lines.append(' | — |')

        lines += ['']

    lines += [
        '## 5. Config',
        '',
        '```yaml',
        cfg_yaml.strip(),
        '```',
        '',
    ]

    return '\n'.join(lines)
