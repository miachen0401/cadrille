"""Per-step v4-vs-v3 comparison report — ops + IID/OOD + max@8.

Reads a single step's JSONL from both v4-holdout and v3 prediction dirs,
computes the full metric grid (IoU + recall + rare_recall + essential_pass
+ feature_F1 + op_entropy + max@8) split by IID/OOD, and emits a Discord-
ready markdown block.

Usage:
    uv run python -m scripts.analysis.eval_report --step 12000
    uv run python -m scripts.analysis.eval_report --step 12000 --post
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

V4_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions'
V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'
V4_LOG = '/home/ubuntu/cadrille/logs/v4_holdout_20260430_082814.log'
V3_LOG = '/home/ubuntu/cadrille/logs/v3_clean_20260428_132042.log'

HOLDOUT = {'tapered_boss', 'taper_pin', 'venturi_tube', 'bucket', 'dome_cap',
           'nozzle', 'enclosure', 'waffle_plate', 'bolt', 'duct_elbow'}


def _load_taxonomy():
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    patterns = {n: re.compile(p) for n, p in tax['patterns'].items()}
    return {
        'patterns': patterns,
        'rare': set(tax['rare']),
        'feature': set(tax['feature']),
    }


def _load_essentials():
    return yaml.safe_load(open(REPO_ROOT / 'configs/eval/canonical_ops.yaml'))


def find_ops(code: str, patterns: dict) -> set[str]:
    if not code:
        return set()
    out = {n for n, p in patterns.items() if p.search(code)}
    if 'sweep' in out and 'helix' in out:
        out.add('sweep+helix')
    return out


def essential_pass(family: str, ops: set[str], spec_dict: dict):
    spec = spec_dict.get(family)
    if not spec:
        return None
    for elem in spec:
        if isinstance(elem, str):
            if elem not in ops:
                return False
        else:
            if not any(o in ops for o in elem):
                return False
    return True


def feature_f1(p: set[str], g: set[str], features: set[str]) -> float:
    pf = p & features
    gf = g & features
    if not gf and not pf:
        return 1.0
    if not gf or not pf:
        return 0.0
    tp = len(pf & gf); fp = len(pf - gf); fn = len(gf - pf)
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0


def op_entropy(rows: list[dict], patterns: dict) -> float:
    if not rows:
        return 0
    op_names = list(patterns)
    counts = np.zeros(len(op_names))
    for r in rows:
        ops = find_ops(r.get('pred_code') or '', patterns)
        for i, n in enumerate(op_names):
            if n in ops:
                counts[i] += 1
    if counts.sum() == 0:
        return 0
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def metrics_for(rows: list[dict], uid2fam: dict, tax: dict, ess_spec: dict) -> dict:
    if not rows:
        return None
    n = len(rows)
    ious, execs = [], 0
    recall_per, rare_recall_per = [], []
    ess_pass, feat_f1s = [], []
    pred_op_set = set()
    for r in rows:
        iou = r.get('iou', 0) or 0
        if iou < 0:
            iou = 0
        ious.append(iou)
        if r.get('has_iou') and (r.get('iou') or -1) >= 0:
            execs += 1
        po = find_ops(r.get('pred_code') or '', tax['patterns'])
        go = find_ops(r.get('gt_code') or '', tax['patterns'])
        pred_op_set.update(po)
        if go:
            recall_per.append(len(go & po) / len(go))
            gr = go & tax['rare']
            if gr:
                rare_recall_per.append(len(gr & po) / len(gr))
        fam = uid2fam.get(r['uid'])
        e = essential_pass(fam, po, ess_spec) if fam else None
        if e is not None:
            ess_pass.append(1 if e else 0)
        feat_f1s.append(feature_f1(po, go, tax['feature']))
    return {
        'iou': float(np.mean(ious)),
        'exec_rate': execs / n * 100,
        'recall': float(np.mean(recall_per)) if recall_per else 0,
        'rare_recall': float(np.mean(rare_recall_per)) if rare_recall_per else 0,
        'distinct_ops': len(pred_op_set),
        'ess_pass': float(np.mean(ess_pass)) if ess_pass else None,
        'ess_n': len(ess_pass),
        'feat_f1': float(np.mean(feat_f1s)),
        'op_entropy': op_entropy(rows, tax['patterns']),
        'n': n,
    }


_MAX8_RE = re.compile(
    r'\[(?P<bucket>BenchCAD val|DeepCAD test|Fusion360 test)\]\s+'
    r'max_iou@8 \(t=[\d.]+\)=(?P<iou>[\d.]+)\s+pass>0\.5=(?P<pass>[\d.]+)%'
)


def parse_max8_for_step(log_path: str, step: int) -> dict:
    """Return {bucket: {iou, pass}} for given step's max@8 block, or {}."""
    text = Path(log_path).read_text(errors='ignore')
    blocks = re.split(r'\[online-eval\] step=(\d+) running IoU eval', text)
    # blocks alternates: prelude, step_str, body, step_str, body, ...
    out = {}
    for i in range(1, len(blocks), 2):
        s = int(blocks[i])
        if s != step:
            continue
        body = blocks[i + 1] if i + 1 < len(blocks) else ''
        for m in _MAX8_RE.finditer(body):
            out[m['bucket']] = {'iou': float(m['iou']), 'pass': float(m['pass'])}
    return out


def build_report(step: int) -> str:
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = _load_taxonomy()
    ess_spec = _load_essentials()

    def parse(pred_dir: str):
        path = Path(pred_dir) / f'step-{step:06d}.jsonl'
        if not path.exists():
            return None
        rows = [json.loads(l) for l in path.open() if l.strip()]
        bc = [r for r in rows if r.get('bucket') == 'BenchCAD val']
        dc = [r for r in rows if r.get('bucket') == 'DeepCAD test']
        fu = [r for r in rows if r.get('bucket') == 'Fusion360 test']
        iid = [r for r in bc if uid2fam.get(r['uid']) not in HOLDOUT]
        ood = [r for r in bc if uid2fam.get(r['uid']) in HOLDOUT]
        return {
            'IID': metrics_for(iid, uid2fam, tax, ess_spec),
            'OOD': metrics_for(ood, uid2fam, tax, ess_spec),
            'DC': metrics_for(dc, uid2fam, tax, ess_spec),
            'FU': metrics_for(fu, uid2fam, tax, ess_spec),
        }

    v4 = parse(V4_DIR)
    v3 = parse(V3_DIR)
    if v4 is None or v3 is None:
        return f'**ERROR**: predictions not available for step {step} on both runs'

    v4m8 = parse_max8_for_step(V4_LOG, step)
    v3m8 = parse_max8_for_step(V3_LOG, step)

    def fmt_d(a, b):
        if a is None or b is None:
            return '   -    '
        return f'{a:.3f}->{b:.3f} ({b-a:+.3f})'

    def fmt_d_pct(a, b):
        if a is None or b is None:
            return '   -    '
        return f'{a:.0f}%->{b:.0f}% ({b-a:+.0f}pp)'

    def fmt_int(a, b):
        if a is None or b is None:
            return '  -  '
        return f'{a}->{b}'

    lines = [f'**v4-holdout step={step} — full metric report (vs v3)**', '']

    # 1. greedy IoU + exec
    lines.append('**Greedy IoU + exec_rate (v3 -> v4):**')
    lines.append('```')
    lines.append(f'{"bucket":<10} {"n":<3} {"IoU":<24} {"exec":<22}')
    for bk, label in [('IID','BC IID'), ('OOD','BC OOD'), ('DC','DC test'), ('FU','Fu test')]:
        v3d = v3[bk]; v4d = v4[bk]
        if not v3d or not v4d: continue
        lines.append(f'{label:<10} {v4d["n"]:<3} {fmt_d(v3d["iou"], v4d["iou"]):<24} {fmt_d_pct(v3d["exec_rate"], v4d["exec_rate"]):<22}')
    lines.append('```')

    # 2. ops metrics
    lines.append('**Op metrics (v3 -> v4):**')
    lines.append('```')
    lines.append(f'{"bucket":<10} {"recall":<24} {"rare_recall":<24} {"distinct":<12}')
    for bk, label in [('IID','BC IID'), ('OOD','BC OOD'), ('DC','DC test'), ('FU','Fu test')]:
        v3d = v3[bk]; v4d = v4[bk]
        if not v3d or not v4d: continue
        lines.append(f'{label:<10} {fmt_d(v3d["recall"], v4d["recall"]):<24} {fmt_d(v3d["rare_recall"], v4d["rare_recall"]):<24} {fmt_int(v3d["distinct_ops"], v4d["distinct_ops"]):<12}')
    lines.append('```')

    # 3. essential_pass + feat_f1 + op_entropy (BC only for ess; all for others)
    lines.append('**essential_pass + feature_F1 + op_entropy (v3 -> v4):**')
    lines.append('```')
    lines.append(f'{"bucket":<10} {"ess_pass":<22} {"ess_n":<10} {"feat_F1":<22} {"op_ent":<14}')
    for bk, label in [('IID','BC IID'), ('OOD','BC OOD'), ('DC','DC test'), ('FU','Fu test')]:
        v3d = v3[bk]; v4d = v4[bk]
        if not v3d or not v4d: continue
        ess_str = fmt_d(v3d["ess_pass"], v4d["ess_pass"]) if v3d["ess_pass"] is not None else '   N/A          '
        ess_n = f'{v3d["ess_n"]}/{v4d["ess_n"]}'
        lines.append(f'{label:<10} {ess_str:<22} {ess_n:<10} {fmt_d(v3d["feat_f1"], v4d["feat_f1"]):<22} {fmt_d(v3d["op_entropy"], v4d["op_entropy"]):<14}')
    lines.append('```')

    # 4. max@8 if available
    if v4m8 and v3m8:
        lines.append('**max@8 (k=8, T=1.0) (v3 -> v4):**')
        lines.append('```')
        lines.append(f'{"bucket":<10} {"max IoU":<22} {"pass>0.5":<22}')
        for bk in ('BenchCAD val', 'DeepCAD test', 'Fusion360 test'):
            v3m = v3m8.get(bk); v4m = v4m8.get(bk)
            if not v3m or not v4m: continue
            lines.append(f'{bk:<10} {fmt_d(v3m["iou"], v4m["iou"]):<22} {fmt_d_pct(v3m["pass"], v4m["pass"]):<22}')
        lines.append('```')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--post', action='store_true', help='post to Discord')
    args = ap.parse_args()

    text = build_report(args.step)
    print(text)

    if args.post:
        import requests
        webhook = os.environ.get('DISCORD_WEBHOOK_URL')
        if not webhook:
            try:
                import subprocess
                webhook = subprocess.check_output(
                    "grep '^export DISCORD_WEBHOOK_URL' ~/.bashrc | "
                    "sed 's/.*=\"\\?//; s/\"$//'", shell=True, text=True).strip()
            except Exception:
                webhook = None
        if not webhook:
            print('DISCORD_WEBHOOK_URL not set', file=sys.stderr)
            return
        r = requests.post(webhook, json={'content': text}, timeout=30)
        print(f'discord HTTP {r.status_code}', file=sys.stderr)


if __name__ == '__main__':
    main()
