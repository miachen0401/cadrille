"""Retrospective family-choice robustness check (T1-3).

Uses v3 ckpt-50000 predictions (which saw all 106 families) to test:
"Is essential_pass uniformly high across families, or does v3 itself
fail on certain family subsets — making our v4-holdout signal meaningful?"

Method:
  1. For each step's predictions, compute per-family essential_pass on v3
  2. Pick 5 random subsets of 10 families and aggregate ess_pass per subset
  3. Show distribution: if v3 scores 0.7-1.0 across many subsets, v4-holdout's
     0.0-0.4 OOD result is meaningfully below baseline.

Also: pick 3 alternative 10-family configs and compute v3's ess_pass on each.

Usage:
    uv run python -m scripts.analysis.retrospective_family_choice
"""
from __future__ import annotations

import json
import pickle
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'

CURRENT_HOLDOUT = ['tapered_boss', 'taper_pin', 'venturi_tube', 'bucket', 'dome_cap',
                   'nozzle', 'enclosure', 'waffle_plate', 'bolt', 'duct_elbow']

# Alternative configs (proposed in §8.2 Tier 1)
CONFIG_FEATURE_HEAVY = ['filleted_l_plate', 'filleted_t_plate', 'cbore_block',
                        'cskd_plate', 'thru_hole_plate', 'blind_hole_plate',
                        'bored_plate', 'flanged_collar', 'lug_plate', 'multi_step_plate']

CONFIG_PROFILE_DRIVEN = ['l_bracket', 'cross_plate', 'star_plate', 'heart_plate',
                         'horseshoe_plate', 'spline_hub', 'dovetail_block',
                         'hex_standoff', 'ratchet_wheel', 'gear_disc']


def load_setup():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    patterns = {n: re.compile(p) for n, p in tax['patterns'].items()}
    ess_spec = yaml.safe_load(open(REPO_ROOT / 'configs/eval/canonical_ops.yaml'))
    return uid2fam, patterns, ess_spec


def find_ops(code, patterns):
    if not code: return set()
    out = {n for n, p in patterns.items() if p.search(code)}
    if 'sweep' in out and 'helix' in out:
        out.add('sweep+helix')
    return out


def ess_pass(family, ops, spec):
    s = spec.get(family)
    if not s: return None
    for elem in s:
        if isinstance(elem, str):
            if elem not in ops: return False
        else:
            if not any(o in ops for o in elem): return False
    return True


def collect_v3_per_family(uid2fam, patterns, ess_spec):
    """Aggregate per-family essential_pass across all v3 eval steps."""
    fam_ess = {}  # family -> [pass_rates_per_step]
    for f in sorted(Path(V3_DIR).glob('step-*.jsonl')):
        if '.max@' in f.name: continue
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0: continue
        # Use only late steps (>=20k) where v3 is well-trained
        if step < 20000: continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        bc = [r for r in rows if r.get('bucket') == 'BenchCAD val']
        for r in bc:
            fam = uid2fam.get(r['uid'])
            if not fam: continue
            po = find_ops(r.get('pred_code') or '', patterns)
            e = ess_pass(fam, po, ess_spec)
            if e is None: continue
            fam_ess.setdefault(fam, []).append(1 if e else 0)
    # Reduce to mean per family
    return {f: (np.mean(vs), len(vs)) for f, vs in fam_ess.items()}


def main():
    uid2fam, patterns, ess_spec = load_setup()
    fam_data = collect_v3_per_family(uid2fam, patterns, ess_spec)

    print(f'v3 per-family essential_pass (mean across step >=20k, num samples per family):')
    print(f'  {len(fam_data)} families with samples')
    print(f'  {"family":<25} {"mean":<8} {"n":<4}')
    for fam, (mean, n) in sorted(fam_data.items(), key=lambda x: -x[1][0])[:30]:
        print(f'  {fam:<25} {mean:.3f}    {n}')
    print('  ...')
    print(f'  {"family":<25} {"mean":<8} {"n":<4}')
    for fam, (mean, n) in sorted(fam_data.items(), key=lambda x: -x[1][0])[-10:]:
        print(f'  {fam:<25} {mean:.3f}    {n}')

    print()
    print('=' * 80)
    print('CONFIG TEST: per 10-family-config v3 essential_pass (mean across families):')
    print()

    def config_score(config_name, fams):
        means = [fam_data[f][0] for f in fams if f in fam_data]
        ns = [fam_data[f][1] for f in fams if f in fam_data]
        if not means:
            print(f'  {config_name}: NO DATA')
            return
        avg = np.mean(means)
        print(f'  {config_name}:  mean ess_pass = {avg:.3f} (n_fams={len(means)}, total samples = {sum(ns)})')
        for f in fams:
            d = fam_data.get(f)
            print(f'    {f:<22} {d[0]:.3f} (n={d[1]})' if d else f'    {f:<22}  -- not in val sample --')

    config_score('Current holdout (structural)', CURRENT_HOLDOUT)
    print()
    config_score('Config B (feature-heavy)', CONFIG_FEATURE_HEAVY)
    print()
    config_score('Config C (profile-driven)', CONFIG_PROFILE_DRIVEN)
    print()

    # Random subsets
    print('=' * 80)
    print('5 random 10-family subsets (drawn from families with n>=2 samples):')
    eligible = [f for f, (_, n) in fam_data.items() if n >= 2]
    print(f'  eligible pool: {len(eligible)} families')
    rng = random.Random(42)
    for i in range(5):
        sub = rng.sample(eligible, 10)
        means = [fam_data[f][0] for f in sub]
        print(f'  subset {i+1}: mean ess_pass = {np.mean(means):.3f}, range [{min(means):.2f}, {max(means):.2f}]')

    # Conclusion check
    print()
    print('=' * 80)
    print('Interpretation guide:')
    print('  If most random subsets score ess_pass > 0.7, v3 is uniformly good →')
    print('  v4-holdout dropping to 0.04-0.50 on the same families is a real gap (not noise).')
    print('  If random subsets vary widely (0.3-0.9), v4-holdout signal is family-pick-dependent.')


if __name__ == '__main__':
    main()
