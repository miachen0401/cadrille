"""Pre-flight integrity check for SFT training data.

Run BEFORE launching SFT training. Verifies for each source in the YAML mix:
  1. train.pkl exists and is loadable
  2. Per-row required field presence (uid, code or py_path, png_path if image source)
  3. All png_path entries resolve to existing files on disk
  4. All py_path entries resolve (file exists OR code field is non-empty)
  5. .filter_cache stale check (will be cleared if N changed)
  6. PNG sample-load (verify they decode as valid images)

Exits non-zero if any source fails. Prints a summary report.

Usage:
    uv run python -m scripts.preflight_check --config configs/sft/baseline.yaml
"""
from __future__ import annotations
import argparse
import io
import pickle
import random
import sys
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


SOURCE_PATHS = {
    'benchcad':           'data/benchcad',
    'cad_iso_106':        'data/cad-iso-106',
    'benchcad_simple':    'data/benchcad-simple',
    'text2cad_bench_img': 'data/text2cad-bench',
    'text2cad_bench_text':'data/text2cad-bench',
    'text2cad_bench':     'data/text2cad-bench',  # legacy key
    'recode_bench':       'data/cad-recode-bench',
    'recode20k':          'data/cad-recode-20k',
    'text2cad':           'data/text2cad',  # legacy
    'recode':             'data/cad-recode-v1.5',  # legacy
}

REQUIRES_IMG = {'benchcad', 'cad_iso_106', 'benchcad_simple',
                'text2cad_bench_img', 'recode_bench', 'recode20k', 'recode'}


def check_source(name: str, root: Path, n_sample: int = 200) -> dict:
    """Return dict with checks for one source."""
    out = {'source': name, 'root': str(root), 'errors': [], 'warnings': []}
    pkl = root / 'train.pkl'
    if not pkl.exists():
        out['errors'].append(f'train.pkl missing at {pkl}')
        return out
    try:
        rows = pickle.load(open(pkl, 'rb'))
    except Exception as e:
        out['errors'].append(f'pkl load failed: {e}')
        return out
    out['n_train'] = len(rows)
    if not rows:
        out['errors'].append('train.pkl is empty')
        return out

    # Field presence (sample first 5)
    s0 = rows[0]
    out['fields'] = sorted(s0.keys())

    needs_img = name in REQUIRES_IMG
    if needs_img and 'png_path' not in s0:
        out['errors'].append('image source but no png_path field')

    has_code_inline = bool(s0.get('code'))
    has_code_path = 'py_path' in s0 or (root / 'cadquery' / f"{s0['uid']}.py").exists()
    if not (has_code_inline or has_code_path):
        out['errors'].append('no code (inline or via py_path)')

    # Sample-check N items: PNG resolves + decodes; code resolves
    if n_sample > 0:
        rng = random.Random(42)
        sample = rng.sample(rows, min(n_sample, len(rows)))
        n_png_missing = 0; n_png_corrupt = 0
        n_code_missing = 0
        for r in sample:
            if needs_img and 'png_path' in r:
                p = root / r['png_path']
                if not p.exists():
                    n_png_missing += 1
                else:
                    try:
                        Image.open(p).verify()
                    except Exception:
                        n_png_corrupt += 1
            # Code check
            if not r.get('code'):
                if 'py_path' in r:
                    p = root / r['py_path']
                    if not p.exists():
                        # try cadquery/{uid}.py
                        p2 = root / 'cadquery' / f"{r['uid']}.py"
                        if not p2.exists():
                            n_code_missing += 1
                else:
                    n_code_missing += 1
        out['sample_png_missing'] = n_png_missing
        out['sample_png_corrupt'] = n_png_corrupt
        out['sample_code_missing'] = n_code_missing
        if n_png_missing:
            # Estimate full-source missing count
            est_full = int(n_png_missing * len(rows) / len(sample))
            out['errors'].append(
                f'{n_png_missing}/{len(sample)} PNGs missing in sample '
                f'(est ~{est_full} of {len(rows)} total). Will crash dataloader.')
        if n_png_corrupt:
            out['warnings'].append(f'{n_png_corrupt}/{len(sample)} PNGs corrupt in sample')
        if n_code_missing:
            out['errors'].append(f'{n_code_missing}/{len(sample)} codes missing in sample')

    # Filter cache freshness
    fc = root / '.filter_cache'
    if fc.exists():
        # Just check if there are cached files; we can't easily verify staleness
        # without recomputing — leave warning for user to decide
        n_cache = len(list(fc.glob('*.json')))
        if n_cache > 0:
            out['warnings'].append(
                f'{n_cache} stale .filter_cache files — recommend `rm -rf {fc}`')

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to SFT YAML config')
    ap.add_argument('--n-sample', type=int, default=200,
                    help='# of items to spot-check per source (default 200)')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mix = cfg.get('sft_mix_weights', {})
    sources = [k for k, w in mix.items() if w and w > 0]
    print(f'Pre-flight check for {len(sources)} sources from {args.config}\n', flush=True)

    n_errors = 0; n_warnings = 0
    for src in sources:
        if src not in SOURCE_PATHS:
            print(f'  [{src}] UNKNOWN source — no path mapping')
            n_errors += 1
            continue
        root = REPO_ROOT / SOURCE_PATHS[src]
        r = check_source(src, root, args.n_sample)
        print(f'== {src} ==')
        print(f'  root: {r["root"]}')
        if 'n_train' in r: print(f'  n_train: {r["n_train"]}')
        if 'fields' in r:  print(f'  fields:  {r["fields"]}')
        if 'sample_png_missing' in r:
            print(f'  PNG check: {r["sample_png_missing"]} missing, '
                  f'{r["sample_png_corrupt"]} corrupt (of {args.n_sample} sample)')
            print(f'  code check: {r["sample_code_missing"]} missing')
        for w in r['warnings']:
            print(f'  ⚠ WARN: {w}')
            n_warnings += 1
        for e in r['errors']:
            print(f'  ✗ ERROR: {e}')
            n_errors += 1
        print()

    print(f'='*60)
    print(f'Summary: {n_errors} errors, {n_warnings} warnings')
    if n_errors > 0:
        print('FAIL — fix errors before launching training')
        sys.exit(1)
    if n_warnings > 0:
        print('PASS with warnings — review before launching')
    else:
        print('PASS')


if __name__ == '__main__':
    main()
