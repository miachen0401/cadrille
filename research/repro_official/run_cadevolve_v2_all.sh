#!/usr/bin/env bash
# Re-run CADEvolve on all 3 eval sets with the FIXED setup matching official
# (image-only prompt, processor from ckpt with resized_w/h, max_new_tokens=4000).
# Writes to *_v2 / cadevolve_v3 dirs to preserve the old (buggy) numbers for diff.
#
# Prereq: transformers==4.50.3 already installed in .venv.
# Run via .venv/bin/python (NOT uv run).

set -uo pipefail
REPO=/home/hula0401/Projects/cadrille
cd "$REPO"

# Sanity check
.venv/bin/python -c "import transformers; assert transformers.__version__.startswith('4.50.'), \
                     f'Expected 4.50.x, got {transformers.__version__}'"

LOG=$REPO/logs/cadevolve_v3.log
mkdir -p $REPO/logs

echo "=== START v3 (official Plotter + pre-norm) $(date -u +%FT%TZ) ===" | tee -a $LOG

# 1. DeepCAD-300
.venv/bin/python research/repro_official/run_cadevolve.py \
    --dataset deepcad --n-samples 300 --seed 42 \
    --batch-size 2 --max-new-tokens 4000 \
    --out eval_outputs/repro_official/deepcad_n300/cadevolve_v3 \
    2>&1 | tee -a $LOG

# 2. Fusion360-300
.venv/bin/python research/repro_official/run_cadevolve.py \
    --dataset fusion360 --n-samples 300 --seed 42 \
    --batch-size 2 --max-new-tokens 4000 \
    --out eval_outputs/repro_official/fusion360_n300/cadevolve_v3 \
    2>&1 | tee -a $LOG

# 3. cad_bench_722 (all 720)
.venv/bin/python research/repro_official/run_cadevolve.py \
    --dataset cad_bench --batch-size 2 --max-new-tokens 4000 \
    --out eval_outputs/repro_official/cad_bench_722_full/cadevolve_v3 \
    2>&1 | tee -a $LOG

echo "=== DONE $(date -u +%FT%TZ) ===" | tee -a $LOG

# Final summary line per dataset
.venv/bin/python <<'PY' 2>&1 | tee -a $LOG
import json
from pathlib import Path
import os, urllib.request

def summarize(path):
    if not Path(path).exists():
        return None
    rs = [json.loads(l) for l in open(path)]
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    return {
        'n': len(rs),
        'exec_pct': len(ok) / len(rs) * 100 if rs else 0,
        'mean_iou': sum(ious)/len(ious) if ious else None,
        'mean_cd':  sum(cds)/len(cds)   if cds  else None,
    }

paths = {
    'deepcad-300':   'eval_outputs/repro_official/deepcad_n300/cadevolve_v3/metadata.jsonl',
    'fusion360-300': 'eval_outputs/repro_official/fusion360_n300/cadevolve_v3/metadata.jsonl',
    'cad_bench_722': 'eval_outputs/repro_official/cad_bench_722_full/cadevolve_v3/metadata.jsonl',
}
old_paths = {
    'deepcad-300':   'eval_outputs/deepcad_n300/cadevolve_rl1_buggy/metadata.jsonl',
    'fusion360-300': 'eval_outputs/fusion360_n300/cadevolve_rl1_buggy/metadata.jsonl',
    'cad_bench_722': 'eval_outputs/cad_bench_722/cadevolve_rl1_buggy/metadata.jsonl',
}

print('\n=== CADEvolve v2 (fixed setup) — final summary ===\n')
print(f'{"dataset":<16} {"n":>4} {"exec":>6} {"mean_iou":>9} {"mean_cd":>8}')
print('-' * 50)
results = {}
for name, p in paths.items():
    s = summarize(p)
    results[name] = s
    if s is None:
        print(f'{name:<16}   no metadata.jsonl')
        continue
    print(f'{name:<16} {s["n"]:>4} {s["exec_pct"]:>5.1f}% '
          f'{s["mean_iou"] if s["mean_iou"] is not None else float("nan"):>9.4f} '
          f'{s["mean_cd"] if s["mean_cd"] is not None else float("nan"):>8.4f}')

# Discord post
url = os.environ.get('DISCORD_WEBHOOK_URL')
if url:
    msg = ['🔧 **CADEvolve re-run with fixed setup** (image-only prompt, '
           'processor from ckpt w/ resized_w/h, max_new_tokens=4000)',
           '',
           '```',
           f'{"dataset":<16} {"n":>4} {"exec":>6} {"mean_iou":>9}  vs old',
           '-' * 50]
    old_iou = {'deepcad-300': 0.1433, 'fusion360-300': 0.0946,
               'cad_bench_722': 0.3672}
    for name in ['deepcad-300', 'fusion360-300', 'cad_bench_722']:
        s = results.get(name)
        if not s or s['mean_iou'] is None:
            msg.append(f'{name:<16}   FAILED')
            continue
        delta = s['mean_iou'] - old_iou[name]
        sign = '+' if delta >= 0 else ''
        msg.append(f'{name:<16} {s["n"]:>4} {s["exec_pct"]:>5.1f}% '
                   f'{s["mean_iou"]:>9.4f}  ({sign}{delta:+.4f} vs old)')
    msg.append('```')
    msg.append('Old numbers from `eval_outputs/cad_bench_722/RESULTS.md`.')
    data = json.dumps({'content': '\n'.join(msg)}).encode()
    req = urllib.request.Request(url, data=data,
        headers={'Content-Type': 'application/json',
                 'User-Agent': 'cadevolve-v2-rerun/1.0'})
    try:
        urllib.request.urlopen(req, timeout=20).read()
        print('  posted to Discord ✓')
    except Exception as e:
        print(f'  Discord post failed: {e}')
PY

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
