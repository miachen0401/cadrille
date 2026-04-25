"""Aggregate op-distribution comparison of SFT pred vs GT on BenchCAD val.

Step 1 (this file): aggregate over a fixed bench subset. For each CadQuery
op (regex-matched), count items in GT / pred where it appears. Also count
distinct code hashes per item (across K samples at each temperature) to
quantify raw generation diversity.

Step 2 (later): per-item side-by-side ops diff. Not in this script yet.

Usage:
    set -a; source .env; set +a
    python -m scripts.analysis.diversity_analysis \\
        --ckpt checkpoints/sft-s4k-.../checkpoint-1000 \\
        --n-items 30 --n-samples 8 --temps 0,0.5,1.0 \\
        --out eval_outputs/diversity_smoke

Outputs:
    <out>/summary.md        — human-readable tables
    <out>/raw.jsonl         — per-(item,temp) raw codes + detected ops
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from common.model import Cadrille, collate  # noqa: E402


# ---------------------------------------------------------------------------
# Op regex — coarse but standalone (no exec). 1 pattern per op.
# ---------------------------------------------------------------------------

_OPS: dict[str, re.Pattern] = {
    # sketch / primitive
    'box':        re.compile(r'\.box\b'),
    'cylinder':   re.compile(r'\.cylinder\b'),
    'sphere':     re.compile(r'\.sphere\b'),
    'circle':     re.compile(r'\.circle\b'),
    'rect':       re.compile(r'\.rect\b'),
    'polygon':    re.compile(r'\.polygon\b'),
    'polyline':   re.compile(r'\.polyline\b'),
    'segment':    re.compile(r'\.segment\b'),
    'arc':        re.compile(r'\.(threePointArc|radiusArc|tangentArc)\b'),
    'spline':     re.compile(r'\.spline\b'),
    # sweep-style
    'extrude':    re.compile(r'\.extrude\b'),
    'revolve':    re.compile(r'\.revolve\b'),
    'sweep':      re.compile(r'\.sweep\b'),
    'loft':       re.compile(r'\.loft\b'),
    # boolean
    'cut':        re.compile(r'\.cut\b'),
    'union':      re.compile(r'\.union\b'),
    'intersect':  re.compile(r'\.intersect\b'),
    # hole family
    'hole':       re.compile(r'\.hole\b'),
    'cbore':      re.compile(r'\.cboreHole\b'),
    'csk':        re.compile(r'\.cskHole\b'),
    # finishing
    'fillet':     re.compile(r'\.fillet\b'),
    'chamfer':    re.compile(r'\.chamfer\b'),
    'shell':      re.compile(r'\.shell\b'),
    'mirror':     re.compile(r'\.mirror\b'),
    # workplane / placement
    'workplane':  re.compile(r'\.workplane\b'),
    'transformed': re.compile(r'\.transformed\b'),
    'moveTo':     re.compile(r'\.moveTo\b'),
    'translate':  re.compile(r'\.translate\b'),
    'rotate':     re.compile(r'\.rotate\b'),
    # sketch block
    'sketch':     re.compile(r'\.sketch\b'),
}


def detect_ops(code: str) -> set[str]:
    return {name for name, pat in _OPS.items() if pat.search(code)}


def code_hash(code: str) -> str:
    # Normalise whitespace so trivial re-formatting doesn't inflate diversity.
    canon = re.sub(r'\s+', ' ', code).strip()
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Load items from local benchcad val
# ---------------------------------------------------------------------------

def load_items(n: int, seed: int = 42) -> list[dict]:
    pkl = Path('data/benchcad/val.pkl')
    with pkl.open('rb') as f:
        rows = pickle.load(f)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    rows = shuffled[:n]

    out: list[dict] = []
    for row in rows:
        stem = row['uid']
        py_path = Path('data/benchcad') / row['py_path']
        png_path = Path('data/benchcad') / row['png_path']
        if not png_path.exists():
            continue
        gt_code = py_path.read_text() if py_path.exists() else ''
        out.append({
            'uid': stem,
            'gt_code': gt_code,
            'png_path': str(png_path),
        })
    print(f'loaded {len(out)} items from benchcad/val', flush=True)
    return out


# ---------------------------------------------------------------------------
# Batched generation (mode=img, pre-rendered PNG)
# ---------------------------------------------------------------------------

_GEN_KEYS = ('input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img',
             'pixel_values_videos', 'video_grid_thw')


def build_inputs(items: list[dict]) -> list[dict]:
    from PIL import Image
    return [{
        'video': [Image.open(it['png_path']).convert('RGB')],
        'description': 'Generate cadquery code',
        'file_name': it['uid'],
    } for it in items]


@torch.no_grad()
def generate_batch(model, chunk, processor, max_new_tokens, temperature, device) -> list[str]:
    batch = collate(chunk, processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]
    if temperature == 0:
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False)
    else:
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True,
                      temperature=temperature, top_p=1.0, top_k=50)
    batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items() if k in _GEN_KEYS}
    out = model.generate(**batch_gpu, **gen_kw)
    return [processor.decode(out[j, prompt_len:], skip_special_tokens=True)
            for j in range(len(chunk))]


def generate_all(items: list[dict], model, processor, device,
                 temps: list[float], n_samples: int,
                 batch_size: int, max_new_tokens: int) -> list[dict]:
    inputs = build_inputs(items)
    results: list[dict] = [{'uid': it['uid'], 'gt_code': it['gt_code'], 'by_temp': {}}
                           for it in items]
    for t in temps:
        n = 1 if t == 0 else n_samples
        per_item_codes: list[list[str]] = [[] for _ in items]
        for s in range(n):
            for start in range(0, len(items), batch_size):
                chunk = inputs[start:start + batch_size]
                codes = generate_batch(model, chunk, processor, max_new_tokens, t, device)
                for j, c in enumerate(codes):
                    per_item_codes[start + j].append(c)
            print(f'  t={t:.2f}  pass {s+1}/{n}', flush=True)
        for i, codes in enumerate(per_item_codes):
            results[i]['by_temp'][f'{t:.2f}'] = codes
    return results


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(results: list[dict], temps: list[float]) -> dict:
    """Per-op counts: how many items have the op in GT vs in pred codes.

    Counted on the UNION of all sampled codes per (item, temp). Regex-only,
    no exec.
    """
    n_items = len(results)
    # gt_counts[op] = # items whose GT has op
    gt_counts: dict[str, int] = {op: 0 for op in _OPS}
    # pred_counts[temp][op] = # items where ANY sample at this temp had op
    pred_counts: dict[str, dict[str, int]] = {f'{t:.2f}': {op: 0 for op in _OPS}
                                              for t in temps}
    # distinct codes per (item, temp) — diversity proxy
    distinct_hashes: dict[str, list[int]] = {f'{t:.2f}': [] for t in temps}

    for r in results:
        gt_ops = detect_ops(r['gt_code'])
        for op in gt_ops:
            gt_counts[op] += 1
        for t_str, codes in r['by_temp'].items():
            any_ops: set[str] = set()
            for c in codes:
                any_ops |= detect_ops(c)
            for op in any_ops:
                pred_counts[t_str][op] += 1
            distinct_hashes[t_str].append(len({code_hash(c) for c in codes}))

    # Diversity stats: for each temp, mean distinct code hashes per item
    diversity = {}
    for t_str, lst in distinct_hashes.items():
        if lst:
            diversity[t_str] = {
                'mean_distinct_per_item': sum(lst) / len(lst),
                'items': len(lst),
            }
    return {
        'n_items': n_items,
        'gt_counts': gt_counts,
        'pred_counts': pred_counts,
        'diversity': diversity,
    }


def write_summary(agg: dict, temps: list[float], out_dir: Path) -> None:
    md = [f'# Diversity analysis — n_items={agg["n_items"]}\n']

    # Diversity
    md.append('## Generation diversity (distinct code hashes per item)\n')
    md.append('| temp | mean distinct / n_samples |')
    md.append('|---|---:|')
    for t_str, d in agg['diversity'].items():
        md.append(f'| {t_str} | {d["mean_distinct_per_item"]:.2f} |')
    md.append('')

    # Op counts
    md.append('## Op presence — GT vs pred (items with op present)\n')
    header = '| op | GT |' + ''.join(f' t={t:.2f} |' for t in temps)
    align  = '|---|---:|' + '---:|' * len(temps)
    md.append(header)
    md.append(align)
    # Sort by GT count desc
    for op, _ in sorted(agg['gt_counts'].items(), key=lambda kv: -kv[1]):
        gt = agg['gt_counts'][op]
        row = f'| `{op}` | {gt} |'
        for t in temps:
            pc = agg['pred_counts'][f'{t:.2f}'][op]
            row += f' {pc} |'
        md.append(row)
    md.append('')

    # Op delta (|pred-gt| at greedy, to flag missing/extra ops)
    greedy = f'{temps[0]:.2f}' if temps else None
    if greedy:
        md.append(f'## Op delta at temp={greedy} (pred minus GT)\n')
        md.append('| op | gt | pred | delta |')
        md.append('|---|---:|---:|---:|')
        for op, _ in sorted(agg['gt_counts'].items(), key=lambda kv: -kv[1]):
            gt = agg['gt_counts'][op]
            pc = agg['pred_counts'][greedy][op]
            md.append(f'| `{op}` | {gt} | {pc} | {pc - gt:+d} |')

    (out_dir / 'summary.md').write_text('\n'.join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base-model', default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--n-items', type=int, default=30)
    ap.add_argument('--n-samples', type=int, default=8)
    ap.add_argument('--temps', default='0,0.5,1.0')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    temps = [float(t) for t in args.temps.split(',')]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.n_items, seed=args.seed)

    print(f'Loading model from {args.ckpt} ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')
    model = Cadrille.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')
    model.eval()
    device = next(model.parameters()).device

    results = generate_all(items, model, processor, device, temps,
                           args.n_samples, args.batch_size, args.max_new_tokens)
    # Persist raw for future per-item step
    with (out_dir / 'raw.jsonl').open('w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    agg = aggregate(results, temps)
    (out_dir / 'aggregate.json').write_text(json.dumps(agg, indent=2))
    write_summary(agg, temps, out_dir)
    print(f'\nSaved:\n  {out_dir}/summary.md\n  {out_dir}/aggregate.json\n  {out_dir}/raw.jsonl',
          flush=True)


if __name__ == '__main__':
    main()
