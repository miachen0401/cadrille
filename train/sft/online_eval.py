"""Online IoU eval during SFT — wandb metrics mirror the RL training loop.

At every `eval_steps` tick, greedy-generate on a fixed subset of each eval
dataset (benchcad val + deepcad test + fusion360 test), exec each code,
compute IoU / CD, and log per-dataset metrics under the same key names as
train/rl/eval.py:

    eval/img/{dataset}/IoU mean
    eval/img/{dataset}/IoU median
    eval/img/{dataset}/CD mean
    eval/img/{dataset}/CD median
    eval/img/{dataset}/Failures fraction

Wired via a TrainerCallback so the HF Trainer's default `eval_loss` path is
untouched — this runs on the same schedule but emits IoU-based metrics
alongside.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from common.model import collate
from common.metrics import compute_metrics


# Op regex — kept in lockstep with scripts/analysis/diversity_analysis.py::_OPS.
# Inlined here so callback imports stay light (no model code).
_OPS: dict[str, re.Pattern] = {
    'box':         re.compile(r'\.box\b'),
    'cylinder':    re.compile(r'\.cylinder\b'),
    'sphere':      re.compile(r'\.sphere\b'),
    'circle':      re.compile(r'\.circle\b'),
    'rect':        re.compile(r'\.rect\b'),
    'polygon':     re.compile(r'\.polygon\b'),
    'polyline':    re.compile(r'\.polyline\b'),
    'segment':     re.compile(r'\.segment\b'),
    'arc':         re.compile(r'\.(threePointArc|radiusArc|tangentArc)\b'),
    'spline':      re.compile(r'\.spline\b'),
    'extrude':     re.compile(r'\.extrude\b'),
    'revolve':     re.compile(r'\.revolve\b'),
    'sweep':       re.compile(r'\.sweep\b'),
    'loft':        re.compile(r'\.loft\b'),
    'cut':         re.compile(r'\.cut\b'),
    'union':       re.compile(r'\.union\b'),
    'intersect':   re.compile(r'\.intersect\b'),
    'hole':        re.compile(r'\.hole\b'),
    'cbore':       re.compile(r'\.cboreHole\b'),
    'csk':         re.compile(r'\.cskHole\b'),
    'fillet':      re.compile(r'\.fillet\b'),
    'chamfer':     re.compile(r'\.chamfer\b'),
    'shell':       re.compile(r'\.shell\b'),
    'mirror':      re.compile(r'\.mirror\b'),
    'workplane':   re.compile(r'\.workplane\b'),
    'transformed': re.compile(r'\.transformed\b'),
    'moveTo':      re.compile(r'\.moveTo\b'),
    'translate':   re.compile(r'\.translate\b'),
    'rotate':      re.compile(r'\.rotate\b'),
    'sketch':      re.compile(r'\.sketch\b'),
}


def _code_hash(code: str) -> str:
    canon = re.sub(r'\s+', ' ', code).strip()
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def _op_label_matrix(codes: list[str]) -> np.ndarray:
    """Return (N, 30) bool matrix: y[i, k] = op-k present in code-i."""
    op_pats = list(_OPS.values())
    return np.array(
        [[bool(pat.search(c)) for pat in op_pats] for c in codes],
        dtype=bool,
    )


def _diversity_stats(codes: list[str]) -> dict:
    """Pred-only diversity stats — works without GT codes (DeepCAD/Fusion360 case).

    Returns op-presence rates, distinct-code fraction, distinct-op count, and
    mean code length. No comparison to GT.
    """
    if not codes:
        return {}
    n = len(codes)
    Y = _op_label_matrix(codes)                        # (n, 30) bool
    op_counts = Y.sum(axis=0)                          # (30,)
    op_names = list(_OPS.keys())
    hashes = {_code_hash(c) for c in codes}
    char_lens = [len(c) for c in codes]
    out = {f'op/{op_names[k]}': float(op_counts[k] / n) for k in range(len(op_names))}
    out['distinct_ops']        = int((op_counts > 0).sum())
    out['distinct_codes_frac'] = len(hashes) / n
    out['code_chars_mean']     = float(np.mean(char_lens))
    out['pred_count_mean']     = float(Y.sum(axis=1).mean())
    return out


def _multilabel_op_metrics(pred_codes: list[str], gt_codes: list[str],
                           freqs: np.ndarray | None = None,
                           rare_op_idx: np.ndarray | None = None,
                           label_smoothing: float = 0.01) -> dict:
    """Multi-label classification metrics treating ops as a 30-class label set.

    For each item we have a pred label vector y_pred ∈ {0,1}^30 (which ops the
    generated code uses) and a gt label vector y_gt ∈ {0,1}^30 (which ops the
    reference code uses). Standard multi-label metrics:

      Sample-level (mean over items):
        op_jaccard          - |pred ∩ gt| / |pred ∪ gt|
        op_hamming_sim      - 1 - mean bit-disagreement / 30
        op_subset_acc       - exact set equality

      Op-level (one-vs-rest, only where GT support > 0):
        op_recall/{op}      - TP / (TP + FN)              [mode-collapse per op]
        op_precision/{op}   - TP / (TP + FP)              [hallucination per op]
        op_f1/{op}

      Macro aggregate:
        op_macro_recall     - mean per-op recall over GT-supported ops
                              [single-scalar mode-collapse summary]
        op_macro_precision
        op_macro_f1
        ops_zero_recall_count    - # ops in GT but never in pred
        ops_zero_precision_count - # ops in pred but never in GT (hallucination)
        op_l1_dist          - L1 between pred op-rate and gt op-rate vectors

      Counts:
        gt_count_mean       - avg # ops per GT code
        pred_count_mean     - avg # ops per pred code
    """
    n = len(pred_codes)
    if n == 0 or n != len(gt_codes):
        return {}
    op_names = list(_OPS.keys())
    K = len(op_names)
    P = _op_label_matrix(pred_codes)                   # (n, K) bool
    G = _op_label_matrix(gt_codes)                     # (n, K) bool

    # Per-item set metrics
    inter = (P & G).sum(axis=1).astype(float)          # (n,)
    union = (P | G).sum(axis=1).astype(float)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1.0), 1.0)  # both empty -> 1
    hamming_sim = 1.0 - (P ^ G).sum(axis=1) / float(K)
    subset_acc = float(((P == G).all(axis=1)).mean())

    # Per-op confusion counts
    tp = (P & G).sum(axis=0).astype(float)             # (K,)
    fp = (P & ~G).sum(axis=0).astype(float)
    fn = (~P & G).sum(axis=0).astype(float)
    gt_support = G.sum(axis=0)                         # (K,) int
    pred_support = P.sum(axis=0)

    # Per-op recall/precision/F1 — only emit where the denominator is defined.
    per_op = {}
    recalls, precisions, f1s = [], [], []
    zero_recall = 0
    zero_precision = 0
    for k, op in enumerate(op_names):
        if gt_support[k] > 0:
            r = tp[k] / (tp[k] + fn[k])
            per_op[f'op_recall/{op}'] = float(r)
            recalls.append(r)
            if r == 0.0:
                zero_recall += 1
        if pred_support[k] > 0:
            p = tp[k] / (tp[k] + fp[k])
            per_op[f'op_precision/{op}'] = float(p)
            precisions.append(p)
            if p == 0.0:
                zero_precision += 1
        if gt_support[k] > 0 and pred_support[k] > 0:
            r = tp[k] / (tp[k] + fn[k])
            p = tp[k] / (tp[k] + fp[k])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_op[f'op_f1/{op}'] = float(f1)
            f1s.append(f1)
        elif gt_support[k] > 0:
            # GT had it, pred never did -> F1 = 0 (counted toward macro)
            per_op[f'op_f1/{op}'] = 0.0
            f1s.append(0.0)

    pred_rate = P.mean(axis=0)
    gt_rate = G.mean(axis=0)
    l1_dist = float(np.abs(pred_rate - gt_rate).sum())

    # Per-case weighted cosine — user-requested algorithm.
    # Treat each (gt_vec, pred_vec) ∈ {0,1}^K as a multi-hot vector. Weight per
    # op = -log(P_k + ε), so rare ops dominate both numerator and denominator.
    # Loss = 1 - cos_w. Range [0, 1]. Lower is better.
    eps = 1e-6
    if freqs is None or freqs.shape[0] != K:
        freqs_use = np.full(K, 1.0 / max(K, 1))
    else:
        freqs_use = freqs
    w = -np.log(np.clip(freqs_use, eps, 1.0))      # (K,) ≥ 0
    G_f = G.astype(np.float64)
    P_f = P.astype(np.float64)
    num   = (w * G_f * P_f).sum(axis=1)            # (n,)
    den_g = np.sqrt((w * G_f).sum(axis=1))
    den_p = np.sqrt((w * P_f).sum(axis=1))
    denom = den_g * den_p
    cos_w = np.where(denom > 0, num / np.maximum(denom, eps), 0.0)
    case_loss_w = 1.0 - cos_w                      # (n,)

    out = {
        'op_jaccard':               float(jaccard.mean()),
        'op_hamming_sim':           float(hamming_sim.mean()),
        'op_subset_acc':            subset_acc,
        'op_macro_recall':          float(np.mean(recalls))    if recalls    else 0.0,
        'op_macro_precision':       float(np.mean(precisions)) if precisions else 0.0,
        'op_macro_f1':              float(np.mean(f1s))        if f1s        else 0.0,
        'ops_zero_recall_count':    int(zero_recall),
        'ops_zero_precision_count': int(zero_precision),
        'op_l1_dist':               l1_dist,
        'op_cos_weighted':          float(cos_w.mean()),
        'op_loss_cos_weighted':     float(case_loss_w.mean()),
        'gt_count_mean':            float(G.sum(axis=1).mean()),
    }
    out.update(per_op)

    # Rare-op cohort aggregates — same per-op math, restricted to rare ops only.
    # `rare_op_idx` is a (K,) bool mask identifying rare ops in the global P
    # (e.g. 0 < P_k ≤ 0.20). Catches mode collapse on chamfer/revolve/hole
    # without being diluted by easy ops like extrude.
    if rare_op_idx is not None and rare_op_idx.any():
        rare_recalls, rare_f1s = [], []
        for k in np.where(rare_op_idx)[0]:
            if gt_support[k] > 0:
                r = tp[k] / (tp[k] + fn[k])
                rare_recalls.append(r)
                if pred_support[k] > 0 and (tp[k] + fp[k]) > 0:
                    p = tp[k] / (tp[k] + fp[k])
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                else:
                    f1 = 0.0
                rare_f1s.append(f1)
        out['rare_op_macro_recall']   = float(np.mean(rare_recalls)) if rare_recalls else 0.0
        out['rare_op_macro_f1']       = float(np.mean(rare_f1s))     if rare_f1s     else 0.0
        out['rare_op_pred_rate_mean'] = float(P[:, rare_op_idx].mean())
        out['rare_op_gt_rate_mean']   = float(G[:, rare_op_idx].mean())
        # Number of rare ops in GT this batch that pred completely missed.
        out['rare_op_zero_recall_count'] = int(
            sum(1 for r in rare_recalls if r == 0.0))
    return out


# Online eval has 5 buckets:
#   ops + IoU:  BenchCAD val (full triplet: img + STL + .py)
#   ops only:   recode20k train, text2cad train
#   IoU only:   DeepCAD test, Fusion360 test (mesh-only datasets, no GT code)
_SUBSETS_DEFAULT = ('benchcad_val', 'recode20k_train', 'text2cad_train',
                    'deepcad_test', 'fusion360_test')


def _seeded_sample(items, n, seed=42):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[:n]


def _load_benchcad_val(n: int, seed: int) -> list[dict]:
    """BenchCAD val — img modality, has STL + GT .py. Used for both ops and IoU."""
    from PIL import Image
    root_p = Path('data/benchcad')
    pkl = root_p / 'val.pkl'
    if not pkl.exists():
        return []
    with pkl.open('rb') as f:
        rows = pickle.load(f)
    rows = _seeded_sample(rows, n, seed)
    out = []
    for r in rows:
        png = root_p / r['png_path']
        stl = root_p / r['mesh_path']
        py  = root_p / r['py_path']
        if not (png.exists() and stl.exists() and py.exists()):
            continue
        out.append({
            '_modality': 'img',
            '_dataset_label': 'BenchCAD val',
            '_gt_code': py.read_text(),
            'file_name': r['uid'],
            'gt_mesh_path': str(stl),
            'video': [Image.open(png).convert('RGB')],
            'description': 'Generate cadquery code',
        })
    return out


def _load_recode20k_train(n: int, seed: int) -> list[dict]:
    """cad-recode-20k train — img modality, has GT .py (no STL → no IoU)."""
    from PIL import Image
    root_p = Path('data/cad-recode-20k')
    pkl = root_p / 'train.pkl'
    if not pkl.exists():
        return []
    with pkl.open('rb') as f:
        rows = pickle.load(f)
    rows = _seeded_sample(rows, n, seed)
    out = []
    for r in rows:
        png = root_p / r['png_path']
        py  = root_p / r['py_path']
        if not (png.exists() and py.exists()):
            continue
        out.append({
            '_modality': 'img',
            '_dataset_label': 'recode20k train',
            '_gt_code': py.read_text(),
            'file_name': r['uid'],
            'video': [Image.open(png).convert('RGB')],
            'description': 'Generate cadquery code',
        })
    return out


def _load_text2cad_train(n: int, seed: int) -> list[dict]:
    """text2cad train — text modality (description → code), GT in cadquery/."""
    root_p = Path('data/text2cad')
    pkl = root_p / 'train.pkl'
    cq_dir = root_p / 'cadquery'
    if not (pkl.exists() and cq_dir.is_dir()):
        return []
    with pkl.open('rb') as f:
        rows = pickle.load(f)
    rows = _seeded_sample(rows, n, seed)
    out = []
    for r in rows:
        py = cq_dir / f'{r["uid"]}.py'
        if not py.exists():
            continue
        out.append({
            '_modality': 'text',
            '_dataset_label': 'text2cad train',
            '_gt_code': py.read_text(),
            'file_name': r['uid'],
            'description': r['description'],
        })
    return out


def _load_stl_dir_test(root: str, label: str, n: int, seed: int) -> list[dict]:
    """DeepCAD/Fusion360 test — img + STL only, no GT code → IoU only."""
    from PIL import Image
    rootp = Path(root)
    stls = sorted(rootp.glob('*.stl'))
    stls = _seeded_sample(stls, n, seed)
    out = []
    for stl in stls:
        png = stl.with_name(stl.stem + '_render.png')
        if not png.exists():
            continue
        out.append({
            '_modality': 'img',
            '_dataset_label': label,
            'file_name': stl.stem,
            'gt_mesh_path': str(stl),
            'video': [Image.open(png).convert('RGB')],
            'description': 'Generate cadquery code',
        })
    return out


_SOURCE_LOADERS = {
    'benchcad_val':    lambda n, s: _load_benchcad_val(n, s),
    'recode20k_train': lambda n, s: _load_recode20k_train(n, s),
    'text2cad_train':  lambda n, s: _load_text2cad_train(n, s),
    'deepcad_test':    lambda n, s: _load_stl_dir_test(
                          'data/deepcad_test_mesh', 'DeepCAD test', n, s),
    'fusion360_test':  lambda n, s: _load_stl_dir_test(
                          'data/fusion360_test_mesh', 'Fusion360 test', n, s),
}


def _list_train_pyfiles(source: str) -> list[Path]:
    """Resolve the full list of training-corpus .py files for a source."""
    if source == 'benchcad':
        root = Path('data/benchcad')
        pkl = root / 'train.pkl'
        if not pkl.exists(): return []
        rows = pickle.load(pkl.open('rb'))
        return [root / r['py_path'] for r in rows]
    if source == 'recode20k':
        root = Path('data/cad-recode-20k')
        pkl = root / 'train.pkl'
        if not pkl.exists(): return []
        rows = pickle.load(pkl.open('rb'))
        return [root / r['py_path'] for r in rows]
    if source == 'text2cad':
        root = Path('data/text2cad/cadquery')
        pkl = Path('data/text2cad/train.pkl')
        if not pkl.exists() or not root.is_dir(): return []
        rows = pickle.load(pkl.open('rb'))
        return [root / f'{r["uid"]}.py' for r in rows]
    return []


def _compute_global_op_freqs(mix_weights: dict, n_total: int = 200,
                             seed: int = 42,
                             cache_dir: Path = Path('data/online_eval_cache'),
                             ) -> np.ndarray:
    """Per-op presence frequency P_k over a `n_total`-sample mix of training
    sources, weighted by `mix_weights` (e.g. {benchcad:4, recode20k:1,
    text2cad:1}).

    Returns shape (K,) bound in [0, 1]. Cached to disk keyed on
    (mix_weights, n_total, seed) so re-runs of the same SFT config are instant.
    """
    # Active sources only (weight > 0); ignore 'recode' alias for legacy keys.
    active = {k: float(v) for k, v in (mix_weights or {}).items()
              if k in {'benchcad', 'recode20k', 'text2cad'} and float(v) > 0}
    if not active:
        return np.zeros(len(_OPS))

    # Cache key — same mix → same P
    key = '_'.join(f'{k}{int(active[k])}' for k in sorted(active))
    cache_path = cache_dir / f'global_op_freqs_{key}_n{n_total}_s{seed}.npz'
    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=False)
            if len(d['freqs']) == len(_OPS):
                return d['freqs']
        except Exception:
            pass

    # Quota per source ∝ weight; round to int summing to n_total
    total_w = sum(active.values())
    quotas = {src: int(round(n_total * w / total_w)) for src, w in active.items()}
    diff = n_total - sum(quotas.values())
    if diff:
        # Hand the leftover (±1) to the largest-weight source
        top = max(active, key=lambda s: active[s])
        quotas[top] += diff

    op_pats = list(_OPS.values())
    K = len(op_pats)
    counts = np.zeros(K, dtype=np.int64)
    total_seen = 0
    for src, n_src in quotas.items():
        if n_src <= 0:
            continue
        files = _list_train_pyfiles(src)
        if not files:
            continue
        files = _seeded_sample(files, n_src, seed)
        for f in files:
            if not f.exists(): continue
            try: code = f.read_text()
            except Exception: continue
            for k, pat in enumerate(op_pats):
                if pat.search(code):
                    counts[k] += 1
            total_seen += 1
    if total_seen == 0:
        return np.zeros(K)
    freqs = counts.astype(np.float64) / total_seen

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, freqs=freqs)
    quota_str = ', '.join(f'{s}={quotas[s]}' for s in sorted(quotas))
    top3 = sorted(enumerate(_OPS.keys()), key=lambda kv: -freqs[kv[0]])[:3]
    print(f'[online-eval] global op freqs over {total_seen} mixed samples '
          f'({quota_str}); top-3: '
          + ', '.join(f'{n}={freqs[i]:.2f}' for i, n in top3),
          flush=True)
    return freqs


def load_online_eval_subsets(n_per: int = 20, seed: int = 42,
                             subsets=None) -> list[dict]:
    """Sample n_per items from each subset; defaults to the 5 buckets above."""
    sources = subsets if subsets is not None else _SUBSETS_DEFAULT
    if isinstance(sources, dict):
        sources = list(sources.keys())  # back-compat: dict form, keys only
    out = []
    for src in sources:
        loader = _SOURCE_LOADERS.get(src)
        if loader is None:
            print(f'[online-eval] unknown source: {src}', flush=True)
            continue
        items = loader(n_per, seed)
        out += items
        print(f'[online-eval] source {src}: {len(items)} items', flush=True)
    return out


@torch.no_grad()
def _run_max_iou_at_temp(model, processor, examples: list[dict],
                         k: int, temperature: float,
                         eval_batch_size: int, reward_workers: int,
                         max_new_tokens: int, eval_timeout: float,
                         seed: int = 42) -> dict:
    """Sample K codes per item at `temperature`, return per-bucket max-IoU metrics.

    Only runs on examples with `gt_mesh_path` (BenchCAD val + DeepCAD test +
    Fusion360 test). For each item: generate K samples, score IoU on each,
    take max. Bucket-aggregate to:
      - max_iou@{K} (t={T})        — mean of per-item max IoU
      - pass_iou_0.5@{K} (t={T})   — fraction of items with max IoU > 0.5
      - pass_iou_0.7@{K} (t={T})   — fraction of items with max IoU > 0.7
      - max_iou_failures@{K} (t={T}) — fraction of items where ALL K samples failed

    Seed is fixed across evals so the K stochastic paths are comparable
    over training steps (not the same path — sampling diverges by step
    1 — but the seed reduces variance from the eval RNG itself).
    """
    iou_items = [(i, ex) for i, ex in enumerate(examples) if 'gt_mesh_path' in ex]
    if not iou_items or k <= 0:
        return {}

    device = next(model.parameters()).device
    # Seed sampling RNG, but isolate from the global generator so eval doesn't
    # rewrite training's CPU/CUDA RNG state (would cause dropout patterns to
    # repeat after every eval tick). `fork_rng` snapshots-and-restores; we
    # only need CPU + the active CUDA device.
    cuda_devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        return _run_max_iou_at_temp_inner(
            model, processor, examples, k, temperature,
            eval_batch_size, reward_workers, max_new_tokens, eval_timeout,
            iou_items, device,
        )


def _run_max_iou_at_temp_inner(model, processor, examples, k, temperature,
                               eval_batch_size, reward_workers, max_new_tokens,
                               eval_timeout, iou_items, device):
    """Sampling+scoring body — split out so the outer can wrap fork_rng."""
    # Build a flat list of (orig_idx, sample_idx, ex) for batched generation.
    work = [(orig, s, ex) for orig, ex in iou_items for s in range(k)]
    codes_by_pair: dict[tuple[int, int], str] = {}

    for batch_start in range(0, len(work), eval_batch_size):
        chunk = work[batch_start:batch_start + eval_batch_size]
        ex_chunk = [x[2] for x in chunk]
        collate_items = [{kk: vv for kk, vv in ex.items()
                          if not kk.startswith('_') and kk != 'gt_mesh_path'}
                         for ex in ex_chunk]
        batch = collate(collate_items, processor=processor, n_points=256, eval=True)
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None

        gen_kw = dict(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            point_clouds=batch['point_clouds'].to(device),
            is_pc=batch['is_pc'].to(device),
            is_img=batch['is_img'].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=None,
            top_k=None,
            bad_words_ids=[[model.config.video_token_id]],
        )
        if batch.get('pixel_values_videos') is not None:
            gen_kw['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
            gen_kw['video_grid_thw']       = batch['video_grid_thw'].to(device)

        generated_ids = model.generate(**gen_kw)
        prompt_len = batch['input_ids'].shape[1]
        for j, (orig_idx, sample_idx, _ex) in enumerate(chunk):
            code = processor.decode(
                generated_ids[j, prompt_len:], skip_special_tokens=True)
            codes_by_pair[(orig_idx, sample_idx)] = code

    # Score IoU on each (orig, sample) pair
    def _score(orig_idx, sample_idx):
        ex = examples[orig_idx]
        code = codes_by_pair[(orig_idx, sample_idx)]
        iou_reward, _cd = compute_metrics(
            code, ex['gt_mesh_path'],
            timeout=eval_timeout, use_pool=True)
        return orig_idx, sample_idx, iou_reward

    pair_iou: dict[tuple[int, int], float] = {}
    with ThreadPoolExecutor(max_workers=reward_workers) as pool:
        futures = [pool.submit(_score, o, s) for (o, s) in codes_by_pair]
        for fut in as_completed(futures):
            o, s, iou = fut.result()
            pair_iou[(o, s)] = iou

    # Aggregate per bucket
    buckets = defaultdict(lambda: {'max_ious': [], 'pass_05': 0,
                                   'pass_07': 0, 'all_fail': 0, 'total': 0})
    for orig_idx, ex in iou_items:
        ious = [pair_iou.get((orig_idx, s), -2.0) for s in range(k)]
        valid = [iou for iou in ious if iou > -1.0]
        if valid:
            max_iou = max(valid)
        else:
            max_iou = 0.0  # every sample failed exec
        key = (ex.get('_modality', 'img'), ex.get('_dataset_label', '?'))
        b = buckets[key]
        b['max_ious'].append(max_iou)
        b['total'] += 1
        if max_iou > 0.5: b['pass_05'] += 1
        if max_iou > 0.7: b['pass_07'] += 1
        if not valid:    b['all_fail'] += 1

    tag = f'@{k} (t={temperature})'
    out = {}
    for (mod, label), b in buckets.items():
        prefix = f'eval/{mod}/{label}'
        out[f'{prefix}/max_iou{tag}']        = float(np.mean(b['max_ious']))
        out[f'{prefix}/pass_iou_0.5{tag}']   = b['pass_05'] / max(b['total'], 1)
        out[f'{prefix}/pass_iou_0.7{tag}']   = b['pass_07'] / max(b['total'], 1)
        out[f'{prefix}/max_iou_failures{tag}'] = b['all_fail'] / max(b['total'], 1)
    return out


@torch.no_grad()
def run_online_eval(model, processor, examples: list[dict],
                    eval_batch_size: int = 8,
                    reward_workers: int = 8,
                    max_new_tokens: int = 768,
                    eval_timeout: float = 30.0,
                    global_freqs: np.ndarray | None = None,
                    rare_op_idx: np.ndarray | None = None,
                    max_iou_k: int = 0,
                    max_iou_temperature: float = 1.0,
                    max_iou_seed: int = 42) -> dict:
    """Greedy eval → per-(modality, dataset) IoU / CD / Failures. wandb-ready dict."""
    if not examples:
        return {}
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # Avoid GC interference with generate() image handling (mirror RL eval).
    had_gc = getattr(model, 'is_gradient_checkpointing', False)
    if had_gc:
        model.gradient_checkpointing_disable()

    try:
        # Sort by modality so each batch is single-modality (avoids edge cases
        # in collate when mixing img+text items in one chunk). Track the
        # original index so per-item results land in the right slot.
        sorted_idx = sorted(range(len(examples)),
                            key=lambda j: examples[j].get('_modality', 'img'))
        sorted_examples = [examples[j] for j in sorted_idx]

        all_codes = [''] * len(examples)
        n = len(examples)
        for i in range(0, n, eval_batch_size):
            chunk = sorted_examples[i:i + eval_batch_size]
            chunk_orig_idx = sorted_idx[i:i + eval_batch_size]
            collate_items = [{k: v for k, v in ex.items() if not k.startswith('_')
                              and k != 'gt_mesh_path'} for ex in chunk]
            batch = collate(collate_items, processor=processor, n_points=256, eval=True)
            if hasattr(model, 'rope_deltas'):
                model.rope_deltas = None

            gen_kw = dict(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                point_clouds=batch['point_clouds'].to(device),
                is_pc=batch['is_pc'].to(device),
                is_img=batch['is_img'].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                bad_words_ids=[[model.config.video_token_id]],
            )
            if batch.get('pixel_values_videos') is not None:
                gen_kw['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
                gen_kw['video_grid_thw']       = batch['video_grid_thw'].to(device)

            generated_ids = model.generate(**gen_kw)
            prompt_len = batch['input_ids'].shape[1]
            for j in range(len(chunk)):
                all_codes[chunk_orig_idx[j]] = processor.decode(
                    generated_ids[j, prompt_len:], skip_special_tokens=True)

        buckets = defaultdict(lambda: {'ious': [], 'cds': [], 'failures': 0,
                                       'total': 0, 'codes': [], 'gt_codes': [],
                                       'has_iou': False})

        def _score(idx):
            ex = examples[idx]
            if 'gt_mesh_path' not in ex:
                return idx, None, None
            iou_reward, cd = compute_metrics(
                all_codes[idx], ex['gt_mesh_path'],
                timeout=eval_timeout, use_pool=True)
            return idx, iou_reward, cd

        # Bucket non-IoU items first (text2cad/recode20k) — no need to schedule
        # them on the cadquery worker pool.
        for idx in range(n):
            ex = examples[idx]
            key = (ex.get('_modality', 'img'), ex.get('_dataset_label', '?'))
            buckets[key]['total'] += 1
            buckets[key]['codes'].append(all_codes[idx])
            buckets[key]['gt_codes'].append(ex.get('_gt_code'))

        with ThreadPoolExecutor(max_workers=reward_workers) as pool:
            futures = [pool.submit(_score, i) for i in range(n)
                       if 'gt_mesh_path' in examples[i]]
            for fut in as_completed(futures):
                idx, iou_reward, cd = fut.result()
                ex = examples[idx]
                key = (ex.get('_modality', 'img'), ex.get('_dataset_label', '?'))
                buckets[key]['has_iou'] = True
                if iou_reward <= -1.0:
                    buckets[key]['failures'] += 1
                else:
                    buckets[key]['ious'].append(iou_reward)
                    if cd is not None:
                        buckets[key]['cds'].append(cd)

        out = {}
        for (mod, label), b in buckets.items():
            prefix = f'eval/{mod}/{label}'
            mean_iou = exec_rate = None
            if b['has_iou']:
                ious, cds = b['ious'], b['cds']
                iou_total  = len(ious) + b['failures']     # only IoU-scored items
                fail_frac  = b['failures'] / iou_total if iou_total else 0.0
                exec_rate  = 1.0 - fail_frac
                mean_iou   = float(np.mean(ious))   if ious else 0.0
                median_iou = float(np.median(ious)) if ious else 0.0
                mean_cd    = float(np.mean(cds))    if cds  else float('nan')
                median_cd  = float(np.median(cds))  if cds  else float('nan')
                out[f'{prefix}/IoU mean']          = mean_iou
                out[f'{prefix}/IoU median']        = median_iou
                out[f'{prefix}/CD mean']           = mean_cd
                out[f'{prefix}/CD median']         = median_cd
                out[f'{prefix}/Failures fraction'] = fail_frac
                out[f'{prefix}/exec_rate']         = exec_rate
            # Pred-only diversity (always available)
            div = _diversity_stats(b['codes'])
            for k, v in div.items():
                out[f'{prefix}/{k}'] = v
            # Multi-label op classification (needs GT codes for every item)
            gt_codes = b['gt_codes']
            ml_loss_w = None
            ml_recall = None
            ml_rare_recall = None
            if gt_codes and all(g is not None for g in gt_codes):
                ml = _multilabel_op_metrics(b['codes'], gt_codes,
                                            freqs=global_freqs,
                                            rare_op_idx=rare_op_idx)
                for k, v in ml.items():
                    out[f'{prefix}/{k}'] = v
                ml_loss_w = ml.get('op_loss_cos_weighted')
                ml_recall = ml.get('op_macro_recall')
                ml_rare_recall = ml.get('rare_op_macro_recall')
            iou_part = (f'IoU={mean_iou:.3f}  exec={exec_rate*100:.1f}%  '
                        if mean_iou is not None else '')
            ml_part = (f'op_loss_w={ml_loss_w:.3f}  recall={ml_recall:.3f}  '
                       f'rare_recall={ml_rare_recall:.3f}  '
                       if ml_loss_w is not None else '')
            print(f'  [{mod}/{label}] {ml_part}{iou_part}'
                  f'distinct_ops={div.get("distinct_ops", 0)}  '
                  f'distinct_codes={div.get("distinct_codes_frac", 0):.2f}  '
                  f'(n={b["total"]})', flush=True)

        # Optional: max IoU @ K samples at temperature T on IoU-bucketed items.
        # Cost is K× the greedy IoU pass time; defaults to 0 (off).
        if max_iou_k > 0:
            print(f'[online-eval] running max_iou@{max_iou_k} (t={max_iou_temperature}) '
                  f'on IoU buckets ...', flush=True)
            mi = _run_max_iou_at_temp(
                model, processor, examples,
                k=max_iou_k, temperature=max_iou_temperature,
                eval_batch_size=eval_batch_size,
                reward_workers=reward_workers,
                max_new_tokens=max_new_tokens,
                eval_timeout=eval_timeout,
                seed=max_iou_seed,
            )
            out.update(mi)
            tag = f'@{max_iou_k} (t={max_iou_temperature})'
            for k_, v in sorted(mi.items()):
                if k_.endswith(f'/max_iou{tag}'):
                    label = k_.split('/')[-2]
                    pass_5 = mi.get(k_.replace(f'/max_iou{tag}',
                                               f'/pass_iou_0.5{tag}'))
                    print(f'  [{label}] max_iou{tag}={v:.3f}'
                          + (f'  pass>0.5={pass_5*100:.1f}%' if pass_5 is not None
                             else ''),
                          flush=True)
        return out

    finally:
        if had_gc:
            model.gradient_checkpointing_enable()
        if was_training:
            model.train()


class OnlineIoUEvalCallback(TrainerCallback):
    """Run IoU eval every `eval_steps` on the same schedule as HF Trainer's loss eval.

    Uses HF Trainer's on_evaluate hook so metrics land in the *same* log dict that
    eval_loss goes into — wandb picks them up without an extra call.
    """

    def __init__(self, processor, n_per_dataset: int = 20, seed: int = 42,
                 subsets: dict | None = None, eval_batch_size: int = 8,
                 reward_workers: int = 8, max_new_tokens: int = 768,
                 eval_timeout: float = 30.0,
                 mix_weights: dict | None = None,
                 freq_sample_n: int = 200,
                 max_iou_k: int = 8,
                 max_iou_temperature: float = 1.0,
                 max_iou_seed: int = 42,
                 max_iou_every_n_evals: int = 1):
        self.processor = processor
        self.n_per_dataset = n_per_dataset
        self.seed = seed
        self.subsets = subsets
        self.eval_batch_size = eval_batch_size
        self.reward_workers = reward_workers
        self.max_new_tokens = max_new_tokens
        self.eval_timeout = eval_timeout
        self.mix_weights = mix_weights or {}
        self.freq_sample_n = freq_sample_n
        self.rare_op_threshold = 0.20    # 0 < P_k ≤ 0.20 → "rare"
        self.max_iou_k = max_iou_k
        self.max_iou_temperature = max_iou_temperature
        self.max_iou_seed = max_iou_seed
        # Throttle: only run max_iou pass every Nth eval (cost is K× greedy
        # generation; on a 5-min eval, K=8 adds ~5 min). 1 = every eval.
        self.max_iou_every_n_evals = max(1, int(max_iou_every_n_evals))
        self._eval_count = 0
        self._examples = None
        self._global_freqs = None
        self._rare_op_idx = None

    def _ensure_examples(self):
        if self._examples is None:
            self._examples = load_online_eval_subsets(
                n_per=self.n_per_dataset, seed=self.seed, subsets=self.subsets)
            print(f'[online-eval] loaded {len(self._examples)} examples across datasets',
                  flush=True)
            # One-time global op-frequency scan: sample `freq_sample_n` codes
            # from the SFT training mix (proportional to mix_weights), count
            # per-op presence rate. Used as inverse-freq weight w = -log(P)
            # in the per-case weighted-cosine loss.
            self._global_freqs = _compute_global_op_freqs(
                self.mix_weights, n_total=self.freq_sample_n, seed=self.seed)
            # Rare-op cohort: ops that appear in the training mix but are uncommon
            # (chamfer/revolve/hole etc). Tracked separately so mode collapse
            # on these is visible without being diluted by extrude/sketch.
            self._rare_op_idx = (
                (self._global_freqs > 0)
                & (self._global_freqs <= self.rare_op_threshold)
            )
            op_names = list(_OPS.keys())
            rare_names = [f'{op_names[k]}({self._global_freqs[k]:.2f})'
                          for k in np.where(self._rare_op_idx)[0]]
            print(f'[online-eval] rare ops (P ≤ {self.rare_op_threshold:.2f}, '
                  f'{int(self._rare_op_idx.sum())}): {rare_names}',
                  flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after HF Trainer's default eval_loss pass. Append our IoU metrics."""
        if not state.is_world_process_zero:
            return
        model = kwargs.get('model')
        if model is None:
            return
        self._ensure_examples()
        print(f'[online-eval] step={state.global_step} running IoU eval ...', flush=True)

        # Decide whether to run max_iou@K this tick (every Nth eval).
        self._eval_count += 1
        run_max_iou = (
            self.max_iou_k > 0
            and (self._eval_count % self.max_iou_every_n_evals == 0)
        )
        max_iou_k_now = self.max_iou_k if run_max_iou else 0

        iou_metrics = run_online_eval(
            model, self.processor, self._examples,
            eval_batch_size=self.eval_batch_size,
            reward_workers=self.reward_workers,
            max_new_tokens=self.max_new_tokens,
            eval_timeout=self.eval_timeout,
            global_freqs=self._global_freqs,
            rare_op_idx=self._rare_op_idx,
            max_iou_k=max_iou_k_now,
            max_iou_temperature=self.max_iou_temperature,
            max_iou_seed=self.max_iou_seed,
        )
        # HF Trainer.evaluate() calls self.log(metrics) BEFORE on_evaluate, so a
        # late metrics.update() never reaches wandb. Push directly to wandb instead.
        # Don't pass step=: wandb 0.16+ silently drops backwards steps, and Trainer
        # already advanced the internal cursor with eval_loss. Carry global_step
        # in the payload so dashboards can use it as the x-axis.
        if metrics is not None:
            metrics.update(iou_metrics)
        try:
            import wandb
            if wandb.run is not None and iou_metrics:
                payload = dict(iou_metrics)
                payload['eval/global_step'] = state.global_step
                wandb.log(payload)
        except Exception as e:
            print(f'[online-eval] wandb.log failed: {e}', flush=True)
