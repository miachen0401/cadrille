"""Pass@k evaluation for Cadrille checkpoints.

Generates *n_samples* completions per example using temperature sampling,
scores each with the IoU reward, and reports pass@k using the unbiased
estimator from Chen et al. (2021):

    pass@k = 1 - prod_{i=0..k-1} (n - c - i) / (n - i)

where n = total samples drawn per example, c = number of "correct" samples
(IoU >= threshold).

Speed design
------------
Generation is batched *across examples* (eval-style), not per-example:

  for sample_pass in range(n_samples):              # outer: n passes
      for chunk in chunks(examples, batch_size):    # inner: ceil(N/B) calls
          model.generate(chunk)                     # B examples at once

This gives ceil(N/B) * n_samples generate() calls instead of N * 1.
Scoring is pipelined: each batch is submitted to a thread-pool immediately
so CadQuery subprocesses run while the GPU generates the next batch.

Two entry points:
  - `run_passk(model, processor, stl_paths, modality, out_dir, ...)` —
    used by the unified eval pipeline (eval/pipeline.py, eval/runner.py).
  - CLI via `python -m eval.passk --checkpoint ... --val-dir ...` —
    standalone sweep/post-training evaluation.

Usage
-----
# Single checkpoint
python -m eval.passk \\
    --checkpoint ./checkpoints/cadrille-rl/checkpoint-10000 \\
    --val-dir    ./data/deepcad_test_mesh

# Sweep all checkpoint-XXXXX dirs → learning curve
python -m eval.passk \\
    --checkpoint-sweep ./checkpoints/cadrille-rl \\
    --val-dir          ./data/deepcad_test_mesh \\
    --wandb-project cadrille-rl --wandb-run-id <id>

# RTX 4080 (16 GB) — use smaller batch + sequential if needed
python -m eval.passk \\
    --checkpoint ./checkpoints/cadrille-rl/checkpoint-10000 \\
    --val-dir    ./data/deepcad_test_mesh \\
    --eval-batch-size 4 --sequential
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from common.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Unbiased pass@k estimator (Chen et al. 2021 / HumanEval)
# ---------------------------------------------------------------------------

def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: n samples drawn, c correct, evaluate pass@k."""
    if n < k:
        return float('nan')
    if n == c or n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def pass_at_k_mean(n_list: List[int], c_list: List[int], k: int) -> float:
    vals = [_pass_at_k(n, c, k) for n, c in zip(n_list, c_list)]
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float('nan')


# ---------------------------------------------------------------------------
# Val example loading
# ---------------------------------------------------------------------------

def load_val_examples(val_dir: str, n_examples: int, n_points: int = 256) -> list:
    import trimesh
    from dataset import mesh_to_point_cloud

    stl_files = sorted(f for f in os.listdir(val_dir) if f.endswith('.stl'))
    rng = random.Random(42)
    rng.shuffle(stl_files)

    examples = []
    for fname in stl_files[:n_examples * 3]:
        if len(examples) >= n_examples:
            break
        gt_path = os.path.join(val_dir, fname)
        try:
            mesh = trimesh.load(gt_path)
            pc   = mesh_to_point_cloud(mesh, n_points)
            pc   = (pc - 0.5) * 2
            examples.append({
                'point_cloud':  pc,
                'description':  'Generate cadquery code',
                'file_name':    fname[:-4],
                'gt_mesh_path': gt_path,
            })
        except Exception:
            pass

    print(f'Loaded {len(examples)} val examples from {val_dir}')
    return examples


# ---------------------------------------------------------------------------
# Batched sampling generation
# ---------------------------------------------------------------------------

_GEN_KEYS = ('input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img',
             'pixel_values_videos', 'video_grid_thw')


@torch.no_grad()
def _generate_one_batch(model, chunk: list, processor,
                        max_new_tokens: int, temperature: float,
                        sequential: bool, device) -> List[str]:
    """Generate one code per example in *chunk* (temperature sampling).

    Returns list of decoded strings, one per example in chunk.
    Uses batched generate() for speed; falls back to sequential on OOM.
    """
    collate_items = [{k: v for k, v in ex.items() if k != 'gt_mesh_path'}
                     for ex in chunk]
    batch = collate(collate_items, processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]

    if temperature == 0:
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    else:
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True,
                          temperature=temperature, top_p=1.0, top_k=50)

    if not sequential:
        try:
            batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items() if k in _GEN_KEYS}
            out = model.generate(**batch_gpu, **gen_kwargs)
            return [processor.decode(out[j, prompt_len:], skip_special_tokens=True)
                    for j in range(len(chunk))]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('[passk] OOM on batched generate — switching to sequential')
            sequential = True

    # Sequential fallback
    codes = []
    for ex in chunk:
        item = {k: v for k, v in ex.items() if k != 'gt_mesh_path'}
        b = collate([item], processor=processor, n_points=256, eval=True)
        p = b['input_ids'].shape[1]
        single = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items() if k in _GEN_KEYS}
        out = model.generate(**single, **gen_kwargs)
        codes.append(processor.decode(out[0, p:], skip_special_tokens=True))
    return codes


# ---------------------------------------------------------------------------
# Core pass@k evaluation
# ---------------------------------------------------------------------------

def eval_passk(
    model,
    processor,
    examples: list,
    n_samples: int,
    k_values: List[int],
    threshold: float,
    max_new_tokens: int,
    eval_batch_size: int = 8,
    reward_workers: int = 8,
    temperature: float = 1.0,
    sequential: bool = False,
) -> dict:
    """Run pass@k evaluation.  Returns dict ready for JSON / W&B.

    Generation strategy: n_samples outer passes × ceil(N/B) batched generate()
    calls → best GPU utilisation.  Scoring is pipelined via thread-pool so
    CadQuery subprocesses run while the GPU works on the next batch.
    """
    model.eval()
    device = next(model.parameters()).device
    n = len(examples)

    all_codes: List[List[str]] = [[] for _ in range(n)]

    score_pool = ThreadPoolExecutor(max_workers=reward_workers)
    pending_futures = []

    print(f'Generating {n_samples} samples x {n} examples '
          f'(batch={eval_batch_size}, {"sequential" if sequential else "batched"})')

    for sample_pass in range(n_samples):
        for batch_start in range(0, n, eval_batch_size):
            chunk = examples[batch_start:batch_start + eval_batch_size]
            codes = _generate_one_batch(
                model, chunk, processor,
                max_new_tokens, temperature, sequential, device)

            for j, (ex, code) in enumerate(zip(chunk, codes)):
                ex_idx = batch_start + j
                all_codes[ex_idx].append(code)
                fut = score_pool.submit(
                    compute_metrics, code, ex['gt_mesh_path'], 30.0)
                pending_futures.append((fut, ex_idx, sample_pass))

        print(f'  pass {sample_pass+1}/{n_samples} generated', flush=True)

    print('Collecting scores ...', flush=True)
    per_example_ious: List[List[Optional[float]]] = [[] for _ in range(n)]
    for fut, ex_idx, _ in pending_futures:
        iou_reward, _cd = fut.result()
        iou = iou_reward if iou_reward > -1.0 else None
        per_example_ious[ex_idx].append(iou)

    score_pool.shutdown(wait=False)

    per_example = []
    for i, ex in enumerate(examples):
        ious = per_example_ious[i]
        valid_ious = [v for v in ious if v is not None]
        c = sum(1 for v in valid_ious if v >= threshold)
        per_example.append({
            'file_name': ex.get('file_name', str(i)),
            'n': len(ious),
            'c': c,
            'ious': [v if v is not None else -1.0 for v in ious],
        })

    n_list = [e['n'] for e in per_example]
    c_list = [e['c'] for e in per_example]

    results = {
        'threshold':   threshold,
        'n_samples':   n_samples,
        'n_examples':  n,
        'per_example': per_example,
        'pass_at_k':   {},
    }
    wandb_metrics = {}

    print(f'\n--- pass@k (IoU >= {threshold}) ---')
    for k in k_values:
        mean_val = pass_at_k_mean(n_list, c_list, k)
        results['pass_at_k'][k] = mean_val
        print(f'  pass@{k}: {mean_val:.3f}')
        wandb_metrics[f'eval/pass@{k}'] = mean_val

    # best_iou@k: mean over examples of max(IoU) among k samples (oracle upper bound)
    results['best_iou_at_k'] = {}
    print(f'\n--- best IoU@k (oracle, mean of per-example max) ---')
    for k in k_values:
        per_ex_best = []
        for e in per_example:
            valid = [v for v in e['ious'][:k] if v >= 0]
            if valid:
                per_ex_best.append(max(valid))
        if per_ex_best:
            bk = float(np.mean(per_ex_best))
            results['best_iou_at_k'][k] = bk
            print(f'  best_iou@{k}: {bk:.3f}')
            wandb_metrics[f'eval/best_iou@{k}'] = bk

    all_valid = [v for e in per_example for v in e['ious'] if v >= 0]
    if all_valid:
        results['mean_iou']   = float(np.mean(all_valid))
        results['valid_frac'] = len(all_valid) / (n * n_samples)
        print(f'  mean IoU (all valid): {results["mean_iou"]:.3f}')
        print(f'  valid fraction:       {results["valid_frac"]:.1%}')
        wandb_metrics['eval/passk_mean_iou']   = results['mean_iou']
        wandb_metrics['eval/passk_valid_frac'] = results['valid_frac']

    results['wandb_metrics'] = wandb_metrics
    return results


# ---------------------------------------------------------------------------
# Pipeline wrapper — used by eval/pipeline.py, eval/runner.py
# ---------------------------------------------------------------------------

def run_passk(
    model,
    processor,
    stl_paths: List[str],
    modality: str,
    out_dir: Path,
    n_samples: int = 8,
    k_values=(1, 5),
    iou_threshold: float = 0.95,
    temperature: float = 0.8,
    batch_size: int = 16,
    max_new_tokens: int = 768,
    score_workers: int = 8,
) -> dict:
    """Run pass@k evaluation for one (model, dataset, modality) combo.

    Builds examples from stl_paths (pc or img), delegates to eval_passk,
    and persists results to out_dir/passk.json.
    """
    examples = _build_examples(stl_paths, modality)
    if not examples:
        print(f'  [pass@k] No examples for {modality} — skipping')
        return {}

    print(f'  [pass@k] {modality} n={len(examples)} n_samples={n_samples} '
          f'k={list(k_values)} threshold={iou_threshold}', flush=True)

    results = eval_passk(
        model=model,
        processor=processor,
        examples=examples,
        n_samples=n_samples,
        k_values=list(k_values),
        threshold=iou_threshold,
        max_new_tokens=max_new_tokens,
        eval_batch_size=batch_size,
        reward_workers=score_workers,
        temperature=temperature,
        sequential=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'passk.json'
    save_data = {k: v for k, v in results.items() if k != 'wandb_metrics'}
    out_file.write_text(json.dumps(save_data, indent=2))
    print(f'  [pass@k] Saved -> {out_file}')
    return results


def _build_examples(stl_paths: List[str], modality: str) -> list:
    """Build example dicts for eval_passk (pc or img mode)."""
    import trimesh
    from dataset import mesh_to_point_cloud
    from common.meshio import render_img

    examples = []
    for gt_path in stl_paths:
        stem = Path(gt_path).stem
        try:
            if modality == 'pc':
                mesh = trimesh.load(gt_path)
                pc = mesh_to_point_cloud(mesh, 256)
                pc = (pc - 0.5) * 2
                examples.append({
                    'point_cloud': pc,
                    'description': 'Generate cadquery code',
                    'file_name': stem,
                    'gt_mesh_path': gt_path,
                    '_modality': 'pc',
                })
            else:  # img
                render_result = render_img(gt_path)
                examples.append({
                    'video': render_result['video'],
                    'description': 'Generate cadquery code',
                    'file_name': stem,
                    'gt_mesh_path': gt_path,
                    '_modality': 'img',
                })
        except Exception:
            continue

    return examples


# ---------------------------------------------------------------------------
# Checkpoint sweep helper
# ---------------------------------------------------------------------------

def _find_checkpoints(sweep_dir: str):
    checkpoints = []
    for name in sorted(os.listdir(sweep_dir)):
        path = os.path.join(sweep_dir, name)
        if not os.path.isdir(path):
            continue
        for prefix in ('checkpoint-', 'step-'):
            if name.startswith(prefix):
                try:
                    step = int(name[len(prefix):])
                    checkpoints.append((step, path))
                except ValueError:
                    pass
    return sorted(checkpoints)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Post-training pass@k evaluation for Cadrille checkpoints.')

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument('--checkpoint',       type=str)
    ckpt_group.add_argument('--checkpoint-sweep', type=str)

    parser.add_argument('--val-dir',        type=str, required=True)
    parser.add_argument('--base-model',     type=str,
                        default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--n-examples',     type=int, default=50)
    parser.add_argument('--n-samples',      type=int, default=5,
                        help='Samples per example; 5 is sufficient for pass@1 and pass@5')
    parser.add_argument('--k-values',       type=str, default='1,5')
    parser.add_argument('--threshold',      type=float, default=0.5)
    parser.add_argument('--temperature',    type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int,   default=1000)
    parser.add_argument('--eval-batch-size',type=int,   default=8,
                        help='Examples per generate() call (default 8; use 4 on 4080)')
    parser.add_argument('--sequential',     action='store_true',
                        help='Force sequential generation within each batch')
    parser.add_argument('--reward-workers', type=int,   default=8)
    parser.add_argument('--output-dir',     type=str,
                        default='./eval_outputs/passk')
    parser.add_argument('--wandb-project',  type=str,   default=None)
    parser.add_argument('--wandb-run-id',   type=str,   default=None)
    parser.add_argument('--wandb-entity',   type=str,   default=None)
    parser.add_argument('--wandb-offline',  action='store_true')

    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    use_wandb = False
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                id=args.wandb_run_id or None,
                resume='allow',
                mode='offline' if args.wandb_offline else 'online',
                config=vars(args),
            )
            use_wandb = True
        except Exception as e:
            print(f'Warning: W&B init failed ({e})')

    _local_ckpt = args.checkpoint or args.checkpoint_sweep
    _proc_src = (args.base_model
                 if (args.base_model and os.path.isdir(args.base_model))
                 else _local_ckpt)
    if _proc_src != args.base_model:
        print(f'Processor: {args.base_model!r} not local -> loading from checkpoint')
    processor = AutoProcessor.from_pretrained(
        _proc_src,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    examples = load_val_examples(args.val_dir, args.n_examples)
    if not examples:
        print('ERROR: No examples loaded. Check --val-dir.')
        sys.exit(1)

    if args.checkpoint:
        ckpt_list = [(None, args.checkpoint)]
    else:
        ckpt_list = _find_checkpoints(args.checkpoint_sweep)
        if not ckpt_list:
            print(f'No checkpoint-XXXXX dirs found in {args.checkpoint_sweep}')
            sys.exit(1)
        print(f'Found {len(ckpt_list)} checkpoints:')
        for step, path in ckpt_list:
            print(f'  step={step:>8}  {path}')

    all_results = []

    for ckpt_step, ckpt_path in ckpt_list:
        label = f'step={ckpt_step}' if ckpt_step is not None else os.path.basename(ckpt_path)
        print(f'\n{"="*60}\nEvaluating {ckpt_path}  [{label}]\n{"="*60}')

        model = Cadrille.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map='auto')

        results = eval_passk(
            model, processor, examples,
            n_samples=args.n_samples,
            k_values=k_values,
            threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            reward_workers=args.reward_workers,
            temperature=args.temperature,
            sequential=args.sequential,
        )
        results['checkpoint'] = ckpt_path
        results['step']       = ckpt_step
        all_results.append(results)

        safe = os.path.basename(ckpt_path).replace('/', '_')
        out_path = os.path.join(args.output_dir, f'passk_{safe}.json')
        with open(out_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'wandb_metrics'},
                      f, indent=2)
        print(f'Saved -> {out_path}')

        if use_wandb and results.get('wandb_metrics'):
            import wandb
            wandb.log(results['wandb_metrics'],
                      step=ckpt_step if ckpt_step is not None else 0)

        del model
        torch.cuda.empty_cache()

    if len(all_results) > 1:
        summary_path = os.path.join(args.output_dir, 'passk_sweep_summary.json')
        summary = [{'checkpoint': r['checkpoint'], 'step': r['step'],
                    'pass_at_k': {str(k): v for k, v in r['pass_at_k'].items()},
                    'mean_iou':  r.get('mean_iou')}
                   for r in all_results]
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSweep summary -> {summary_path}')
        print(f'\n{"Step":>8}', end='')
        for k in k_values:
            print(f'  pass@{k}', end='')
        print('  mean_IoU')
        for row in summary:
            step_str = str(row['step']) if row['step'] is not None else 'final'
            print(f'{step_str:>8}', end='')
            for k in k_values:
                v = row['pass_at_k'].get(str(k))
                print(f'  {v:.3f}' if v is not None else '      -', end='')
            iou = row.get('mean_iou')
            print(f'  {iou:.3f}' if iou is not None else '       -')

    if use_wandb:
        import wandb
        wandb.finish()

    print('\nDone.')


if __name__ == '__main__':
    main()
