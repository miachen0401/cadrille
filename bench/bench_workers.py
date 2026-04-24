"""Benchmark reward worker throughput at different concurrency levels.

Generates a fixed set of CadQuery code strings from the SFT model (one-shot,
not reused), then times compute_rewards_parallel() with varying worker counts.
Reports throughput (completions/s) and wall time per 64-completion batch so you
can find the optimal reward_workers for your machine.

Also sweeps eval_workers for the eval pool (used during validation steps).

Usage
-----
uv run python tools/bench_workers.py \
    --pkl data/mined/combined_hard.pkl \
    --checkpoint checkpoints/cadrille-sft \
    --n-codes 64 \
    --worker-counts 8 12 16 20 24 \
    --timeout 10
"""

import argparse
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so HF_TOKEN / WANDB_API_KEY are available
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())


def _generate_codes(pkl_path: str, checkpoint: str, n: int, seed: int) -> tuple:
    """Generate n CadQuery code strings from the SFT model (fresh, not cached)."""
    import pickle
    import torch
    from transformers import AutoProcessor
    from cadrille import Cadrille
    from rl.dataset import render_img

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    valid = [d for d in data if os.path.exists(d.get('gt_mesh_path', ''))]
    rng = random.Random(seed)
    samples = rng.sample(valid, min(n, len(valid)))

    print(f'  Loading model from {checkpoint} ...')
    model = Cadrille.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')
    model.eval()

    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')

    from cadrille import collate
    codes, gt_paths = [], []
    print(f'  Generating {n} completions ...')
    for d in samples:
        mesh_path = d['gt_mesh_path']
        item = render_img(mesh_path)
        item['description'] = 'Generate cadquery code'
        item['gt_mesh_path'] = mesh_path

        batch = collate([{k: v for k, v in item.items() if k != 'gt_mesh_path'}],
                        processor=processor, n_points=256, eval=True)
        device = next(model.parameters()).device
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        with torch.no_grad():
            out = model.generate(
                **{k: v for k, v in batch.items()
                   if k in ('input_ids', 'attention_mask', 'pixel_values_videos',
                             'video_grid_thw', 'is_img', 'is_pc', 'point_clouds')},
                max_new_tokens=400,
                do_sample=True,
                temperature=0.3,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        prompt_len = batch['input_ids'].shape[1]
        completion = out[0, prompt_len:]
        code = processor.tokenizer.decode(completion, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        codes.append(code)
        gt_paths.append(mesh_path)

    del model
    torch.cuda.empty_cache()
    return codes, gt_paths


def _sweep_workers(codes: list, gt_paths: list, worker_counts: list,
                   timeout: float, n_repeats: int) -> list:
    """Time compute_rewards_parallel at each worker count."""
    from rl.reward import compute_rewards_parallel, init_reward_pool, shutdown_pools

    results = []
    n = len(codes)

    for n_workers in worker_counts:
        print(f'\n  [sweep] reward_workers={n_workers} ({n} completions × {n_repeats} runs)')
        shutdown_pools()
        init_reward_pool(n_workers)
        # Warm-up run (workers need to start + import cadquery)
        _ = compute_rewards_parallel(codes[:min(n_workers, n)], gt_paths[:min(n_workers, n)],
                                     workers=n_workers, timeout=timeout)

        times = []
        for r in range(n_repeats):
            # Shuffle codes each run so worker caches don't help
            idx = list(range(n))
            random.shuffle(idx)
            shuffled_codes = [codes[i] for i in idx]
            shuffled_paths = [gt_paths[i] for i in idx]

            t0 = time.perf_counter()
            rewards = compute_rewards_parallel(shuffled_codes, shuffled_paths,
                                               workers=n_workers, timeout=timeout)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            n_valid = sum(1 for r in rewards if r >= 0)
            print(f'    run {r+1}: {elapsed:.1f}s  valid={n_valid}/{n}  '
                  f'throughput={n/elapsed:.1f} completions/s')

        import numpy as np
        avg = float(np.mean(times))
        results.append({
            'reward_workers': n_workers,
            'wall_s_mean':    avg,
            'throughput':     n / avg,
        })

    shutdown_pools()
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pkl',           default='data/mined/combined_hard.pkl')
    parser.add_argument('--checkpoint',    default='checkpoints/cadrille-sft')
    parser.add_argument('--n-codes',       type=int, default=64,
                        help='Number of completions to evaluate (= B×G in training)')
    parser.add_argument('--worker-counts', type=int, nargs='+',
                        default=[8, 12, 16, 20, 24],
                        help='reward_workers values to sweep')
    parser.add_argument('--timeout',       type=float, default=10.0,
                        help='Per-completion CadQuery timeout (s)')
    parser.add_argument('--n-repeats',     type=int, default=2,
                        help='Timed runs per worker count (more = more stable estimate)')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip model generation; use dummy codes for timing only')
    args = parser.parse_args()

    print(f'nproc = {os.cpu_count()} cores')

    if args.skip_generate:
        # Use trivial CadQuery code for fast testing of pool overhead
        import pickle
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)
        valid = [d for d in data if os.path.exists(d.get('gt_mesh_path', ''))]
        samples = random.Random(args.seed).sample(valid, min(args.n_codes, len(valid)))
        codes = ['import cadquery as cq\nr = cq.Workplane("XY").box(1,1,1)'] * len(samples)
        gt_paths = [d['gt_mesh_path'] for d in samples]
    else:
        print(f'\nGenerating {args.n_codes} fresh completions from {args.checkpoint} ...')
        codes, gt_paths = _generate_codes(args.pkl, args.checkpoint,
                                          args.n_codes, args.seed)
        print(f'  Done. {len(codes)} codes generated.')

    print(f'\nSweeping reward_workers: {args.worker_counts}')
    results = _sweep_workers(codes, gt_paths, args.worker_counts,
                             args.timeout, args.n_repeats)

    print(f'\n{"="*60}')
    print(f'WORKER SWEEP SUMMARY  (n_codes={args.n_codes}  timeout={args.timeout}s)')
    print(f'{"="*60}')
    print(f'  {"workers":>8}  {"wall(s)":>8}  {"compl/s":>8}  {"speedup":>8}')
    print(f'  {"-"*38}')
    base = results[0]['wall_s_mean']
    for r in results:
        speedup = base / r['wall_s_mean']
        print(f'  {r["reward_workers"]:>8}  {r["wall_s_mean"]:>8.1f}  '
              f'  {r["throughput"]:>6.1f}  {speedup:>8.2f}x')

    import json
    os.makedirs('work_dirs/bench', exist_ok=True)
    with open('work_dirs/bench/worker_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved → work_dirs/bench/worker_sweep.json')


if __name__ == '__main__':
    main()
