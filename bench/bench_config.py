"""Benchmark RL training throughput for one or more config files.

Runs N warm-up steps followed by N timed steps using the same model checkpoint
and the same data examples for every config, so results are directly comparable.

Metrics reported per config:
  step_time   — wall-clock seconds per training step (rollout + reward + PPO)
  gen_s       — rollout generation time
  rew_s       — reward subprocess time
  grad_s      — PPO forward + backward time
  steps/hr    — projected throughput
  GPU_peak    — peak GPU memory allocated (GB)
  avg_gen_len — average generated token length

Usage
-----
uv run python tools/bench_config.py \\
    --configs configs/rl/h100.yaml configs/rl/h100_bs8.yaml \\
    --n-warmup 2 \\
    --n-bench  3 \\
    --checkpoint checkpoints/cadrille-sft \\
    --pkl data/mined/combined_hard.pkl \\
    --n-examples 8 \\
    --seed 42

Add --gc-frag to enable PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
(same setting used in rl/train.py) to test OOM resilience on large batches.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_examples(pkl_path: str, n: int, modality: str, seed: int = 42):
    """Load n random examples from a pkl for representative benchmarking.

    Uses random sampling (not smallest files) so avg_gen_len matches real training.
    """
    import pickle
    import random
    from rl.dataset import render_img

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    valid = [d for d in data if os.path.exists(d.get('gt_mesh_path', d.get('mesh_path', '')))]
    rng = random.Random(seed)
    examples = rng.sample(valid, min(n, len(valid)))

    out = []
    for d in examples:
        mesh_path = d.get('gt_mesh_path', d.get('mesh_path', ''))
        if modality == 'img':
            item = render_img(mesh_path)
        else:
            import trimesh
            from common.datasets import mesh_to_point_cloud
            mesh = trimesh.load(mesh_path)
            mesh.apply_translation(-mesh.bounds[0])
            mesh.apply_scale(1.0 / mesh.extents.max())
            import numpy as np
            pc = mesh_to_point_cloud(mesh, 256)
            pc = (pc - 0.5) * 2
            item = {'point_cloud': pc}
        item['gt_mesh_path'] = mesh_path
        item['file_name'] = d.get('file_name', os.path.basename(mesh_path)[:-4])
        item['description'] = 'Generate cadquery code'
        item['_modality'] = modality
        out.append(item)
    return out


def _bench_one(cfg_path: str, examples: list, n_warmup: int, n_bench: int,
               checkpoint: str | None) -> dict:
    """Set up model+optimizer for one config, run steps, return timing stats."""
    import numpy as np
    import torch
    from transformers import AutoProcessor

    from cadrille import Cadrille, collate
    from rl.config import load_yaml, resolve_args
    from rl.algorithms.cppo import cppo_step
    from rl.reward import init_reward_pool, shutdown_pools

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_yaml(cfg_path)
    import argparse as _ap
    _dummy = _ap.Namespace(
        config=cfg_path, run_name=None, checkpoint_path=cfg.get('checkpoint_path'),
        max_steps=None, wandb_offline=True, mode=None, sequential_generation=None,
    )
    resolve_args(_dummy, cfg)
    args = _dummy
    if checkpoint:
        args.checkpoint_path = checkpoint
    args.wandb_project = None          # disable W&B during bench
    args.sequential_generation = getattr(args, 'sequential_generation', False)

    label = os.path.splitext(os.path.basename(cfg_path))[0]
    print(f'\n{"="*60}')
    print(f'[bench] {label}  ({cfg_path})')
    print(f'  G={args.G}  top_N={args.top_N}  batch_size={getattr(args,"batch_size",1)}'
          f'  max_new_tokens={args.max_new_tokens}'
          f'  batch_updates={args.batch_updates}'
          f'  sequential_gen={args.sequential_generation}')
    print(f'{"="*60}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Processor ────────────────────────────────────────────────────────────
    base_model = getattr(args, 'base_model', 'Qwen/Qwen2-VL-2B-Instruct')
    processor = AutoProcessor.from_pretrained(
        base_model, min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')

    # ── Model ────────────────────────────────────────────────────────────────
    model = Cadrille.from_pretrained(
        args.checkpoint_path, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')
    model.gradient_checkpointing_enable()
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=0.01, foreach=False)
    # Pre-warm optimizer states
    with torch.no_grad():
        for p in trainable_params:
            state = optimizer.state[p]
            state['step']       = torch.tensor(0.0)
            state['exp_avg']    = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    # ── Reward pool — use full config worker count for realistic timing ───────
    n_workers = getattr(args, 'reward_workers', 4)
    init_reward_pool(n_workers)

    # ── Collate examples into items ──────────────────────────────────────────
    batch_size = max(1, int(getattr(args, 'batch_size', 1)))
    # Tile examples to fill batch_size (repeat if fewer examples than batch_size)
    tiled = (examples * ((batch_size // len(examples)) + 1))[:batch_size]
    items = [dict(ex) for ex in tiled]

    # ── Benchmark loop ───────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    gen_list, rew_list, grad_list, total_list, glen_list = [], [], [], [], []

    n_total = n_warmup + n_bench
    for s in range(n_total):
        phase = 'warmup' if s < n_warmup else 'bench'
        t0 = time.perf_counter()
        try:
            result = cppo_step(
                model=model,
                optimizer=optimizer,
                items=items,
                processor=processor,
                args=args,
                step=s,
                compute_diag=False,
                debug_rollouts=False,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f'  OOM at step {s}: {e}')
            result = None

        wall = time.perf_counter() - t0
        if result is None:
            print(f'  [{phase} step {s}] OOM')
            continue

        if phase == 'bench':
            gen_list.append(result.get('train/gen_seconds', 0))
            rew_list.append(result.get('train/rew_seconds', 0))
            grad_list.append(result.get('train/grad_seconds', 0))
            total_list.append(wall)
            glen_list.append(result.get('train/avg_gen_len', 0))
            print(f'  [bench step {s-n_warmup+1}/{n_bench}] '
                  f'total={wall:.1f}s  gen={gen_list[-1]:.1f}s  '
                  f'rew={rew_list[-1]:.1f}s  grad={grad_list[-1]:.1f}s  '
                  f'avg_len={glen_list[-1]:.0f}  '
                  f'reward={result["train/mean_reward"]:.3f}')
        else:
            print(f'  [warmup  step {s+1}/{n_warmup}] '
                  f'total={wall:.1f}s  reward={result["train/mean_reward"]:.3f}')

    # Cleanup
    shutdown_pools()
    del model
    torch.cuda.empty_cache()

    if not total_list:
        return {'label': label, 'error': 'all steps OOM'}

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        'label':       label,
        'step_time':   float(np.mean(total_list)),
        'gen_s':       float(np.mean(gen_list)),
        'rew_s':       float(np.mean(rew_list)),
        'grad_s':      float(np.mean(grad_list)),
        'steps_hr':    3600.0 / float(np.mean(total_list)),
        'gpu_peak_gb': peak_gb,
        'avg_gen_len': float(np.mean(glen_list)),
        'n_bench':     len(total_list),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='+', required=True,
                        help='Config yaml files to benchmark')
    parser.add_argument('--n-warmup',   type=int, default=2,
                        help='Warm-up steps (not counted in timing)')
    parser.add_argument('--n-bench',    type=int, default=5,
                        help='Timed benchmark steps')
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint_path for all configs')
    parser.add_argument('--pkl',        default='data/mined/combined_hard.pkl',
                        help='Examples pkl for benchmarking')
    parser.add_argument('--n-examples', type=int, default=8,
                        help='Number of examples (randomly sampled for representative gen length)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed for example sampling')
    parser.add_argument('--gc-frag',    action='store_true',
                        help='Enable PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 '
                             '(same as rl/train.py) to reduce OOM on large batches')
    parser.add_argument('--out',        default='work_dirs/bench/results.json',
                        help='Output JSON path')
    args = parser.parse_args()

    if args.gc_frag:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')
        print('  [gc-frag] PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8')

    modality = 'img'   # always benchmark in img mode (training mode)
    print(f'Loading {args.n_examples} random examples from {args.pkl} (modality={modality}, seed={args.seed}) ...')
    examples = _load_examples(args.pkl, args.n_examples, modality, seed=args.seed)
    print(f'  {len(examples)} examples loaded')

    results = []
    for cfg in args.configs:
        r = _bench_one(cfg, examples, args.n_warmup, args.n_bench, args.checkpoint)
        results.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'BENCHMARK SUMMARY  (n_warmup={args.n_warmup}  n_bench={args.n_bench}'
          f'  examples={args.n_examples})')
    print(f'{"="*80}')
    hdr = f'  {"Config":<18}  {"step(s)":>7}  {"gen(s)":>6}  {"rew(s)":>6}'
    hdr += f'  {"grad(s)":>7}  {"steps/hr":>8}  {"GPU_pk(GB)":>10}  {"avg_len":>7}'
    print(hdr)
    print(f'  {"-"*76}')
    for r in results:
        if 'error' in r:
            print(f'  {r["label"]:<18}  {"OOM":>7}')
            continue
        print(f'  {r["label"]:<18}'
              f'  {r["step_time"]:>7.1f}'
              f'  {r["gen_s"]:>6.1f}'
              f'  {r["rew_s"]:>6.1f}'
              f'  {r["grad_s"]:>7.1f}'
              f'  {r["steps_hr"]:>8.1f}'
              f'  {r["gpu_peak_gb"]:>10.2f}'
              f'  {r["avg_gen_len"]:>7.0f}')

    import json
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
