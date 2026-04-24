"""Diagnose entropy-blow-up after checkpoint resume.

Measures entropy at three independent points so you can tell WHERE the
problem originates:

  1. entropy_after_load         — model.forward on a fixed greedy completion,
                                  no rollout sampling, no optimizer touched.
                                  Sanity-checks that the checkpoint loaded correctly.

  2. entropy_before_backward    — model.forward on stochastic rollout completions
                                  (temperature > 0, same rollout used for training),
                                  BEFORE optimizer.step().  Equivalent to
                                  train/entropy_k0 in the main loop.

  3. entropy_after_optimizer_step — AFTER one full optimizer.step() on those
                                  rollouts.  If entropy explodes here but stage 2
                                  was fine, the bug is in optimizer state.

Usage
-----
# Use the local 4080 config with the A100 debug checkpoint:
uv run python tools/diag_resume_entropy.py \
    --config   configs/rl/4080.yaml \
    --ckpt     checkpoints/debug-3step-a100-6000/checkpoint-final \
    --data-idx 0
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from train.rl.config import load_yaml, resolve_args
from common.meshio import MeshDataset
from train.rl.dataset import RLDataset
from common.metrics import init_reward_pool, compute_rewards_parallel
from train.rl.algorithms.cppo import (
    generate_rollouts,
    model_forward,
    compute_token_log_probs,
    compute_policy_entropy,
    create_completion_mask,
    cppo_loss_fn,
    expand_batch,
    slice_batch,
    _GEN_INPUT_KEYS,
)


# ---------------------------------------------------------------------------
# Stage 1 helper: entropy from a single greedy forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def entropy_after_load(model, dataset, processor, args, idx: int = 0) -> float:
    """Greedy generation → model_forward → entropy.

    No temperature, no sampling — gives a deterministic entropy baseline
    that tells you whether the checkpoint itself is healthy.
    """
    device = next(model.parameters()).device
    gen_model = model.module if hasattr(model, 'module') else model

    item = {k: v for k, v in dataset[idx].items() if not k.startswith('_')}
    batch = collate([item], processor=processor, n_points=256, eval=True)

    # Disable GC for generation (same as generate_rollouts)
    had_gc = getattr(gen_model, 'is_gradient_checkpointing', False)
    if had_gc:
        gen_model.gradient_checkpointing_disable()
    model.eval()

    if hasattr(gen_model, 'rope_deltas'):
        gen_model.rope_deltas = None

    gen_input = {k: batch[k].to(device) if isinstance(batch.get(k), torch.Tensor)
                 else batch.get(k)
                 for k in _GEN_INPUT_KEYS}

    greedy_ids = gen_model.generate(
        **gen_input,
        max_new_tokens=min(64, args.max_new_tokens),   # short for speed
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
    )

    if had_gc:
        gen_model.gradient_checkpointing_enable()
    model.train()

    prompt_len = batch['input_ids'].shape[1]
    full_ids    = greedy_ids.cpu()                                # [1, full_len]
    comp_ids    = full_ids[:, prompt_len:]                        # [1, T]
    T           = comp_ids.shape[1]
    if T == 0:
        print('  [WARN] greedy generation produced 0 tokens — returning NaN')
        return float('nan')

    eos_id     = processor.tokenizer.eos_token_id
    comp_mask  = create_completion_mask(comp_ids, eos_id)         # [1, T]
    prompt_mask = batch['attention_mask']                          # [1, prompt_len]
    full_attn  = torch.cat([prompt_mask, comp_mask.long()], dim=1)

    g_batch = {k: batch[k] for k in _GEN_INPUT_KEYS if k in batch}

    out    = model_forward(model, full_ids.to(device), full_attn.to(device),
                           {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                            for k, v in g_batch.items()}, device)
    lp     = compute_token_log_probs(out.logits, full_ids.to(device), T)  # [1, T]
    H      = compute_policy_entropy(lp, comp_mask.to(device)).item()
    return H


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',   required=True,  help='Path to YAML config')
    parser.add_argument('--ckpt',     required=True,  help='Checkpoint dir to diagnose')
    parser.add_argument('--data-idx', type=int, default=0,
                        help='Dataset index used for all three stages (default: 0)')
    cli = parser.parse_args()

    # ── resolve args (mirrors train.py) ─────────────────────────────────────
    cfg  = load_yaml(cli.config)
    # Namespace with the bare minimum fields resolve_args() needs
    class _Args:
        config              = cli.config
        run_name            = 'diag-entropy'
        checkpoint_path     = cli.ckpt
        max_steps           = 1
        mode                = None
        wandb_offline       = True
        sequential_generation = None
    args_obj = _Args()
    resolve_args(args_obj, cfg)
    args_obj.checkpoint_path = cli.ckpt           # always use CLI ckpt
    args_obj.max_steps       = 1
    args_obj.wandb_project   = None               # no W&B
    args_obj.batch_size      = 1
    args_obj.G               = max(2, getattr(args_obj, 'G', 2))
    args_obj.top_N           = min(getattr(args_obj, 'top_N', 1),
                                   args_obj.G)
    args_obj.batch_updates   = 1                  # only one inner step needed
    args_obj.reward_workers  = getattr(args_obj, 'reward_workers', 4)
    args_obj.sequential_generation = True         # safest for local device

    print(f'\n{"="*60}')
    print(f'[diag] checkpoint : {cli.ckpt}')
    print(f'[diag] config     : {cli.config}')
    print(f'[diag] data_idx   : {cli.data_idx}')
    print(f'{"="*60}\n')

    # ── processor ────────────────────────────────────────────────────────────
    _proc_kwargs = dict(min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')
    processor = AutoProcessor.from_pretrained(cli.ckpt, **_proc_kwargs)

    # ── model ────────────────────────────────────────────────────────────────
    model = Cadrille.from_pretrained(
        cli.ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto')
    model.gradient_checkpointing_enable()

    # ── freeze vision encoder if config says so ──────────────────────────────
    if getattr(args_obj, 'freeze_vision_encoder', False):
        raw = model.module if hasattr(model, 'module') else model
        if hasattr(raw, 'visual'):
            for p in raw.visual.parameters():
                p.requires_grad_(False)
            print('[diag] vision encoder frozen (matches config)')

    # ── optimizer (fresh, same as train.py) ─────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args_obj.lr,
                                  weight_decay=0.01, foreach=False)
    # pre-warm (identical to train.py)
    for p in trainable:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print('[diag] AdamW pre-warmed (fresh state, NO optimizer.pt loaded)\n')

    # ── dataset ──────────────────────────────────────────────────────────────
    modality = getattr(args_obj, 'train_modality', 'img')
    if args_obj.data_dir and os.path.isdir(args_obj.data_dir):
        dataset = MeshDataset(args_obj.data_dir, noise_scale=0.01, modality=modality)
    elif getattr(args_obj, 'hard_examples_pkl', None):
        dataset = RLDataset(args_obj.hard_examples_pkl, modality=modality)
    else:
        raise ValueError('No data_dir or hard_examples_pkl in config')

    idx = min(cli.data_idx, len(dataset) - 1)
    print(f'[diag] dataset size={len(dataset)}, using index {idx}\n')

    # ── reward pool (needed for stage 2/3) ──────────────────────────────────
    init_reward_pool(n_workers=args_obj.reward_workers)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1  entropy_after_load
    # Greedy forward pass — no sampling, no training.  Reflects the checkpoint.
    # ════════════════════════════════════════════════════════════════════════
    print('── STAGE 1: entropy_after_load ─────────────────────────────────')
    print('   greedy generation → model_forward → entropy')
    print('   (no optimizer touched, no stochastic sampling)')
    H1 = entropy_after_load(model, dataset, processor, args_obj, idx=idx)
    print(f'\n   entropy_after_load = {H1:.4f}\n')

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 + 3  — run one cppo_step manually, capturing entropy at each gate
    # ════════════════════════════════════════════════════════════════════════
    device     = next(model.parameters()).device
    eos_id     = processor.tokenizer.eos_token_id
    pad_id     = processor.tokenizer.pad_token_id
    G          = args_obj.G
    N          = args_obj.top_N

    item       = {k: v for k, v in dataset[idx].items() if not k.startswith('_')}
    item['_dataset_idx'] = idx
    collate_item = {k: v for k, v in item.items() if not k.startswith('_')}
    batch      = collate([collate_item], processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]

    # ── rollout ──────────────────────────────────────────────────────────────
    print('── STAGE 2: entropy_before_backward ────────────────────────────')
    print(f'   generating G={G} rollouts ...')
    generated_ids = generate_rollouts(
        model, {k: batch.get(k) for k in _GEN_INPUT_KEYS},
        G, args_obj, pad_id, processor)                             # [G, full_len]

    code_strings = [
        processor.decode(generated_ids[i, prompt_len:],
                         skip_special_tokens=True,
                         clean_up_tokenization_spaces=False)
        for i in range(G)
    ]
    gt_path  = item['gt_mesh_path']
    rewards  = compute_rewards_parallel(code_strings, [gt_path] * G,
                                        workers=args_obj.reward_workers)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)  # [1, G]
    mean_r    = rewards_t.mean(dim=1, keepdim=True)
    adv_raw   = rewards_t - mean_r                                         # [1, G]
    adv_raw   = torch.nan_to_num(adv_raw, nan=0.0)

    if adv_raw.abs().max().item() < 1e-6:
        print('   [WARN] all rewards identical → advantages≈0, skipping stages 2/3')
        print(f'   rewards: {rewards}')
        return

    print(f'   rewards: {[f"{r:.3f}" for r in rewards]}')

    # top-N selection
    _, top_idx = torch.topk(adv_raw.abs(), N, dim=1)                # [1, N]
    flat_idx   = (torch.arange(1).unsqueeze(1) * G + top_idx).reshape(-1)  # [N]

    sel_ids_cpu  = generated_ids[flat_idx]                          # [N, full_len]
    comp_ids_cpu = sel_ids_cpu[:, prompt_len:]                      # [N, T]
    T            = comp_ids_cpu.shape[1]
    comp_mask_cpu = create_completion_mask(comp_ids_cpu, eos_id)    # [N, T]

    g_batch_cpu   = expand_batch(batch, G)                          # [G, ...]
    sel_g_batch_cpu = slice_batch(g_batch_cpu, flat_idx)
    prompt_mask   = sel_g_batch_cpu['attention_mask']
    full_attn_cpu = torch.cat([prompt_mask, comp_mask_cpu.long()], dim=1)
    adv_sel       = adv_raw.reshape(-1)[flat_idx].unsqueeze(1)      # [N, 1]

    sel_ids   = sel_ids_cpu.to(device)
    full_attn = full_attn_cpu.to(device)
    comp_mask = comp_mask_cpu.to(device)
    advantages = adv_sel.to(device)
    sel_g_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in sel_g_batch_cpu.items()}

    # old_lp = entropy_before_backward
    with torch.no_grad():
        old_out = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
        old_lp  = compute_token_log_probs(old_out.logits, sel_ids, T).detach()

    H2 = compute_policy_entropy(old_lp, comp_mask).item()
    print(f'\n   entropy_before_backward = {H2:.4f}\n')

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3  entropy_after_optimizer_step
    # ════════════════════════════════════════════════════════════════════════
    print('── STAGE 3: entropy_after_optimizer_step ───────────────────────')
    entropy_coef = float(getattr(args_obj, 'entropy_coef', 0.0))

    model.train()
    new_out = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
    new_lp  = compute_token_log_probs(new_out.logits, sel_ids, T)

    loss = cppo_loss_fn(new_lp, old_lp, advantages, comp_mask,
                        args_obj.eps_high, args_obj.eps_low)
    if entropy_coef > 0:
        step_entropy = compute_policy_entropy(new_lp, comp_mask)
        loss = loss - entropy_coef * step_entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()

    with torch.no_grad():
        new_out2 = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
        new_lp2  = compute_token_log_probs(new_out2.logits, sel_ids, T).detach()
    H3 = compute_policy_entropy(new_lp2, comp_mask).item()
    print(f'\n   entropy_after_optimizer_step = {H3:.4f}\n')

    # ════════════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('SUMMARY')
    print(f'  entropy_after_load           = {H1:.4f}')
    print(f'  entropy_before_backward      = {H2:.4f}')
    print(f'  entropy_after_optimizer_step = {H3:.4f}')
    print()
    if H1 > 3.0:
        print('  ► STAGE 1 HIGH → checkpoint did not load correctly '
              '(wrong path, corrupt weights, or missing keys)')
    elif H2 > H1 + 1.0:
        print('  ► STAGE 1→2 jump → rollout / rope_deltas / model_forward '
              'behaves differently from greedy path')
    elif H3 > H2 + 1.0:
        print('  ► STAGE 2→3 jump → optimizer update blows entropy '
              '(missing exp_avg_sq: effective LR too large)')
    else:
        print('  ► All stages stable — entropy looks healthy at this checkpoint')
    print('=' * 60)


if __name__ == '__main__':
    main()
