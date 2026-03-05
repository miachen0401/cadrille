"""Dr. CPPO / GRPO algorithm — aligned with official rl branch grpo_mm.py.

Key algorithmic choices matching the reference:
  1. Per-TOKEN log-prob ratios (not per-sequence) — clipping is meaningful
  2. Advantages = (reward − group_mean), NO std normalisation
  3. Old log-probs computed from the *current* model at rollout time (no stale
     old_model); reused across batch_updates inner iterations (standard PPO)
  4. Gradient clipping max_norm=0.1  (ref uses 0.1, we had 1.0 — 10× too loose)
  5. nan_to_num + clamp([-10, 10]) on the per-sequence loss before .mean()
  6. bad_words_ids blocks the video-token during generation
"""

import os
import time
import shutil
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cadrille import Cadrille, collate
from rl.reward import compute_rewards_parallel
from rl.eval import run_validation, log_eval

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Batch manipulation helpers
# ---------------------------------------------------------------------------

def expand_batch(batch: dict, G: int) -> dict:
    """Replicate every tensor/list in batch G times along the batch axis."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.repeat(G, *([1] * (v.dim() - 1)))
        elif isinstance(v, list):
            out[k] = v * G
        else:
            out[k] = v
    return out


def slice_batch(batch: dict, indices: torch.Tensor) -> dict:
    """Index into a batch dict along the first axis using *indices*."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[indices]
        elif isinstance(v, list):
            out[k] = [v[i] for i in indices.tolist()]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Per-token log-prob helpers  (matches grpo_mm.py exactly)
# ---------------------------------------------------------------------------

def create_completion_mask(completion_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Float mask: 1 for tokens up to and including the first EOS, 0 after.

    Identical to grpo_mm.create_completion_mask.
    """
    is_eos  = (completion_ids == eos_id)
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1),
                         dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).unsqueeze(0)
    return (seq_idx <= eos_idx.unsqueeze(1)).float()   # [B, T]


def compute_token_log_probs(logits: torch.Tensor, full_ids: torch.Tensor,
                             logits_to_keep: int) -> torch.Tensor:
    """Per-token log probs for the completion portion of each sequence.

    Identical to grpo_mm.selective_log_softmax + compute_log_probs.

    Args:
        logits:         [B, L, V] — full model output (prompt + completion)
        full_ids:       [B, L]   — full input_ids (prompt + completion)
        logits_to_keep: T        — number of completion tokens

    Returns:
        [B, T] per-token log probs
    """
    # Causal-LM shift: logits[:, i] predicts token[:, i+1]
    shift_logits = logits[:, :-1, :].float()             # [B, L-1, V]
    shift_logits = shift_logits[:, -logits_to_keep:, :]  # [B, T, V]
    completion_ids = full_ids[:, -logits_to_keep:]        # [B, T]
    log_probs = F.log_softmax(shift_logits, dim=-1)       # [B, T, V]
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]


def cppo_loss_fn(new_lp: torch.Tensor, old_lp: torch.Tensor,
                 advantages: torch.Tensor, completion_mask: torch.Tensor,
                 eps_high: float, eps_low: float) -> torch.Tensor:
    """Clipped PPO surrogate loss.  Identical to grpo_mm.grpo_loss.

    Args:
        new_lp, old_lp:  [B, T] per-token log probs
        advantages:       [B, 1] one scalar per sequence (broadcasts over T)
        completion_mask:  [B, T] float — 1 for valid tokens, 0 after EOS/pad
        eps_high, eps_low: clip bounds

    Returns:
        scalar loss (minimised during training)
    """
    ratio   = torch.exp(new_lp - old_lp)                          # [B, T]
    surr1   = ratio * advantages
    surr2   = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages
    per_tok = torch.min(surr1, surr2)                              # [B, T]

    n_tok    = completion_mask.sum(dim=1).clamp(min=1)             # [B]
    seq_loss = (per_tok * completion_mask).sum(dim=1) / n_tok      # [B]
    seq_loss = torch.nan_to_num(seq_loss, nan=0.0, posinf=0.0, neginf=0.0)
    seq_loss = torch.clamp(seq_loss, min=-10.0, max=10.0)
    return -seq_loss.mean()


def compute_policy_entropy(logits: torch.Tensor, completion_mask: torch.Tensor,
                           logits_to_keep: int) -> torch.Tensor:
    """Mean per-token entropy over valid completion positions."""
    shift_logits = logits[:, :-1, :].float()[:, -logits_to_keep:, :]  # [B, T, V]
    log_p = F.log_softmax(shift_logits, dim=-1)
    token_entropy = -(torch.exp(log_p) * log_p).sum(dim=-1)            # [B, T]
    total = completion_mask.sum().clamp(min=1)
    return (token_entropy * completion_mask).sum() / total


# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------

_GEN_INPUT_KEYS = ('input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img',
                   'pixel_values_videos', 'video_grid_thw')


def model_forward(model, full_ids, attention_mask, g_batch, device):
    """Forward pass returning model output (logits).

    Uses the provided attention_mask (prompt + completion) rather than
    all-ones so the model doesn't attend to padding tokens beyond EOS.
    """
    return model(
        input_ids=full_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=None,
        point_clouds=g_batch['point_clouds'].to(device),
        is_pc=g_batch['is_pc'].to(device),
        is_img=g_batch['is_img'].to(device),
        pixel_values_videos=(
            g_batch['pixel_values_videos'].to(device)
            if g_batch.get('pixel_values_videos') is not None else None),
        video_grid_thw=(
            g_batch['video_grid_thw'].to(device)
            if g_batch.get('video_grid_thw') is not None else None),
    )


# ---------------------------------------------------------------------------
# Batched / sequential rollout generation
# ---------------------------------------------------------------------------

def generate_rollouts(model, single_batch: dict, G: int, args,
                      pad_token_id: int, processor=None) -> torch.Tensor:
    """Generate G completions for one prompt.

    Tries batched generation first (fast on H100); falls back to sequential
    on OOM and sticks to sequential for the rest of the run.
    Returns tensor of shape [G, max_len] on CPU.
    """
    # Qwen2VL requires padding_side='left' for generate().
    # eval_one_pass / collate may mutate it — restore unconditionally.
    if processor is not None:
        processor.tokenizer.padding_side = 'left'

    # Block video tokens — model must not emit them in CadQuery code
    bad_words = None
    if hasattr(model, 'config') and hasattr(model.config, 'video_token_id'):
        bad_words = [[model.config.video_token_id]]

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      temperature=1.0, top_p=1.0, top_k=50,
                      bad_words_ids=bad_words)
    sequential = getattr(args, 'sequential_generation', False)
    device = next(model.parameters()).device

    if not sequential:
        try:
            expanded = {}
            for k in _GEN_INPUT_KEYS:
                v = single_batch.get(k)
                if v is None:
                    expanded[k] = None
                elif isinstance(v, torch.Tensor):
                    expanded[k] = v.repeat(G, *([1] * (v.dim() - 1))).to(device)
                else:
                    expanded[k] = v
            with torch.no_grad():
                out = model.generate(**expanded, **gen_kwargs)
            return out.cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'[rollout] batched G={G} OOM — switching to sequential for rest of run')
            args.sequential_generation = True

    # Sequential fallback: generate one at a time
    single_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in ((k, single_batch.get(k)) for k in _GEN_INPUT_KEYS)}
    generated_ids_list = []
    with torch.no_grad():
        for _ in range(G):
            ids = model.generate(**single_kwargs, **gen_kwargs)
            generated_ids_list.append(ids.cpu())

    max_len = max(ids.shape[1] for ids in generated_ids_list)
    padded = []
    for ids in generated_ids_list:
        if ids.shape[1] < max_len:
            pad = torch.full((1, max_len - ids.shape[1]), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=1)
        padded.append(ids)
    return torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
# CPPO step  (aligned with grpo_mm.py)
# ---------------------------------------------------------------------------

def cppo_step(model, optimizer, item: dict, processor, args) -> dict:
    """One Dr. CPPO / GRPO update step.

    1. Collate prompt → batch (CPU)
    2. Generate G completions (batched or sequential)
    3. Compute IoU rewards → unnormalised advantages  (NO std normalisation)
    4. Select top_N by |advantage|
    5. Compute old-policy log probs from the *current* model (no_grad)
       — old_log_probs stay fixed across the batch_updates inner iterations
    6. Do batch_updates gradient steps with per-token clipped PPO loss
    """
    device = next(model.parameters()).device
    G = args.G
    N = min(args.top_N, G)
    eos_id = processor.tokenizer.eos_token_id

    collate_item = {k: v for k, v in item.items() if not k.startswith('_')}
    batch = collate([collate_item], processor=processor, n_points=256, eval=True)
    prompt_len  = batch['input_ids'].shape[1]
    gt_mesh_path = item['gt_mesh_path']

    single_batch = {k: batch.get(k) for k in _GEN_INPUT_KEYS}

    t_gen = time.perf_counter()
    generated_ids = generate_rollouts(model, single_batch, G, args,
                                      processor.tokenizer.pad_token_id,
                                      processor=processor)
    gen_seconds = time.perf_counter() - t_gen

    code_strings = [
        processor.decode(generated_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(G)
    ]

    rewards   = compute_rewards_parallel(
        code_strings, [gt_mesh_path] * G, workers=args.reward_workers)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r    = rewards_t.mean()
    std_r     = rewards_t.std()

    # Advantages: raw (reward - group_mean), NO std normalisation (matches ref)
    advantages_raw = rewards_t - mean_r
    _, top_idx = torch.topk(advantages_raw.abs(), N)

    # Build per-token tensors for selected rollouts
    sel_ids_cpu        = generated_ids[top_idx]           # [N, full_len]
    completion_ids_cpu = sel_ids_cpu[:, prompt_len:]      # [N, T]
    logits_to_keep     = completion_ids_cpu.shape[1]

    comp_mask_cpu   = create_completion_mask(completion_ids_cpu, eos_id)  # [N, T]
    g_batch         = expand_batch(batch, G)
    sel_g_batch_cpu = slice_batch(g_batch, top_idx)

    # Full attention mask: prompt mask (from collation) + completion mask
    prompt_mask_cpu   = sel_g_batch_cpu['attention_mask']  # [N, prompt_len]
    full_attn_mask_cpu = torch.cat(
        [prompt_mask_cpu, comp_mask_cpu.long()], dim=1)   # [N, full_len]

    advantages_sel = advantages_raw[top_idx].unsqueeze(1)  # [N, 1] — broadcast over tokens

    # Move to GPU
    torch.cuda.empty_cache()
    sel_ids       = sel_ids_cpu.to(device)
    full_attn_mask = full_attn_mask_cpu.to(device)
    comp_mask     = comp_mask_cpu.to(device)
    advantages    = advantages_sel.to(device)
    sel_g_batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in sel_g_batch_cpu.items()}

    # Compute old log-probs from current model BEFORE any gradient update
    # (identical to grpo_mm: old_log_probs computed at rollout time, reused
    #  for all batch_updates inner iterations)
    with torch.no_grad():
        old_out  = model_forward(model, sel_ids, full_attn_mask, sel_g_batch, device)
        old_lp   = compute_token_log_probs(old_out.logits, sel_ids, logits_to_keep).detach()

    last_metrics = {}
    for _ in range(args.batch_updates):
        model.train()
        new_out  = model_forward(model, sel_ids, full_attn_mask, sel_g_batch, device)
        new_lp   = compute_token_log_probs(new_out.logits, sel_ids, logits_to_keep)

        loss = cppo_loss_fn(new_lp, old_lp, advantages, comp_mask,
                            args.eps_high, args.eps_low)

        optimizer.zero_grad()
        loss.backward()
        # grad clip 0.1 — matches ref; 1.0 (previous value) was 10× too loose
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        # Diagnostics (per-sequence ratio for logging convenience)
        with torch.no_grad():
            ratio_tok = torch.exp(new_lp - old_lp)              # [N, T]
            # Sequence-level ratio: exp of mean token log-ratio
            ratio_seq = (ratio_tok * comp_mask).sum(1) / comp_mask.sum(1).clamp(min=1)
            entropy   = compute_policy_entropy(new_out.logits, comp_mask, logits_to_keep)

            clip_lower = (ratio_tok < 1 - args.eps_low)
            clip_upper = (ratio_tok > 1 + args.eps_high)
            clip_frac  = ((clip_lower | clip_upper).float() * comp_mask).sum() \
                         / comp_mask.sum().clamp(min=1)
            kl_approx  = ((ratio_tok - 1 - torch.log(ratio_tok.clamp(min=1e-8)))
                          * comp_mask).sum() / comp_mask.sum().clamp(min=1)

            _degen = std_r.item() < 1e-6

        last_metrics = {
            'train/loss':            loss.item(),
            'train/mean_reward':     mean_r.item(),
            'train/reward_std':      std_r.item(),
            'train/reward_max':      rewards_t.max().item(),
            'train/reward_min':      rewards_t.min().item(),
            'train/entropy':         entropy.item(),
            'train/clip_fraction':   clip_frac.item(),
            'train/clip_lower_frac': ((clip_lower.float() * comp_mask).sum()
                                      / comp_mask.sum().clamp(min=1)).item(),
            'train/clip_upper_frac': ((clip_upper.float() * comp_mask).sum()
                                      / comp_mask.sum().clamp(min=1)).item(),
            'train/ratio_mean':      ratio_seq.mean().item(),
            'train/ratio_std':       ratio_seq.std().item(),
            'train/kl_approx':       kl_approx.item(),
            'train/adv_pos_frac':    float('nan') if _degen else
                                     (advantages > 0).float().mean().item(),
            'train/adv_abs_mean':    advantages.abs().mean().item(),
            # 4-quadrant IS × advantage decomposition (sequence-level ratio)
            'train/q_pp': float('nan') if _degen else
                          ((advantages.squeeze(1) > 0) & (ratio_seq > 1.0)).float().mean().item(),
            'train/q_pn': float('nan') if _degen else
                          ((advantages.squeeze(1) > 0) & (ratio_seq <= 1.0)).float().mean().item(),
            'train/q_np': float('nan') if _degen else
                          ((advantages.squeeze(1) <= 0) & (ratio_seq > 1.0)).float().mean().item(),
            'train/q_nn': float('nan') if _degen else
                          ((advantages.squeeze(1) <= 0) & (ratio_seq <= 1.0)).float().mean().item(),
            'train/gen_seconds':     gen_seconds,
            '_rewards_list':         rewards,
            '_ratio_list':           ratio_seq.detach().cpu().tolist(),
            '_adv_list':             advantages.squeeze(1).detach().cpu().tolist(),
        }

    return last_metrics


# ---------------------------------------------------------------------------
# Checkpoint rotation
# ---------------------------------------------------------------------------

def _rotate_checkpoints(output_dir: str, save_total_limit: Optional[int]):
    """Delete oldest checkpoint-XXXXX dirs when limit is exceeded."""
    if not save_total_limit or save_total_limit <= 0:
        return
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith('checkpoint-'):
            try:
                step = int(name[len('checkpoint-'):])
                checkpoints.append((step, os.path.join(output_dir, name)))
            except ValueError:
                pass
    checkpoints.sort()
    for _, path in checkpoints[:-save_total_limit]:
        shutil.rmtree(path, ignore_errors=True)
        print(f'[checkpoint] deleted old checkpoint: {path}')


# ---------------------------------------------------------------------------
# CPPO training loop
# ---------------------------------------------------------------------------

def train_cppo(model, optimizer, dataset, processor,
               val_examples, use_wandb, args):
    """Main Dr. CPPO training loop.

    No longer requires a separate old_model — old log-probs are computed
    from the current model at rollout time (matching the reference).
    """
    log_path = os.path.join(args.output_dir, 'log.txt')
    step = getattr(args, 'start_step', 0)
    if step > 0:
        print(f'Resuming from step {step}')
    indices = list(range(len(dataset)))

    if val_examples and step == 0:
        print('\n[eval step=0 (pre-training baseline)]')
        try:
            val_metrics = run_validation(model, val_examples, processor, args)
            log_eval(val_metrics, step=0, log_path=log_path, use_wandb=use_wandb)
        except Exception as e:
            print(f'[eval step=0] failed (skipping): {e}')
        model.train()

    pbar = tqdm(total=args.max_steps, desc='Dr. CPPO')
    while step < args.max_steps:
        np.random.shuffle(indices)
        for idx in indices:
            if step >= args.max_steps:
                break
            try:
                metrics = cppo_step(model, optimizer, dataset[idx], processor, args)
            except Exception as e:
                print(f'[step {step}] cppo_step error: {e}')
                continue

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{metrics['train/loss']:.3f}",
                reward=f"{metrics['train/mean_reward']:.2f}",
                H=f"{metrics['train/entropy']:.2f}")

            if step % args.log_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                log_line = (
                    f"step={step}"
                    f" loss={metrics['train/loss']:.4f}"
                    f" average_reward={metrics['train/mean_reward']:.4f}"
                    f" train/reward_std={metrics['train/reward_std']:.4f}"
                    f" train/reward_max={metrics['train/reward_max']:.4f}"
                    f" train/reward_min={metrics['train/reward_min']:.4f}"
                    f" train/entropy={metrics['train/entropy']:.4f}"
                    f" train/clip_fraction={metrics['train/clip_fraction']:.4f}"
                    f" train/kl_approx={metrics['train/kl_approx']:.6f}"
                    f" train/ratio_mean={metrics['train/ratio_mean']:.4f}"
                    f" train/lr={lr:.2e}"
                )
                with open(log_path, 'a') as f:
                    f.write(log_line + '\n')

                if use_wandb:
                    wandb.log({
                        'loss':           metrics['train/loss'],
                        'average_reward': metrics['train/mean_reward'],
                        'train/reward_std':      metrics['train/reward_std'],
                        'train/reward_max':      metrics['train/reward_max'],
                        'train/reward_min':      metrics['train/reward_min'],
                        'train/entropy':         metrics['train/entropy'],
                        'train/kl_approx':       metrics['train/kl_approx'],
                        'train/clip_fraction':   metrics['train/clip_fraction'],
                        'train/clip_lower_frac': metrics['train/clip_lower_frac'],
                        'train/clip_upper_frac': metrics['train/clip_upper_frac'],
                        'train/ratio_mean':      metrics['train/ratio_mean'],
                        'train/ratio_std':       metrics['train/ratio_std'],
                        'train/adv_pos_frac':    metrics['train/adv_pos_frac'],
                        'train/adv_abs_mean':    metrics['train/adv_abs_mean'],
                        'train/q_pp':            metrics['train/q_pp'],
                        'train/q_pn':            metrics['train/q_pn'],
                        'train/q_np':            metrics['train/q_np'],
                        'train/q_nn':            metrics['train/q_nn'],
                        'train/gen_seconds':     metrics['train/gen_seconds'],
                        'train/lr':              lr,
                        'dist/rewards': wandb.Histogram(metrics['_rewards_list']),
                        'dist/ratios':  wandb.Histogram(metrics['_ratio_list']),
                        'dist/advs':    wandb.Histogram(metrics['_adv_list']),
                    }, step=step)

            if val_examples and step % args.eval_steps == 0:
                print(f'\n[eval step={step}]')
                val_metrics = run_validation(model, val_examples, processor, args)
                log_eval(val_metrics, step=step, log_path=log_path, use_wandb=use_wandb)
                model.train()

            if step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                _rotate_checkpoints(args.output_dir,
                                    getattr(args, 'save_total_limit', None))

    pbar.close()
    final_dir = os.path.join(args.output_dir, 'checkpoint-final')
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f'Training complete. Final checkpoint → {final_dir}')
