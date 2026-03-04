"""Dr. CPPO / GRPO algorithm — matches official rl branch grpo_mm.py."""

import os
import time
import copy

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


def make_labels(full_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    """Build label tensor: -100 for prompt tokens and post-EOS tokens."""
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    for i in range(full_ids.shape[0]):
        eos_pos = (full_ids[i, prompt_len:] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            labels[i, prompt_len + eos_pos[0].item() + 1:] = -100
    return labels


def model_forward(model, full_ids, g_batch, device):
    """Forward pass returning model output (logits)."""
    return model(
        input_ids=full_ids.to(device),
        attention_mask=torch.ones_like(full_ids).to(device),
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


def compute_policy_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean per-token entropy over unmasked (completion) positions."""
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100)
    log_p = F.log_softmax(shift_logits, dim=-1)
    token_entropy = -(torch.exp(log_p) * log_p).sum(dim=-1)
    return (token_entropy * mask.float()).sum() / mask.float().sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Batched / sequential rollout generation
# ---------------------------------------------------------------------------

_GEN_INPUT_KEYS = ('input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img',
                   'pixel_values_videos', 'video_grid_thw')


def generate_rollouts(model, single_batch: dict, G: int, args,
                      pad_token_id: int) -> torch.Tensor:
    """Generate G completions for one prompt.

    Tries batched generation first (fast on H100); falls back to sequential
    on OOM and sticks to sequential for the rest of the run.
    Returns tensor of shape [G, max_len] on CPU.
    """
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      temperature=1.0, top_p=1.0, top_k=50)
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
# CPPO step
# ---------------------------------------------------------------------------

def cppo_step(model, old_model, optimizer, item: dict, processor, args) -> dict:
    """One Dr. CPPO / GRPO update step.

    1. Collate prompt → batch (CPU)
    2. Generate G completions (batched or sequential)
    3. Compute IoU rewards → unnormalised advantages
    4. Select top_N by |advantage|
    5. Compute old-policy log probs (no grad; old model CPU-offloaded)
    6. Do batch_updates gradient steps
    """
    device = next(model.parameters()).device
    G = args.G
    N = min(args.top_N, G)

    collate_item = {k: v for k, v in item.items() if not k.startswith('_')}
    batch = collate([collate_item], processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]
    gt_mesh_path = item['gt_mesh_path']

    single_batch = {k: batch.get(k) for k in _GEN_INPUT_KEYS}

    t_gen = time.perf_counter()
    generated_ids = generate_rollouts(model, single_batch, G, args,
                                      processor.tokenizer.pad_token_id)
    gen_seconds = time.perf_counter() - t_gen

    code_strings = [
        processor.decode(generated_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(G)
    ]

    rewards = compute_rewards_parallel(
        code_strings, [gt_mesh_path] * G, workers=args.reward_workers)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = rewards_t.mean()
    std_r  = rewards_t.std()
    advantages_raw = rewards_t - mean_r

    _, top_idx = torch.topk(advantages_raw.abs(), N)

    eos_id = processor.tokenizer.eos_token_id
    sel_ids_cpu    = generated_ids[top_idx]
    sel_labels_cpu = make_labels(sel_ids_cpu, prompt_len, eos_id)
    g_batch        = expand_batch(batch, G)
    sel_g_batch_cpu = slice_batch(g_batch, top_idx)
    advantages = (advantages_raw[top_idx] / (std_r + 1e-8))

    torch.cuda.empty_cache()
    sel_ids     = sel_ids_cpu.to(device)
    sel_labels  = sel_labels_cpu.to(device)
    sel_g_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in sel_g_batch_cpu.items()}
    advantages  = advantages.to(device)

    old_model.to(device)
    old_model.eval()
    with torch.no_grad():
        old_out      = model_forward(old_model, sel_ids, sel_g_batch, device)
        old_log_probs = Cadrille.compute_sequence_logprob(
            old_out.logits, sel_labels, mean_reduction=True).detach()
    old_model.cpu()
    torch.cuda.empty_cache()

    last_metrics = {}
    for _ in range(args.batch_updates):
        model.train()
        new_out      = model_forward(model, sel_ids, sel_g_batch, device)
        new_log_probs = Cadrille.compute_sequence_logprob(
            new_out.logits, sel_labels, mean_reduction=True)
        entropy = compute_policy_entropy(new_out.logits, sel_labels)

        ratio   = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - args.eps_low, 1.0 + args.eps_high)
        loss    = -torch.mean(torch.min(ratio * advantages, clipped * advantages))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        clip_lower = (ratio < 1 - args.eps_low)
        clip_upper = (ratio > 1 + args.eps_high)
        kl_approx  = (ratio - 1 - torch.log(ratio.clamp(min=1e-8))).mean()
        _degen     = std_r.item() < 1e-6
        last_metrics = {
            'train/loss':            loss.item(),
            'train/mean_reward':     mean_r.item(),
            'train/reward_std':      std_r.item(),
            'train/reward_max':      rewards_t.max().item(),
            'train/reward_min':      rewards_t.min().item(),
            'train/entropy':         entropy.item(),
            'train/clip_fraction':   (clip_lower | clip_upper).float().mean().item(),
            'train/clip_lower_frac': clip_lower.float().mean().item(),
            'train/clip_upper_frac': clip_upper.float().mean().item(),
            'train/ratio_mean':      ratio.mean().item(),
            'train/ratio_std':       ratio.std().item(),
            'train/kl_approx':       kl_approx.item(),
            'train/adv_pos_frac':    float('nan') if _degen else (advantages > 0).float().mean().item(),
            'train/adv_abs_mean':    advantages.abs().mean().item(),
            # 4-quadrant IS × advantage decomposition (NaN when degenerate)
            'train/q_pp': float('nan') if _degen else ((advantages > 0) & (ratio > 1.0)).float().mean().item(),
            'train/q_pn': float('nan') if _degen else ((advantages > 0) & (ratio <= 1.0)).float().mean().item(),
            'train/q_np': float('nan') if _degen else ((advantages <= 0) & (ratio > 1.0)).float().mean().item(),
            'train/q_nn': float('nan') if _degen else ((advantages <= 0) & (ratio <= 1.0)).float().mean().item(),
            'train/gen_seconds':     gen_seconds,
            '_rewards_list':         rewards,
            '_ratio_list':           ratio.detach().cpu().tolist(),
            '_adv_list':             advantages.detach().cpu().tolist(),
        }

    return last_metrics


# ---------------------------------------------------------------------------
# CPPO training loop
# ---------------------------------------------------------------------------

def train_cppo(model, old_model, optimizer, dataset, processor,
               val_examples, use_wandb, args):
    log_path = os.path.join(args.output_dir, 'log.txt')
    step = 0
    indices = list(range(len(dataset)))

    if val_examples:
        print('\n[eval step=0 (pre-training baseline)]')
        val_metrics = run_validation(model, val_examples, processor, args)
        log_eval(val_metrics, step=0, log_path=log_path, use_wandb=use_wandb)
        model.train()

    pbar = tqdm(total=args.max_steps, desc='Dr. CPPO')
    while step < args.max_steps:
        np.random.shuffle(indices)
        for idx in indices:
            if step >= args.max_steps:
                break
            try:
                metrics = cppo_step(
                    model, old_model, optimizer, dataset[idx], processor, args)
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
                    f" train/clip_lower_frac={metrics['train/clip_lower_frac']:.4f}"
                    f" train/clip_upper_frac={metrics['train/clip_upper_frac']:.4f}"
                    f" train/kl_approx={metrics['train/kl_approx']:.6f}"
                    f" train/adv_pos_frac={metrics['train/adv_pos_frac']:.4f}"
                    f" train/adv_abs_mean={metrics['train/adv_abs_mean']:.4f}"
                    f" train/q_pp={metrics['train/q_pp']:.4f}"
                    f" train/q_pn={metrics['train/q_pn']:.4f}"
                    f" train/q_np={metrics['train/q_np']:.4f}"
                    f" train/q_nn={metrics['train/q_nn']:.4f}"
                    f" train/ratio_mean={metrics['train/ratio_mean']:.4f}"
                    f" train/ratio_std={metrics['train/ratio_std']:.4f}"
                    f" train/gen_seconds={metrics['train/gen_seconds']:.2f}"
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

            if step % args.K_update == 0:
                old_model.load_state_dict(
                    {k: v.cpu() for k, v in model.state_dict().items()})

            if step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

    pbar.close()
    final_dir = os.path.join(args.output_dir, 'checkpoint-final')
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f'Training complete. Final checkpoint → {final_dir}')
