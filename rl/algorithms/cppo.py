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
from collections import deque

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
            out[k] = v.repeat_interleave(G, dim=0)
        elif isinstance(v, list):
            out[k] = [x for x in v for _ in range(G)]
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

    use_cache=False is mandatory: with use_cache=True the model creates a
    DynamicCache() even when past_key_values=None, making past_key_values
    non-null; _update_causal_mask then checks attention_mask[:, -1] and
    raises ValueError if any completion ended before max_len (right-padding).
    """
    return model(
        input_ids=full_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=None,
        use_cache=False,
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
    """Generate G completions per prompt.

    Tries batched generation first (fast on H100); falls back to sequential
    on OOM and sticks to sequential for the rest of the run.
    Returns tensor of shape [B*G, max_len] on CPU, grouped by prompt.
    """
    # Qwen2VL requires padding_side='left' for generate().
    # eval_one_pass / collate may mutate it — restore unconditionally.
    if processor is not None:
        processor.tokenizer.padding_side = 'left'

    # DDP: .generate() is not available on the DDP wrapper; unwrap to raw model.
    gen_model = model.module if hasattr(model, 'module') else model

    # Block video tokens — model must not emit them in CadQuery code
    bad_words = None
    if hasattr(gen_model, 'config') and hasattr(gen_model.config, 'video_token_id'):
        bad_words = [[gen_model.config.video_token_id]]

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      temperature=1.0, top_p=1.0, top_k=50,
                      early_stopping=False,
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
                    expanded[k] = v.repeat_interleave(G, dim=0).to(device)
                else:
                    expanded[k] = v
            with torch.no_grad():
                out = gen_model.generate(**expanded, **gen_kwargs)
            return out.cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'[rollout] batched G={G} OOM — switching to sequential for rest of run')
            args.sequential_generation = True

    # Sequential fallback: generate one prompt at a time, G rollouts each.
    single_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in ((k, single_batch.get(k)) for k in _GEN_INPUT_KEYS)}
    batch_size = single_kwargs['input_ids'].shape[0]
    generated_ids_list = []
    with torch.no_grad():
        for i in range(batch_size):
            one = {}
            for k, v in single_kwargs.items():
                if isinstance(v, torch.Tensor):
                    one[k] = v[i:i + 1]
                else:
                    one[k] = v
            for _ in range(G):
                ids = gen_model.generate(**one, **gen_kwargs)
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


class IndexBuffer:
    """Simple replay buffer for hard examples (high reward-std prompts)."""

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._buf = deque()

    def __len__(self):
        return len(self._buf)

    def add_many(self, indices):
        for idx in indices:
            if idx is None:
                continue
            self._buf.append(int(idx))
            if len(self._buf) > self.max_size:
                self._buf.popleft()

    def sample(self, n: int):
        n = min(n, len(self._buf))
        if n <= 0:
            return []
        arr = list(self._buf)
        sel = np.random.choice(len(arr), size=n, replace=False)
        return [arr[i] for i in sel]


_NAN_DIAG = {
    'train/clip_fraction':   float('nan'),
    'train/clip_lower_frac': float('nan'),
    'train/clip_upper_frac': float('nan'),
    'train/ratio_mean':      float('nan'),
    'train/ratio_std':       float('nan'),
    'train/kl_approx':       float('nan'),
    'train/adv_pos_frac':    float('nan'),
    'train/kl_q_pp':         float('nan'),
    'train/kl_q_pn':         float('nan'),
    'train/kl_q_np':         float('nan'),
    'train/kl_q_nn':         float('nan'),
    '_ratio_list':           [],
    '_adv_list':             [],
}


def cppo_step(model, optimizer, items, processor, args,
              compute_diag: bool = True) -> dict:
    """Dr. CPPO / GRPO update step over a batch of B prompts.

    Fully batched tensor flow (matches reference grpo_mm.py style):
      Phase 1  — Collate B prompts into one batch; generate all B*G rollouts
                 in one generate() call → [B*G, full_len] (blocked order).
                 Decode + reward all B*G completions in parallel.
                 Row-wise top-N per prompt → flat [B*N, T] PPO tensors.
                 Single model_forward for old_lp on [B*N, T].
      Phase 2  — batch_updates PPO steps; each step is one forward on [B*N, T].
                 No gradient accumulation — the flat [B*N] batch handles all B
                 prompts in one pass exactly as the reference does.
      Phase 3  — KL diagnostics on [B*N, T] flat tensors using last IS only.

    With B=1 this reduces to the single-prompt behaviour.

    Key properties:
      • Per-prompt group-relative advantages: adv_i = reward_i − mean(group)
      • Per-prompt row-wise top-N selection (not global across prompts)
      • Advantages are per-SEQUENCE scalars [B*N, 1] broadcast over tokens —
        no token-level advantage in CPPO/GRPO
      • Degenerate prompts (std≈0) have adv≈0 → contribute ~zero to loss
      • All-degenerate early exit skips the expensive forward passes entirely
    """
    if isinstance(items, dict):
        items = [items]

    B      = len(items)
    G      = args.G
    N      = min(args.top_N, G)
    device = next(model.parameters()).device
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Phase 1: Batched rollout — one collate, one generate, one reward call.
    #
    # generate_rollouts receives the B-item batch and expands it internally
    # via repeat_interleave(G) → [B*G, ...] before calling model.generate.
    # Output [B*G, full_len] is in blocked order: [p0×G, p1×G, ..., p(B-1)×G].
    # All completions start at prompt_len because prompts are left-padded to
    # the same max length by collate().
    # ------------------------------------------------------------------
    collate_items = [{k: v for k, v in it.items() if not k.startswith('_')}
                     for it in items]
    batch      = collate(collate_items, processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]          # left-padded max prompt length

    t_gen = time.perf_counter()
    generated_ids = generate_rollouts(                # [B*G, full_len]
        model, {k: batch.get(k) for k in _GEN_INPUT_KEYS},
        G, args, pad_id, processor)
    gen_seconds = time.perf_counter() - t_gen

    # Average effective generation length across ALL B*G rollouts (CPU, cheap).
    # Uses create_completion_mask to count tokens up to and including first EOS,
    # so padding beyond EOS is excluded. Measures the true generation distribution
    # before top-N selection (which would bias toward extreme advantage sequences).
    _all_comp = generated_ids[:, prompt_len:]                        # [B*G, T]
    _all_mask = create_completion_mask(_all_comp, eos_id)            # [B*G, T] float
    avg_gen_len = _all_mask.sum(dim=1).float().mean().item()

    code_strings = [
        processor.decode(generated_ids[i, prompt_len:], skip_special_tokens=True,
                         clean_up_tokenization_spaces=False)
        for i in range(B * G)
    ]
    gt_paths = [it['gt_mesh_path'] for it in items for _ in range(G)]  # blocked order

    rewards   = compute_rewards_parallel(code_strings, gt_paths,
                                         workers=args.reward_workers)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).view(B, G)  # [B, G]
    # Guard against non-finite reward values before mean/std/topk.
    rewards_t = torch.nan_to_num(
        rewards_t, nan=-10.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    mean_r    = rewards_t.mean(dim=1, keepdim=True)                     # [B, 1]
    std_r     = rewards_t.std(dim=1)                                    # [B]
    adv_raw   = rewards_t - mean_r                                      # [B, G]
    # Extra safety before topk in case any upstream op returns non-finite.
    adv_raw   = torch.nan_to_num(adv_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if adv_raw.abs().max().item() < 1e-6:
        # All B prompts degenerate — advantages≈0, nothing to learn
        return {
            'train/loss':         0.0,
            'train/mean_reward':  rewards_t.mean().item(),
            'train/reward_std':   std_r.mean().item(),
            'train/reward_max':   rewards_t.max().item(),
            'train/reward_min':   rewards_t.min().item(),
            'train/entropy':      float('nan'),
            'train/adv_abs_mean': 0.0,
            'train/adv_mean_seq': 0.0,
            'train/adv_mean_tok': 0.0,
            'train/avg_gen_len':  avg_gen_len,
            'train/gen_seconds':  gen_seconds,
            '_rewards_list':      rewards,
            '_reward_std_groups': std_r.tolist(),
            **_NAN_DIAG,
        }

    # Row-wise top-N per prompt by |advantage| → flat [B*N] index into [B*G]
    _, top_idx = torch.topk(adv_raw.abs(), N, dim=1)                    # [B, N]
    flat_idx   = (torch.arange(B).unsqueeze(1) * G + top_idx).reshape(-1)  # [B*N]

    # Build flat [B*N, T] PPO tensors — single tensor for all prompts
    sel_ids_cpu  = generated_ids[flat_idx]            # [B*N, full_len]
    comp_ids_cpu = sel_ids_cpu[:, prompt_len:]        # [B*N, T]
    logits_to_keep = comp_ids_cpu.shape[1]

    comp_mask_cpu      = create_completion_mask(comp_ids_cpu, eos_id)   # [B*N, T]
    g_batch_cpu        = expand_batch(batch, G)                          # [B*G, ...]
    sel_g_batch_cpu    = slice_batch(g_batch_cpu, flat_idx)              # [B*N, ...]
    prompt_mask_cpu    = sel_g_batch_cpu['attention_mask']               # [B*N, prompt_len]
    full_attn_mask_cpu = torch.cat(
        [prompt_mask_cpu, comp_mask_cpu.long()], dim=1)                  # [B*N, full_len]
    adv_sel = adv_raw.reshape(-1)[flat_idx].unsqueeze(1)                 # [B*N, 1]

    torch.cuda.empty_cache()
    sel_ids     = sel_ids_cpu.to(device)
    full_attn   = full_attn_mask_cpu.to(device)
    comp_mask   = comp_mask_cpu.to(device)
    advantages  = adv_sel.to(device)
    sel_g_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in sel_g_batch_cpu.items()}

    # Single forward for old log-probs on the flat [B*N, T] batch
    with torch.no_grad():
        old_out = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
        old_lp  = compute_token_log_probs(
            old_out.logits, sel_ids, logits_to_keep).detach()            # [B*N, T]

    # ------------------------------------------------------------------
    # Phase 2: batch_updates PPO steps on the flat [B*N, T] batch.
    # One forward per inner step — no gradient accumulation needed.
    # ------------------------------------------------------------------
    last_loss    = 0.0
    last_entropy = float('nan')
    last_new_lp  = None

    for k in range(args.batch_updates):
        is_last = (k == args.batch_updates - 1)
        model.train()
        new_out = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
        new_lp  = compute_token_log_probs(new_out.logits, sel_ids, logits_to_keep)

        loss = cppo_loss_fn(new_lp, old_lp, advantages, comp_mask,
                            args.eps_high, args.eps_low)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        if is_last:
            last_loss = loss.item()
            with torch.no_grad():
                last_entropy = compute_policy_entropy(
                    new_out.logits, comp_mask, logits_to_keep).item()
            last_new_lp = new_lp.detach()

    # ------------------------------------------------------------------
    # Phase 3: Diagnostics on the flat [B*N, T] tensor — no aggregation
    # loop, all prompts handled in one vectorised pass.
    # ------------------------------------------------------------------
    with torch.no_grad():
        if compute_diag and last_new_lp is not None:
            ratio_tok = torch.exp(last_new_lp - old_lp)               # [B*N, T]
            ratio_seq = (ratio_tok * comp_mask).sum(1) \
                        / comp_mask.sum(1).clamp(min=1)                # [B*N]

            clip_lower = (ratio_tok < 1 - args.eps_low)
            clip_upper = (ratio_tok > 1 + args.eps_high)
            n_tok      = comp_mask.sum().clamp(min=1)
            clip_frac  = ((clip_lower | clip_upper).float() * comp_mask).sum() / n_tok

            kl_tok     = (ratio_tok - 1 - torch.log(ratio_tok.clamp(min=1e-8))) \
                         * comp_mask
            kl_per_seq = kl_tok.sum(dim=1)                             # [B*N]
            kl_total   = kl_per_seq.sum().clamp(min=1e-8)
            kl_approx  = kl_per_seq.mean() / comp_mask.sum(1).mean().clamp(min=1)

            adv_sq  = advantages.squeeze(1)                            # [B*N]
            kl_q_pp = kl_per_seq[(adv_sq > 0)  & (ratio_seq > 1.0)].sum() / kl_total
            kl_q_pn = kl_per_seq[(adv_sq > 0)  & (ratio_seq <= 1.0)].sum() / kl_total
            kl_q_np = kl_per_seq[(adv_sq <= 0) & (ratio_seq > 1.0)].sum() / kl_total
            kl_q_nn = kl_per_seq[(adv_sq <= 0) & (ratio_seq <= 1.0)].sum() / kl_total

            diag = {
                'train/clip_fraction':   clip_frac.item(),
                'train/clip_lower_frac': ((clip_lower.float() * comp_mask).sum()
                                          / n_tok).item(),
                'train/clip_upper_frac': ((clip_upper.float() * comp_mask).sum()
                                          / n_tok).item(),
                'train/ratio_mean':      ratio_seq.mean().item(),
                'train/ratio_std':       ratio_seq.std().item(),
                'train/kl_approx':       kl_approx.item(),
                'train/adv_pos_frac':    (advantages > 0).float().mean().item(),
                'train/kl_q_pp':         kl_q_pp.item(),
                'train/kl_q_pn':         kl_q_pn.item(),
                'train/kl_q_np':         kl_q_np.item(),
                'train/kl_q_nn':         kl_q_nn.item(),
                '_ratio_list':           ratio_seq.cpu().tolist(),
                '_adv_list':             adv_sq.cpu().tolist(),
            }
        else:
            diag = dict(_NAN_DIAG)

    _n_tok = comp_mask.sum().clamp(min=1)
    return {
        'train/loss':         last_loss,
        'train/mean_reward':  rewards_t.mean().item(),
        'train/reward_std':   rewards_t.std().item(),
        'train/reward_max':   rewards_t.max().item(),
        'train/reward_min':   rewards_t.min().item(),
        'train/entropy':      last_entropy,
        'train/adv_abs_mean': advantages.abs().mean().item(),
        # adv_mean_seq: mean of per-sequence advantage scalars [B*N].
        #   Should be ≈0 (reward−group_mean; slight drift from top-N selection).
        # adv_mean_tok: token-weighted mean (advantages broadcast over comp_mask).
        #   Also ≈0 but reveals if longer completions skew the advantage sign.
        'train/adv_mean_seq': advantages.mean().item(),
        'train/adv_mean_tok': ((advantages * comp_mask).sum() / _n_tok).item(),
        'train/avg_gen_len':  avg_gen_len,
        'train/gen_seconds':  gen_seconds,
        '_rewards_list':      rewards,
        '_reward_std_groups': std_r.tolist(),
        **diag,
    }


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
               val_examples, use_wandb, args, rank=0, world_size=1):
    """Main Dr. CPPO training loop.

    No longer requires a separate old_model — old log-probs are computed
    from the current model at rollout time (matching the reference).

    DDP: each rank processes its own data shard (DistributedSampler).
    Side effects (W&B, eval, checkpoint save) run on rank 0 only.
    """
    import torch.distributed as _dist
    from torch.utils.data.distributed import DistributedSampler

    is_distributed = world_size > 1
    log_path = os.path.join(args.output_dir, 'log.txt')
    step = getattr(args, 'start_step', 0)
    if step > 0 and rank == 0:
        print(f'Resuming from step {step}')
    indices = list(range(len(dataset)))
    batch_size = max(1, int(getattr(args, 'batch_size', 1)))
    use_buffer = bool(getattr(args, 'use_buffer', False))
    buffer_sample_size = int(getattr(args, 'buffer_sample_size', batch_size))
    buffer_expand_frac = float(getattr(args, 'buffer_expand_frac', 0.5))
    replay_buffer = (
        IndexBuffer(max_size=int(getattr(args, 'buffer_max_size', 4096)))
        if use_buffer else None
    )

    if val_examples and step == 0 and rank == 0:
        print('\n[eval step=0 (pre-training baseline)]')
        try:
            raw_model = model.module if hasattr(model, 'module') else model
            val_metrics = run_validation(raw_model, val_examples, processor, args)
            log_eval(val_metrics, step=0, log_path=log_path, use_wandb=use_wandb)
        except Exception as e:
            print(f'[eval step=0] failed (skipping): {e}')
        model.train()

    # DistributedSampler gives each rank a disjoint shard of the dataset.
    # set_epoch() re-shuffles with a different seed each epoch.
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    pbar = tqdm(total=args.max_steps, desc='Dr. CPPO', disable=(rank != 0))
    epoch = 0
    while step < args.max_steps:
        if is_distributed:
            sampler.set_epoch(epoch)
            rank_indices = list(sampler)
        else:
            np.random.shuffle(indices)
            rank_indices = indices
        epoch += 1
        for start in range(0, len(rank_indices), batch_size):
            if step >= args.max_steps:
                break
            base_indices = rank_indices[start:start + batch_size]
            train_indices = list(base_indices)

            if replay_buffer is not None and len(replay_buffer) > 0:
                train_indices.extend(replay_buffer.sample(buffer_sample_size))

            batch_items = []
            for idx in train_indices:
                item = dict(dataset[idx])
                item['_dataset_idx'] = int(idx)
                batch_items.append(item)

            try:
                # compute_diag=True only on steps that will be logged.
                # step+1 because step is incremented after cppo_step returns.
                compute_diag = ((step + 1) % args.log_steps == 0)
                metrics = cppo_step(model, optimizer, batch_items, processor, args,
                                    compute_diag=compute_diag)
                if replay_buffer is not None:
                    std_list = metrics.get('_reward_std_groups', [])
                    n_pick = int(len(std_list) * buffer_expand_frac)
                    if n_pick > 0:
                        std_arr = np.array(std_list)
                        top_local = std_arr.argsort()[-n_pick:]
                        picked = [train_indices[int(i)] for i in top_local
                                  if int(i) < len(train_indices)]
                        replay_buffer.add_many(picked)
            except Exception as e:
                print(f'[step {step}] cppo_step error: {e}')
                continue

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{metrics['train/loss']:.3f}",
                reward=f"{metrics['train/mean_reward']:.2f}",
                H=f"{metrics['train/entropy']:.2f}")

            if step % args.log_steps == 0 and rank == 0:
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
                    f" train/kl_q_np={metrics['train/kl_q_np']:.4f}"
                    f" train/kl_q_nn={metrics['train/kl_q_nn']:.4f}"
                    f" train/kl_q_pp={metrics['train/kl_q_pp']:.4f}"
                    f" train/adv_mean_seq={metrics['train/adv_mean_seq']:.4f}"
                    f" train/adv_mean_tok={metrics['train/adv_mean_tok']:.4f}"
                    f" train/avg_gen_len={metrics['train/avg_gen_len']:.1f}"
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
                        'train/adv_mean_seq':    metrics['train/adv_mean_seq'],
                        'train/adv_mean_tok':    metrics['train/adv_mean_tok'],
                        'train/avg_gen_len':     metrics['train/avg_gen_len'],
                        'train/kl_q_pp':         metrics['train/kl_q_pp'],
                        'train/kl_q_pn':         metrics['train/kl_q_pn'],
                        'train/kl_q_np':         metrics['train/kl_q_np'],
                        'train/kl_q_nn':         metrics['train/kl_q_nn'],
                        'train/gen_seconds':     metrics['train/gen_seconds'],
                        'train/lr':              lr,
                        'dist/rewards': wandb.Histogram(metrics['_rewards_list']),
                        'dist/ratios':  wandb.Histogram(metrics['_ratio_list']),
                        'dist/advs':    wandb.Histogram(metrics['_adv_list']),
                    }, step=step)

            if val_examples and step % args.eval_steps == 0:
                if is_distributed:
                    _dist.barrier()   # sync before eval
                if rank == 0:
                    print(f'\n[eval step={step}]')
                    raw_model = model.module if hasattr(model, 'module') else model
                    val_metrics = run_validation(raw_model, val_examples, processor, args)
                    log_eval(val_metrics, step=step, log_path=log_path, use_wandb=use_wandb)
                if is_distributed:
                    _dist.barrier()   # non-rank-0 wait for rank-0 eval
                model.train()

            if step % args.save_steps == 0 and rank == 0:
                ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                save_model = model.module if hasattr(model, 'module') else model
                save_model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                _rotate_checkpoints(args.output_dir,
                                    getattr(args, 'save_total_limit', None))

    pbar.close()
    if rank == 0:
        final_dir = os.path.join(args.output_dir, 'checkpoint-final')
        save_model = model.module if hasattr(model, 'module') else model
        save_model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f'Training complete. Final checkpoint → {final_dir}')
