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
import threading
from typing import Optional
from collections import deque

# ---------------------------------------------------------------------------
# Async HuggingFace checkpoint upload
# ---------------------------------------------------------------------------

_bg_upload_threads: list = []


def _upload_checkpoint_async(ckpt_dir: str, repo_id: str, token: Optional[str] = None) -> None:
    """Upload a checkpoint directory to HuggingFace in a background thread.

    Organises as: <repo_id>/<run_name>/<checkpoint_name>/
    Errors are printed but never crash training.
    """
    def _run():
        try:
            from huggingface_hub import HfApi
            api  = HfApi(token=token or os.environ.get('HF_TOKEN'))
            run  = os.path.basename(os.path.dirname(ckpt_dir))
            name = os.path.basename(ckpt_dir)
            api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=repo_id,
                repo_type='model',
                path_in_repo=f'{run}/{name}',
            )
            print(f'[HF] ✓ {name} → {repo_id}/{run}/{name}', flush=True)
        except Exception as e:
            print(f'[HF] Upload failed for {os.path.basename(ckpt_dir)}: {e}', flush=True)

    t = threading.Thread(target=_run, daemon=False,
                         name=f'hf-upload-{os.path.basename(ckpt_dir)}')
    t.start()
    _bg_upload_threads.append(t)
    print(f'[HF] Upload queued: {os.path.basename(ckpt_dir)} → {repo_id}', flush=True)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cadrille import Cadrille, collate
from rl.reward import compute_rewards_parallel, get_and_reset_pool_crashes
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
    # Causal-LM shift: logits[:, i] predicts token[:, i+1].
    # Keep in the model's native dtype (bf16) — float32 cast doubles memory usage
    # (~1.6 GB vs ~0.8 GB for the log_probs tensor in the autograd graph) and is
    # unnecessary on Ada/Ampere GPUs where bf16 log_softmax is numerically stable.
    shift_logits = logits[:, :-1, :][:, -logits_to_keep:, :]  # [B, T, V]
    completion_ids = full_ids[:, -logits_to_keep:]             # [B, T]
    log_probs = F.log_softmax(shift_logits, dim=-1)            # [B, T, V]
    # Clamp IDs to [0, V-1] to prevent CUDA device-side assert if any generated
    # token ID (e.g. a vision special token) is >= vocab_size.
    completion_ids = completion_ids.clamp(0, shift_logits.shape[-1] - 1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]


def cppo_loss_fn(new_lp: torch.Tensor, old_lp: torch.Tensor,
                 advantages: torch.Tensor, completion_mask: torch.Tensor,
                 eps_high: float, eps_low: float) -> tuple:
    """Clipped PPO surrogate loss.  Identical to grpo_mm.grpo_loss.

    Args:
        new_lp, old_lp:  [B, T] per-token log probs
        advantages:       [B, 1] one scalar per sequence (broadcasts over T)
        completion_mask:  [B, T] float — 1 for valid tokens, 0 after EOS/pad
        eps_high, eps_low: clip bounds

    Returns:
        (loss, seq_loss_detached) — scalar loss (minimised) and [B] per-sequence
        surrogate values (detached, before negation) for contribution analysis.
    """
    ratio   = torch.exp(new_lp - old_lp)                          # [B, T]
    surr1   = ratio * advantages
    surr2   = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages
    per_tok = torch.min(surr1, surr2)                              # [B, T]

    n_tok    = completion_mask.sum(dim=1).clamp(min=1)             # [B]
    seq_loss = (per_tok * completion_mask).sum(dim=1) / n_tok      # [B]
    seq_loss = torch.nan_to_num(seq_loss, nan=0.0, posinf=0.0, neginf=0.0)
    seq_loss = torch.clamp(seq_loss, min=-1.0, max=1.0)
    return -seq_loss.mean(), seq_loss.detach()


def compute_policy_entropy(log_probs: torch.Tensor,
                           completion_mask: torch.Tensor) -> torch.Tensor:
    """Per-token policy entropy estimated from sampled token log-probs.

    Uses the identity: E_{a~π}[-log π(a)] = H(π)
    This is a memory-efficient, unbiased estimate of mean per-token entropy
    that only requires the log-probs of GENERATED tokens ([B*N, T] tensor),
    not the full vocabulary distribution ([B*N, T, V] = ~600 MB+).

    Compared to the full-vocab sum, this estimate:
    - Has higher variance (noisy per-token estimate vs deterministic sum)
    - Has the same magnitude in nats (both equal log(V) at uniform dist)
    - Is gradient-compatible: grads flow through log_probs → model weights
    """
    total = completion_mask.sum().clamp(min=1)
    return -(log_probs * completion_mask).sum() / total


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

    # Block vision tokens — model must not emit image/video tokens in CadQuery code.
    # Both image_token_id and video_token_id must be blocked; previously only
    # video_token_id was blocked, which could allow image tokens through and trigger
    # a CUDA device-side assert in compute_token_log_probs (.gather OOB).
    bad_words = None
    _cfg = getattr(gen_model, 'config', None)
    _blocked = []
    if _cfg is not None:
        for _attr in ('video_token_id', 'image_token_id'):
            _tid = getattr(_cfg, _attr, None)
            if _tid is not None:
                _blocked.append([_tid])
    if _blocked:
        bad_words = _blocked

    rollout_temp = getattr(args, 'rollout_temperature', 1.0)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      do_sample=(rollout_temp > 0),
                      temperature=rollout_temp if rollout_temp > 0 else 1.0,
                      top_p=1.0, top_k=50,
                      early_stopping=False,
                      bad_words_ids=bad_words)
    sequential = getattr(args, 'sequential_generation', False)
    device = next(model.parameters()).device

    # Gradient checkpointing causes transformers to override use_cache=False in
    # generate().  With use_cache=False, Qwen2VL's prepare_inputs_for_generation
    # sets pixel_values_videos=None for positions > 0, making the model blind to
    # the image after the first token.  Disable GC for the generate() call and
    # re-enable it before the backward pass.
    had_gc = getattr(gen_model, 'is_gradient_checkpointing', False)
    if had_gc:
        gen_model.gradient_checkpointing_disable()

    try:
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
                if hasattr(gen_model, 'rope_deltas'):
                    gen_model.rope_deltas = None
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
                  # Reset cached rope_deltas so Qwen2VL recomputes from scratch
                  # for each generate() call (stale values from prior examples
                  # cause wrong positional encodings).
                  if hasattr(gen_model, 'rope_deltas'):
                      gen_model.rope_deltas = None
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

    finally:
        # Re-enable gradient checkpointing for the subsequent backward pass.
        if had_gc:
            gen_model.gradient_checkpointing_enable()


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
    'train/adv_pos_frac':      float('nan'),
    'train/neg_rew_loss_frac':   float('nan'),
    'train/loss_contrib_neg_rew': float('nan'),
    'train/loss_contrib_pos_rew': float('nan'),
    'train/kl_q_pp':           float('nan'),
    'train/kl_q_pn':         float('nan'),
    'train/kl_q_np':         float('nan'),
    'train/kl_q_nn':         float('nan'),
    '_ratio_list':           [],
    '_adv_list':             [],
}


def cppo_step(model, optimizer, items, processor, args,
              compute_diag: bool = True,
              step: int = 0,
              debug_rollouts: bool = False) -> dict:
    """Dr. CPPO / GRPO update step over a batch of B prompts.

    Supports mini-batch gradient accumulation when mini_batch_size < batch_size.
    B prompts are chunked into ceil(B/mini_batch_size) mini-batches; each
    mini-batch generates G rollouts independently, then all rewards are computed
    in one parallel call, and gradients are accumulated across mini-batches
    before a single optimizer.step() per batch_update epoch.

    With mini_batch_size == batch_size (or unset), behaviour is identical to
    the original single-mini-batch implementation.

    Key properties:
      • Per-prompt group-relative advantages: adv_i = reward_i − mean(group)
      • Per-prompt row-wise top-N selection (not global across prompts)
      • Advantages are per-SEQUENCE scalars [M*N, 1] broadcast over tokens
      • Degenerate mini-batches (all advantages ≈ 0) are skipped
      • Gradient normalised by n_nondegen_mb so loss scale is consistent
    """
    if isinstance(items, dict):
        items = [items]

    B      = len(items)
    G      = args.G
    N      = min(args.top_N, G)
    device = next(model.parameters()).device
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    entropy_coef = float(getattr(args, 'entropy_coef', 0.0))

    mini_batch_size = int(getattr(args, 'mini_batch_size', B))
    mini_batch_size = min(max(1, mini_batch_size), B)
    mb_items_list   = [items[i:i + mini_batch_size]
                       for i in range(0, B, mini_batch_size)]

    # ------------------------------------------------------------------
    # Phase 1: Generate rollouts for each mini-batch (keeps VRAM flat at
    # [mini_batch_size*G, full_len] peak), then compute ALL rewards in one
    # parallel call to maximise worker utilisation.
    # ------------------------------------------------------------------
    all_code_strings: list = []
    all_gt_paths:     list = []
    mb_gen_data:      list = []   # (mb_batch, mb_gen_ids, prompt_len, M)

    t_gen = time.perf_counter()
    for mb_items in mb_items_list:
        M = len(mb_items)
        collate_items = [{k: v for k, v in it.items() if not k.startswith('_')}
                         for it in mb_items]
        mb_batch   = collate(collate_items, processor=processor, n_points=256, eval=True)
        prompt_len = mb_batch['input_ids'].shape[1]
        mb_gen_ids = generate_rollouts(                      # [M*G, full_len]
            model, {k: mb_batch.get(k) for k in _GEN_INPUT_KEYS},
            G, args, pad_id, processor)
        mb_codes = [
            processor.decode(mb_gen_ids[i, prompt_len:], skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)
            for i in range(M * G)
        ]
        mb_gt = [it['gt_mesh_path'] for it in mb_items for _ in range(G)]
        all_code_strings.extend(mb_codes)
        all_gt_paths.extend(mb_gt)
        mb_gen_data.append((mb_batch, mb_gen_ids, prompt_len, M))
    gen_seconds = time.perf_counter() - t_gen

    # avg_gen_len across all B*G rollouts (before top-N selection)
    total_gen_len = 0.0
    for _, mb_gen_ids, prompt_len, M in mb_gen_data:
        _comp = mb_gen_ids[:, prompt_len:]
        _mask = create_completion_mask(_comp, eos_id)
        total_gen_len += _mask.sum().item()
    avg_gen_len = total_gen_len / max(1, B * G)

    t_rew = time.perf_counter()
    all_rewards = compute_rewards_parallel(all_code_strings, all_gt_paths,
                                           workers=args.reward_workers,
                                           soft_invalid=float(getattr(args, 'soft_invalid_reward', -1.0)))
    rew_seconds = time.perf_counter() - t_rew

    # ------------------------------------------------------------------
    # Build per-mini-batch PPO tensors: advantages, old_lp, masks.
    # ------------------------------------------------------------------
    all_rewards_t_list: list = []   # [M, G] per mini-batch (for metrics)
    all_std_r_list:     list = []   # [M]    per mini-batch
    mb_ppo_list:        list = []   # (sel_ids, full_attn, comp_mask, advantages,
                                    #  old_lp, sel_g_batch, logits_to_keep) or None

    topN_neg_count  = 0   # # top-N selected sequences with reward < 0
    topN_total      = 0   # total top-N selected sequences
    topN_rews_list: list = []   # [M, N] per non-degenerate mini-batch (for prompt-level metrics)

    rew_offset = 0
    for mb_batch, mb_gen_ids, prompt_len, M in mb_gen_data:
        mb_rews   = all_rewards[rew_offset:rew_offset + M * G]
        rew_offset += M * G

        rewards_t_mb = torch.tensor(mb_rews, dtype=torch.float32)
        rewards_t_mb = torch.nan_to_num(
            rewards_t_mb, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        rewards_t_mb = rewards_t_mb.view(M, G)                   # [M, G]

        mean_r   = rewards_t_mb.mean(dim=1, keepdim=True)        # [M, 1]
        std_r_mb = rewards_t_mb.std(dim=1)                       # [M]
        adv_raw  = rewards_t_mb - mean_r                         # [M, G]
        if getattr(args, 'reward_normalization', False):
            norm_eps = getattr(args, 'reward_norm_eps', 0.01)
            adv_raw = adv_raw / (std_r_mb.unsqueeze(1) + norm_eps)
        adv_raw  = torch.nan_to_num(adv_raw, nan=0.0, posinf=0.0, neginf=0.0)

        all_rewards_t_list.append(rewards_t_mb)
        all_std_r_list.append(std_r_mb)

        if adv_raw.abs().max().item() < 1e-6:
            mb_ppo_list.append(None)   # degenerate mini-batch — skip
            continue

        # Row-wise top-N per prompt by |advantage|
        _, top_idx = torch.topk(adv_raw.abs(), N, dim=1)                   # [M, N]
        flat_idx   = (torch.arange(M).unsqueeze(1) * G + top_idx).reshape(-1)  # [M*N]

        # Track top-N reward distribution for topN_neg_frac and prompt-level metrics
        sel_rews_mb = rewards_t_mb.reshape(-1)[flat_idx]   # [M*N]
        topN_neg_count += (sel_rews_mb < 0).sum().item()
        topN_total     += sel_rews_mb.numel()
        topN_rews_list.append(sel_rews_mb.view(M, N))      # [M, N]

        sel_ids_cpu        = mb_gen_ids[flat_idx]                           # [M*N, full_len]
        comp_ids_cpu       = sel_ids_cpu[:, prompt_len:]                    # [M*N, T]
        logits_to_keep     = comp_ids_cpu.shape[1]
        comp_mask_cpu      = create_completion_mask(comp_ids_cpu, eos_id)   # [M*N, T]
        g_batch_cpu        = expand_batch(mb_batch, G)                      # [M*G, ...]
        sel_g_batch_cpu    = slice_batch(g_batch_cpu, flat_idx)             # [M*N, ...]
        prompt_mask_cpu    = sel_g_batch_cpu['attention_mask']
        full_attn_mask_cpu = torch.cat(
            [prompt_mask_cpu, comp_mask_cpu.long()], dim=1)                 # [M*N, full_len]
        adv_sel = adv_raw.reshape(-1)[flat_idx].unsqueeze(1)                # [M*N, 1]

        sel_ids     = sel_ids_cpu.to(device)
        full_attn   = full_attn_mask_cpu.to(device)
        comp_mask   = comp_mask_cpu.to(device)
        advantages  = adv_sel.to(device)
        sel_g_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in sel_g_batch_cpu.items()}

        with torch.no_grad():
            old_out = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
            old_lp  = compute_token_log_probs(
                old_out.logits, sel_ids, logits_to_keep).detach()           # [M*N, T]
        del old_out  # logits [B, L, V] not needed after log_probs computed

        mb_ppo_list.append(
            (sel_ids, full_attn, comp_mask, advantages, old_lp, sel_g_batch, logits_to_keep,
             sel_rews_mb))   # sel_rews_mb [M*N]: raw reward per selected sequence

    # Aggregate reward tensors for global metrics
    all_rewards_t = torch.cat(all_rewards_t_list, dim=0)   # [B, G]
    all_std_r     = torch.cat(all_std_r_list,     dim=0)   # [B]

    # Reward-distribution metrics (computed once, reused in both early-exit and normal return)
    _flat_rews    = all_rewards_t.reshape(-1)                           # [B*G]
    failure_rate  = (_flat_rews < 0).float().mean().item()
    topN_neg_frac = topN_neg_count / max(1, topN_total)

    # Prompt-level metrics based on top-N selected sequences (non-degenerate prompts only).
    # topN_rews: [B_nd, N] — rewards of the N sequences that entered gradient update.
    if topN_rews_list:
        topN_rews            = torch.cat(topN_rews_list, dim=0)           # [B_nd, N]
        _topN_pos            = topN_rews > 0                              # [B_nd, N]
        fail_prompt_frac     = (topN_rews < 0).all(dim=1).float().mean().item()
        prompt_all_pos_frac  = _topN_pos.all(dim=1).float().mean().item()
        prompt_geq_half_pos  = (_topN_pos.sum(dim=1) > N / 2).float().mean().item()
    else:
        fail_prompt_frac    = float('nan')
        prompt_all_pos_frac = float('nan')
        prompt_geq_half_pos = float('nan')

    _reward_dist_metrics = {
        'train/failure_rate':          failure_rate,
        'train/topN_neg_frac':         topN_neg_frac,
        'train/fail_prompt_frac':      fail_prompt_frac,
        'train/prompt_all_pos_frac':   prompt_all_pos_frac,
        'train/prompt_geq_half_pos':   prompt_geq_half_pos,
    }

    # ---- Debug: print per-mini-batch summary --------------------------------
    if debug_rollouts:
        rew_off = 0
        print(f'\n{"="*70}')
        print(f'[DEBUG step={step}] B={B} G={G} mini_batch_size={mini_batch_size}')
        print(f'{"="*70}')
        for mbi, (mb_batch, mb_gen_ids, prompt_len, M) in enumerate(mb_gen_data):
            mb_rews = all_rewards[rew_off:rew_off + M * G]
            rew_off += M * G
            rmat = torch.tensor(mb_rews).view(M, G)
            for bi in range(M):
                row = rmat[bi].tolist()
                print(f'  [mb{mbi} p{bi}]: {[f"{r:+.3f}" for r in row]}  '
                      f'mean={rmat[bi].mean():.3f}  std={rmat[bi].std():.3f}')
        print(f'{"="*70}\n')
    # -------------------------------------------------------------------------

    # k=0 entropy: token-count-weighted average across non-degenerate mini-batches
    first_entropy = float('nan')
    _ent_num = 0.0
    _ent_den = 0.0
    for mb_ppo in mb_ppo_list:
        if mb_ppo is None:
            continue
        _, _, comp_mask, _, old_lp, _, _, _ = mb_ppo
        _ent_num += (-(old_lp * comp_mask).sum().item())
        _ent_den += comp_mask.sum().item()
    if _ent_den > 0:
        first_entropy = _ent_num / _ent_den

    n_nondegen_mb = sum(1 for mb in mb_ppo_list if mb is not None)
    # Warn if mini-batches are unequal (last batch smaller) — normalising by count
    # over-weights the smaller batch. Safe when B % mini_batch_size == 0.
    if B % mini_batch_size != 0:
        print(f'[cppo] WARNING: B={B} % mini_batch_size={mini_batch_size} != 0 — '
              f'last mini-batch is smaller; gradient scale will be slightly biased.')
    if n_nondegen_mb == 0:
        return {
            'train/loss':         0.0,
            'train/mean_reward':  all_rewards_t.mean().item(),
            'train/reward_std':   all_std_r.mean().item(),
            'train/reward_max':   all_rewards_t.max().item(),
            'train/reward_min':   all_rewards_t.min().item(),
            'train/entropy':      float('nan'),
            'train/entropy_k0':   float('nan'),
            'train/adv_abs_mean': 0.0,
            **_reward_dist_metrics,
            'train/adv_mean_seq': 0.0,
            'train/adv_mean_tok': 0.0,
            'train/avg_gen_len':  avg_gen_len,
            'train/gen_seconds':  gen_seconds,
            'train/rew_seconds':  rew_seconds,
            'train/grad_seconds': 0.0,
            '_rewards_list':      all_rewards,
            '_reward_std_groups': all_std_r.tolist(),
            **_NAN_DIAG,
        }

    # ------------------------------------------------------------------
    # Phase 2: batch_updates gradient accumulation across mini-batches.
    # For each epoch k: loop over all non-degenerate mini-batches,
    # accumulate scaled gradients, then call optimizer.step() once.
    # Loss is divided by n_nondegen_mb so the effective update magnitude
    # matches a single-mini-batch step (no lr re-tuning needed).
    # ------------------------------------------------------------------
    last_loss    = 0.0
    last_entropy = float('nan')
    last_mb_new_lp_list:  list = []
    last_mb_comp_mask_list: list = []
    last_mb_old_lp_list:  list = []
    last_mb_adv_list:     list = []
    last_mb_seq_loss_list: list = []   # per-sequence surrogate values for neg-rew contribution
    last_mb_sel_rews_list: list = []   # [M*N] raw rewards of selected sequences (last update)

    def _mem(tag, k_):
        if debug_rollouts:
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            print(f'[MEM step={step} k={k_} {tag}] alloc={a:.2f}GB  reserved={r:.2f}GB')

    t_grad = time.perf_counter()
    for k in range(args.batch_updates):
        is_last = (k == args.batch_updates - 1)
        model.train()
        optimizer.zero_grad()
        k_loss = 0.0
        if is_last:
            last_mb_new_lp_list.clear()
            last_mb_comp_mask_list.clear()
            last_mb_old_lp_list.clear()
            last_mb_adv_list.clear()
            last_mb_seq_loss_list.clear()
            last_mb_sel_rews_list.clear()

        for mb_ppo in mb_ppo_list:
            if mb_ppo is None:
                continue
            sel_ids, full_attn, comp_mask, advantages, old_lp, sel_g_batch, logits_to_keep, sel_rews = mb_ppo
            _mem('pre-fwd', k)
            new_out = model_forward(model, sel_ids, full_attn, sel_g_batch, device)
            _mem('post-fwd', k)
            new_lp  = compute_token_log_probs(new_out.logits, sel_ids, logits_to_keep)
            del new_out  # logits [B, L, V] not needed for backward (log_softmax saves output)
            _mem('post-log_probs', k)

            loss, seq_loss_det = cppo_loss_fn(new_lp, old_lp, advantages, comp_mask,
                                              args.eps_high, args.eps_low)
            if entropy_coef > 0:
                step_entropy = compute_policy_entropy(new_lp, comp_mask)
                loss = loss - entropy_coef * step_entropy

            # Normalise so accumulated gradient = single-mini-batch gradient
            loss = loss / n_nondegen_mb
            _mem('pre-backward', k)
            loss.backward()
            _mem('post-backward', k)
            k_loss += loss.item()

            if is_last:
                last_mb_new_lp_list.append(new_lp.detach())
                last_mb_comp_mask_list.append(comp_mask)
                last_mb_old_lp_list.append(old_lp)
                last_mb_adv_list.append(advantages)
                last_mb_seq_loss_list.append(seq_loss_det)  # [M*N] surrogate values
                last_mb_sel_rews_list.append(sel_rews)      # [M*N] raw rewards

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        _mem('post-opt-step', k)

        if is_last:
            last_loss = k_loss
            if compute_diag and last_mb_new_lp_list:
                _en = 0.0; _ed = 0.0
                for nlp, cm in zip(last_mb_new_lp_list, last_mb_comp_mask_list):
                    _en += (-(nlp * cm).sum().item())
                    _ed += cm.sum().item()
                last_entropy = _en / max(1.0, _ed)

    if debug_rollouts:
        print(f'[DEBUG step={step}] entropy_k0={first_entropy:.4f}  '
              f'entropy_kfinal={last_entropy:.4f}  loss={last_loss:.4f}  '
              f'batch_updates={args.batch_updates}  n_minibatches={n_nondegen_mb}\n')

    grad_seconds = time.perf_counter() - t_grad

    # ------------------------------------------------------------------
    # Phase 3: Diagnostics — aggregate token-count-weighted across mini-batches.
    # ------------------------------------------------------------------
    all_adv_cat = torch.cat(last_mb_adv_list, dim=0)   # [B_nd*N, 1]
    with torch.no_grad():
        if compute_diag and last_mb_new_lp_list:
            _total_tok   = 0.0
            _clip_num    = 0.0; _clip_lo_num = 0.0; _clip_hi_num = 0.0
            _kl_sum_seq  = 0.0; _kl_n_seq    = 0.0
            _kl_seq_list:   list = []
            _ratio_seq_list: list = []
            _adv_seq_list:  list = []

            for nlp, olp, cm, adv_mb in zip(last_mb_new_lp_list, last_mb_old_lp_list,
                                             last_mb_comp_mask_list, last_mb_adv_list):
                ratio_tok = torch.exp(nlp - olp)                          # [M*N, T]
                ratio_seq = (ratio_tok * cm).sum(1) / cm.sum(1).clamp(min=1)  # [M*N]
                n_tok     = cm.sum().clamp(min=1)
                clip_lo   = ratio_tok < 1 - args.eps_low
                clip_hi   = ratio_tok > 1 + args.eps_high
                _total_tok   += n_tok.item()
                _clip_num    += ((clip_lo | clip_hi).float() * cm).sum().item()
                _clip_lo_num += (clip_lo.float() * cm).sum().item()
                _clip_hi_num += (clip_hi.float() * cm).sum().item()
                kl_tok     = (ratio_tok - 1 - torch.log(ratio_tok.clamp(min=1e-8))) * cm
                kl_per_seq = kl_tok.sum(dim=1)                            # [M*N]
                _kl_sum_seq += kl_per_seq.sum().item()
                _kl_n_seq   += kl_per_seq.shape[0]
                _kl_seq_list.append(kl_per_seq)
                _ratio_seq_list.append(ratio_seq)
                _adv_seq_list.append(adv_mb.squeeze(1))

            kl_seq_all   = torch.cat(_kl_seq_list,    dim=0)   # [B_nd*N]
            ratio_seq_all = torch.cat(_ratio_seq_list, dim=0)  # [B_nd*N]
            adv_seq_all  = torch.cat(_adv_seq_list,   dim=0)   # [B_nd*N]
            kl_total     = kl_seq_all.sum().clamp(min=1e-8)

            mean_seq_len = _total_tok / max(1.0, _kl_n_seq)
            kl_approx    = (_kl_sum_seq / max(1.0, _kl_n_seq)) / max(1.0, mean_seq_len)

            # neg_rew_loss_frac: fraction of total positive loss contribution
            # that comes from sequences with raw reward < 0.
            # seq_loss[i] < 0  → contributes positively to -seq_loss.mean() (loss goes up).
            # neg_rew_loss_frac = sum(-seq[rew<0]) / sum(-seq.clamp(max=0))
            seq_loss_all = torch.cat(last_mb_seq_loss_list, dim=0)   # [B_nd*N] CUDA
            rew_seq_all  = torch.cat(last_mb_sel_rews_list,  dim=0).to(seq_loss_all.device)  # [B_nd*N]
            neg_rew_mask = rew_seq_all < 0
            pos_rew_mask = ~neg_rew_mask
            _pos_loss_total   = (-seq_loss_all.clamp(max=0)).sum().item()
            _neg_rew_contrib  = (-seq_loss_all[neg_rew_mask].clamp(max=0)).sum().item()
            neg_rew_loss_frac = _neg_rew_contrib / max(1e-8, _pos_loss_total)
            # Per-category loss contribution weighted by count fraction so that
            # loss_contrib_neg_rew + loss_contrib_pos_rew == train/loss exactly.
            loss_contrib_neg_rew = (-seq_loss_all * neg_rew_mask.float()).mean().item()
            loss_contrib_pos_rew = (-seq_loss_all * pos_rew_mask.float()).mean().item()

            diag = {
                'train/clip_fraction':   _clip_num    / max(1.0, _total_tok),
                'train/clip_lower_frac': _clip_lo_num / max(1.0, _total_tok),
                'train/clip_upper_frac': _clip_hi_num / max(1.0, _total_tok),
                'train/ratio_mean':      ratio_seq_all.mean().item(),
                'train/ratio_std':       ratio_seq_all.std().item(),
                'train/kl_approx':       kl_approx,
                'train/adv_pos_frac':    (all_adv_cat > 0).float().mean().item(),
                'train/neg_rew_loss_frac':   neg_rew_loss_frac,
                'train/loss_contrib_neg_rew': loss_contrib_neg_rew,
                'train/loss_contrib_pos_rew': loss_contrib_pos_rew,
                'train/kl_q_pp': kl_seq_all[(adv_seq_all >  0) & (ratio_seq_all >  1)].sum().item() / kl_total.item(),
                'train/kl_q_pn': kl_seq_all[(adv_seq_all >  0) & (ratio_seq_all <= 1)].sum().item() / kl_total.item(),
                'train/kl_q_np': kl_seq_all[(adv_seq_all <= 0) & (ratio_seq_all >  1)].sum().item() / kl_total.item(),
                'train/kl_q_nn': kl_seq_all[(adv_seq_all <= 0) & (ratio_seq_all <= 1)].sum().item() / kl_total.item(),
                '_ratio_list':   ratio_seq_all.cpu().tolist(),
                '_adv_list':     adv_seq_all.cpu().tolist(),
            }
        else:
            diag = dict(_NAN_DIAG)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    _n_tok = sum(cm.sum().item() for cm in last_mb_comp_mask_list) or 1.0
    _adv_tok_sum = sum(
        (adv_mb * cm).sum().item()
        for adv_mb, cm in zip(last_mb_adv_list, last_mb_comp_mask_list)
    )

    return {
        'train/loss':         last_loss,
        'train/mean_reward':  all_rewards_t.mean().item(),
        'train/reward_std':   all_std_r.mean().item(),
        'train/reward_max':   all_rewards_t.max().item(),
        'train/reward_min':   all_rewards_t.min().item(),
        'train/entropy':      last_entropy,
        'train/entropy_k0':   first_entropy,
        'train/adv_abs_mean': all_adv_cat.abs().mean().item(),
        'train/adv_mean_seq': all_adv_cat.mean().item(),
        'train/adv_mean_tok': _adv_tok_sum / _n_tok,
        'train/avg_gen_len':  avg_gen_len,
        'train/gen_seconds':  gen_seconds,
        'train/rew_seconds':  rew_seconds,
        'train/grad_seconds': grad_seconds,
        '_rewards_list':      all_rewards,
        '_reward_std_groups': all_std_r.tolist(),
        **_reward_dist_metrics,
        **diag,
    }


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _safe_histogram(data):
    """wandb.Histogram crashes when all values are identical (zero-range).
    Fall back to a plain scalar (mean) in that case."""
    import wandb, numpy as np
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0 or np.all(arr == arr[0]):
        return float(arr[0]) if arr.size > 0 else 0.0
    try:
        return wandb.Histogram(arr)
    except Exception:
        return float(arr.mean())


# Checkpoint rotation
# ---------------------------------------------------------------------------

def _rotate_checkpoints(output_dir: str, save_total_limit: Optional[int]):
    """Delete oldest checkpoint-XXXXX dirs when limit is exceeded.

    Skips checkpoints that still have an active HF upload thread — deleting
    a checkpoint directory while upload_folder() is reading it corrupts the
    upload.  The directory will be cleaned up on the next rotation call once
    the upload thread has finished.
    """
    if not save_total_limit or save_total_limit <= 0:
        return
    # Names of checkpoints currently being uploaded (thread name = hf-upload-<name>)
    uploading = {t.name.removeprefix('hf-upload-')
                 for t in _bg_upload_threads if t.is_alive()}
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith('checkpoint-'):
            try:
                step = int(name[len('checkpoint-'):])
                checkpoints.append((step, name, os.path.join(output_dir, name)))
            except ValueError:
                pass
    checkpoints.sort()
    for _, name, path in checkpoints[:-save_total_limit]:
        if name in uploading:
            print(f'[checkpoint] skipping deletion of {name} (upload in progress)')
            continue
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
    seed = int(getattr(args, 'seed', 42))
    rng = np.random.RandomState(seed)   # seeded RNG for data order — reproducible per run
    torch.manual_seed(seed)
    if rank == 0:
        print(f'RNG seed: {seed}')
    if step > 0 and rank == 0:
        print(f'Resuming from step {step}')
    indices = list(range(len(dataset)))

    # Resume RNG + epoch position so data order is exactly reproducible.
    _resume_rank_indices = None   # saved shuffled order for the in-progress epoch
    _resume_start_in_epoch = 0   # next batch position within that epoch
    _resume_epoch = 0
    if step > 0:
        rng_path = os.path.join(args.checkpoint_path, 'rng_state.pt')
        if os.path.exists(rng_path):
            rs = torch.load(rng_path, map_location='cpu', weights_only=False)
            rng.set_state(rs['numpy_rng'])
            torch.set_rng_state(rs['torch_rng'])
            if rs.get('cuda_rng') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rs['cuda_rng'])
            _resume_epoch          = rs.get('epoch', 0)
            _resume_start_in_epoch = rs.get('next_start', 0)
            _resume_rank_indices   = rs.get('rank_indices', None)
            if rank == 0:
                print(f'[resume] RNG state restored — epoch={_resume_epoch}, '
                      f'pos={_resume_start_in_epoch}')
        elif rank == 0:
            print('[resume] WARNING: no rng_state.pt — data order will differ from '
                  'original run (optimizer state is still restored correctly)')
    batch_size = max(1, int(getattr(args, 'batch_size', 1)))
    use_buffer = bool(getattr(args, 'use_buffer', False))
    buffer_sample_size = int(getattr(args, 'buffer_sample_size', batch_size))
    buffer_expand_frac = float(getattr(args, 'buffer_expand_frac', 0.5))
    replay_buffer = (
        IndexBuffer(max_size=int(getattr(args, 'buffer_max_size', 4096)))
        if use_buffer else None
    )

    if val_examples and rank == 0:
        print(f'\n[eval step={step} (pre-training baseline)]')
        try:
            raw_model = model.module if hasattr(model, 'module') else model
            val_metrics = run_validation(raw_model, val_examples, processor, args)
            log_eval(val_metrics, step=step, log_path=log_path, use_wandb=use_wandb)
        except Exception as e:
            print(f'[eval step={step}] failed (skipping): {e}')
        model.train()
        import gc as _gc; _gc.collect()   # drop Python circular refs before CUDA flush
        torch.cuda.empty_cache()          # flush fragmentation left by step-0 eval

    # DistributedSampler gives each rank a disjoint shard of the dataset.
    # set_epoch() re-shuffles with a different seed each epoch.
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    pbar = tqdm(total=args.max_steps, initial=step, desc='Dr. CPPO', disable=(rank != 0))
    last_entropy = float('nan')
    epoch = _resume_epoch
    _first_epoch = (_resume_rank_indices is not None)  # use saved indices for first epoch
    while step < args.max_steps:
        if is_distributed:
            sampler.set_epoch(epoch)
            rank_indices = list(sampler)
            _start_in_epoch = 0
        elif _first_epoch:
            rank_indices    = _resume_rank_indices
            _start_in_epoch = _resume_start_in_epoch
            _first_epoch    = False
        else:
            rng.shuffle(indices)
            rank_indices    = indices
            _start_in_epoch = 0
        epoch += 1
        # Curriculum: update active pool at epoch boundaries (cheap operation).
        if hasattr(dataset, 'set_step'):
            phase = dataset.set_step(step)
            # Rebuild indices for the (possibly expanded) pool each epoch.
            indices = list(range(len(dataset)))
            if not is_distributed:
                rank_indices = indices
                rng.shuffle(rank_indices)
                _start_in_epoch = 0
            if rank == 0:
                print(f'[curriculum] step={step} phase={phase} pool={len(dataset)}')
        for start in range(_start_in_epoch, len(rank_indices), batch_size):
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
                debug_steps = int(getattr(args, 'debug_rollout_steps', 0))
                metrics = cppo_step(model, optimizer, batch_items, processor, args,
                                    compute_diag=compute_diag,
                                    step=step,
                                    debug_rollouts=(rank == 0 and step < debug_steps))
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
                msg = str(e)
                # Fatal CUDA allocator corruption — cannot recover, must restart.
                # Happens when a previous OOM leaves the allocator in a bad state.
                if 'INTERNAL ASSERT FAILED' in msg or 'handles_.at' in msg:
                    print(f'[step {step}] FATAL CUDA allocator error — restarting from last checkpoint is required.\n{e}')
                    raise
                print(f'[step {step}] cppo_step error: {e}')
                torch.cuda.empty_cache()
                continue

            step += 1
            pbar.update(1)
            e = metrics['train/entropy']
            if not (e != e):  # update only when not nan
                last_entropy = e
                # Early-step sanity check: healthy entropy is ~0.1–1.5.
                # Entropy > 5 at step ≤ 5 almost certainly means the GC bug
                # (model blind to image during generate → garbage tokens).
                if step <= 5 and e > 5.0 and rank == 0:
                    print(f'\n[WARNING] step={step} entropy={e:.2f} > 5.0 — '
                          f'possible gradient-checkpointing + use_cache bug. '
                          f'Check that gradient_checkpointing_disable() is being '
                          f'called before model.generate() in generate_rollouts().')
            pbar.set_postfix(
                loss=f"{metrics['train/loss']:.3f}",
                reward=f"{metrics['train/mean_reward']:.2f}",
                H=f"{last_entropy:.2f}" if not (last_entropy != last_entropy) else "...")

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
                    f" train/entropy_k0={metrics['train/entropy_k0']:.4f}"
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
                    # W&B custom expressions → docs/wandb_expressions.md
                    wandb.log({
                        # ── Core ──────────────────────────────────────────────
                        'loss':           metrics['train/loss'],           # = loss_contrib_neg_rew + loss_contrib_pos_rew
                        'average_reward': metrics['train/mean_reward'],
                        'train/reward_std':      metrics['train/reward_std'],
                        'train/reward_max':      metrics['train/reward_max'],
                        'train/reward_min':      metrics['train/reward_min'],
                        # ── Policy entropy ────────────────────────────────────
                        'train/entropy':         metrics['train/entropy'],   # H after last batch_update
                        'train/entropy_k0':      metrics['train/entropy_k0'], # H before any gradient update
                        # ── KL & ratio ────────────────────────────────────────
                        'train/kl_approx':       metrics['train/kl_approx'],
                        'train/clip_fraction':   metrics['train/clip_fraction'],
                        'train/clip_lower_frac': metrics['train/clip_lower_frac'],
                        'train/clip_upper_frac': metrics['train/clip_upper_frac'],
                        'train/ratio_mean':      metrics['train/ratio_mean'],
                        'train/ratio_std':       metrics['train/ratio_std'],
                        # ── KL quadrants (fraction of total KL mass) ──────────
                        # expr: kl_q_pp+kl_q_nn = healthy frac; kl_q_np+kl_q_pn = unhealthy frac
                        # collapse signal: kl_q_np → 1
                        'train/kl_q_pp':         metrics['train/kl_q_pp'],  # adv>0 & ratio>1 ✓
                        'train/kl_q_pn':         metrics['train/kl_q_pn'],  # adv>0 & ratio<1 ✗
                        'train/kl_q_np':         metrics['train/kl_q_np'],  # adv<0 & ratio>1 ✗ collapse
                        'train/kl_q_nn':         metrics['train/kl_q_nn'],  # adv<0 & ratio<1 ✓
                        # ── Advantage ─────────────────────────────────────────
                        'train/adv_pos_frac':    metrics['train/adv_pos_frac'],
                        'train/adv_abs_mean':    metrics['train/adv_abs_mean'],
                        'train/adv_mean_seq':    metrics['train/adv_mean_seq'],
                        'train/adv_mean_tok':    metrics['train/adv_mean_tok'],
                        # ── Loss contribution (neg+pos = loss) ────────────────
                        # expr: neg / (0 - pos) = penalty_reward_ratio (>1 bad seqs dominate)
                        'train/loss_contrib_neg_rew': metrics['train/loss_contrib_neg_rew'],  # >0 pushes loss up
                        'train/loss_contrib_pos_rew': metrics['train/loss_contrib_pos_rew'],  # <0 pulls loss down
                        'train/neg_rew_loss_frac':    metrics['train/neg_rew_loss_frac'],     # of loss-increasing seqs, frac with rew<0
                        # ── Rollout distribution (top-N based) ────────────────
                        # expr: 1-failure_rate = success_rate
                        # expr: prompt_all_pos_frac / (1-fail_prompt_frac) = learnable_prompt_all_pos
                        'train/failure_rate':         metrics['train/failure_rate'],          # rew<0 / B×G
                        'train/topN_neg_frac':        metrics['train/topN_neg_frac'],         # rew<0 / B_nd×N
                        'train/fail_prompt_frac':     metrics['train/fail_prompt_frac'],      # all-fail prompts / B
                        'train/prompt_all_pos_frac':  metrics['train/prompt_all_pos_frac'],   # all-pass prompts / B
                        'train/prompt_geq_half_pos':  metrics['train/prompt_geq_half_pos'],   # >N/2 pass prompts / B
                        # ── Timing ────────────────────────────────────────────
                        # expr: gen+rew+grad = total_step_seconds
                        'train/avg_gen_len':     metrics['train/avg_gen_len'],
                        'train/gen_seconds':     metrics['train/gen_seconds'],
                        'train/rew_seconds':     metrics['train/rew_seconds'],
                        'train/grad_seconds':    metrics['train/grad_seconds'],
                        'train/pool_crashes':    get_and_reset_pool_crashes(),
                        'train/lr':              lr,
                        # ── Distributions ─────────────────────────────────────
                        'dist/rewards': _safe_histogram(metrics['_rewards_list']),
                        'dist/ratios':  _safe_histogram(metrics['_ratio_list']),
                        'dist/advs':    _safe_histogram(metrics['_adv_list']),
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
                import gc as _gc; _gc.collect()
                torch.cuda.empty_cache()

            if step % args.save_steps == 0 and rank == 0:
                ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                save_model = model.module if hasattr(model, 'module') else model
                save_model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_dir, 'optimizer.pt'))
                # Save RNG state so any resume is exactly reproducible
                torch.save({
                    'numpy_rng':   rng.get_state(),
                    'torch_rng':   torch.get_rng_state(),
                    'cuda_rng':    torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'epoch':       epoch - 1,        # 0-indexed epoch whose indices are active
                    'next_start':  start + batch_size,  # first unprocessed position in epoch
                    'rank_indices': rank_indices,    # this epoch's shuffled data order
                }, os.path.join(ckpt_dir, 'rng_state.pt'))
                _rotate_checkpoints(args.output_dir,
                                    getattr(args, 'save_total_limit', None))
                # Async upload to HuggingFace (non-blocking)
                hf_repo = getattr(args, 'hf_upload_repo', None)
                if hf_repo:
                    _upload_checkpoint_async(ckpt_dir, hf_repo)

    pbar.close()
    if rank == 0:
        final_dir = os.path.join(args.output_dir, 'checkpoint-final')
        save_model = model.module if hasattr(model, 'module') else model
        save_model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        torch.save(optimizer.state_dict(), os.path.join(final_dir, 'optimizer.pt'))
        print(f'Training complete. Final checkpoint → {final_dir}')
        # Upload final checkpoint and wait for all in-flight uploads
        hf_repo = getattr(args, 'hf_upload_repo', None)
        if hf_repo:
            _upload_checkpoint_async(final_dir, hf_repo)
        live = [t for t in _bg_upload_threads if t.is_alive()]
        if live:
            print(f'[HF] Waiting for {len(live)} upload(s) to finish...')
            for t in live:
                t.join(timeout=600)
