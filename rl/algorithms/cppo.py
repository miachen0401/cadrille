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

    # Block video tokens — model must not emit them in CadQuery code
    bad_words = None
    if hasattr(model, 'config') and hasattr(model.config, 'video_token_id'):
        bad_words = [[model.config.video_token_id]]

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
                out = model.generate(**expanded, **gen_kwargs)
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
                ids = model.generate(**one, **gen_kwargs)
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

    Design for B > 1 (multi-GPU cluster alignment):
      Phase 1  — Per-prompt: generate G rollouts, compute rewards + old_lp.
                 Each prompt is processed independently so variable prompt
                 lengths never require cross-prompt padding.
      Phase 2  — batch_updates inner PPO steps with gradient accumulation:
                 zero_grad once → backward(loss/B_valid) for each valid
                 prompt → clip → optimizer.step() once per inner step.
                 Effective batch = B_valid × N sequences per gradient update.
      Phase 3  — KL-quadrant diagnostics aggregated over all valid prompts,
                 using LAST inner iteration's IS only (paper Phase-3 convention).

    With B=1 (default) the behaviour is identical to the single-prompt version.

    Key properties:
      • Per-prompt group-relative advantages: adv_i = reward_i − mean(group)
      • Per-prompt row-wise top-N selection (not global)
      • Advantages are per-SEQUENCE scalars [N,1] broadcast over tokens —
        no token-level advantage in CPPO/GRPO
      • Degenerate prompts (all G rewards identical) are skipped individually
    """
    if isinstance(items, dict):
        items = [items]

    device = next(model.parameters()).device
    G      = args.G
    N      = min(args.top_N, G)
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Phase 1: Per-prompt rollouts, rewards, and old log-probs.
    # Generate G completions per prompt independently (one call each) so
    # each prompt keeps its own prompt_len — no cross-prompt padding needed.
    # Old log-probs are all computed BEFORE any gradient update.
    # ------------------------------------------------------------------
    t_gen = time.perf_counter()

    prompt_data   = []   # one entry per valid (non-degenerate) prompt
    all_rewards   = []   # flat list for global reward stats
    std_r_groups  = []   # per-prompt std_r for replay-buffer logic

    for it in items:
        ci  = {k: v for k, v in it.items() if not k.startswith('_')}
        bat = collate([ci], processor=processor, n_points=256, eval=True)
        pl  = bat['input_ids'].shape[1]   # this prompt's (padded) length
        gt  = it['gt_mesh_path']

        single = {k: bat.get(k) for k in _GEN_INPUT_KEYS}
        gen    = generate_rollouts(model, single, G, args, pad_id, processor)
        # gen: [G, full_len_i]  — G completions for this single prompt

        codes   = [processor.decode(gen[g, pl:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for g in range(G)]
        rewards = compute_rewards_parallel(codes, [gt] * G,
                                           workers=args.reward_workers)
        all_rewards.extend(rewards)

        r_t    = torch.tensor(rewards, dtype=torch.float32)   # [G]
        mean_r = r_t.mean()
        std_r  = r_t.std()
        std_r_groups.append(std_r.item())

        # Skip degenerate prompt: all G rewards identical → zero gradient
        if std_r.item() < 1e-6:
            continue

        adv_raw = r_t - mean_r                              # [G]  no std-norm
        _, top_idx = torch.topk(adv_raw.abs(), N)           # [N]

        sel_ids_cpu  = gen[top_idx]                         # [N, full_len_i]
        comp_ids_cpu = sel_ids_cpu[:, pl:]                  # [N, T_i]
        logits_to_keep = comp_ids_cpu.shape[1]

        comp_mask_cpu = create_completion_mask(comp_ids_cpu, eos_id)  # [N, T_i]
        g_bat         = expand_batch(bat, G)
        sel_bat_cpu   = slice_batch(g_bat, top_idx)         # [N, ...]

        prompt_mask_cpu    = sel_bat_cpu['attention_mask']  # [N, pl]
        full_attn_mask_cpu = torch.cat(
            [prompt_mask_cpu, comp_mask_cpu.long()], dim=1) # [N, pl+T_i]

        adv_sel = adv_raw[top_idx].unsqueeze(1)             # [N, 1]

        # Compute old log-probs from the current (pre-update) model
        torch.cuda.empty_cache()
        sel_ids      = sel_ids_cpu.to(device)
        full_attn    = full_attn_mask_cpu.to(device)
        comp_mask    = comp_mask_cpu.to(device)
        advantages   = adv_sel.to(device)
        sel_bat      = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in sel_bat_cpu.items()}

        with torch.no_grad():
            old_out = model_forward(model, sel_ids, full_attn, sel_bat, device)
            old_lp  = compute_token_log_probs(
                old_out.logits, sel_ids, logits_to_keep).detach()

        prompt_data.append({
            'sel_ids':       sel_ids,        # [N, full_len_i] — on GPU
            'full_attn':     full_attn,      # [N, pl+T_i]
            'comp_mask':     comp_mask,      # [N, T_i]
            'advantages':    advantages,     # [N, 1]
            'sel_bat':       sel_bat,        # model inputs on GPU
            'old_lp':        old_lp,         # [N, T_i]
            'logits_to_keep': logits_to_keep,
            'mean_r':        mean_r.item(),
            'std_r':         std_r.item(),
        })

    gen_seconds = time.perf_counter() - t_gen
    rewards_all = torch.tensor(all_rewards, dtype=torch.float32)

    if not prompt_data:
        # All B prompts degenerate — no gradient update
        return {
            'train/loss':         0.0,
            'train/mean_reward':  rewards_all.mean().item(),
            'train/reward_std':   rewards_all.std().item(),
            'train/reward_max':   rewards_all.max().item(),
            'train/reward_min':   rewards_all.min().item(),
            'train/entropy':      float('nan'),
            'train/adv_abs_mean': 0.0,
            'train/gen_seconds':  gen_seconds,
            '_rewards_list':      all_rewards,
            '_reward_std_groups': std_r_groups,
            **_NAN_DIAG,
        }

    B_valid = len(prompt_data)

    # ------------------------------------------------------------------
    # Phase 2: batch_updates inner PPO steps with gradient accumulation.
    # optimizer.zero_grad() once at the start of each inner step;
    # loss is divided by B_valid before backward so gradients are averaged
    # (not summed) over prompts — equivalent to processing B_valid*N
    # sequences in one forward pass.
    # optimizer.step() is called once per inner step after all B prompts.
    # ------------------------------------------------------------------
    last_loss    = 0.0
    last_entropy = float('nan')
    last_new_lps = None     # [B_valid] list of new_lp tensors for Phase 3

    for k in range(args.batch_updates):
        is_last = (k == args.batch_updates - 1)
        model.train()
        optimizer.zero_grad()

        step_loss     = 0.0
        step_ent_sum  = 0.0
        step_new_lps  = [] if is_last else None

        for pd in prompt_data:
            new_out = model_forward(model,
                                    pd['sel_ids'], pd['full_attn'],
                                    pd['sel_bat'], device)
            new_lp = compute_token_log_probs(
                new_out.logits, pd['sel_ids'], pd['logits_to_keep'])

            loss = cppo_loss_fn(new_lp, pd['old_lp'], pd['advantages'],
                                pd['comp_mask'], args.eps_high, args.eps_low)

            # Divide by B_valid: gradient accumulation = average over prompts
            (loss / B_valid).backward()
            step_loss += loss.item()

            if is_last:
                with torch.no_grad():
                    step_ent_sum += compute_policy_entropy(
                        new_out.logits, pd['comp_mask'],
                        pd['logits_to_keep']).item()
                step_new_lps.append({
                    'new_lp':    new_lp.detach(),
                    'old_lp':    pd['old_lp'],
                    'comp_mask': pd['comp_mask'],
                    'advantages': pd['advantages'],
                })

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        last_loss = step_loss / B_valid
        if is_last:
            last_entropy = step_ent_sum / B_valid
            last_new_lps = step_new_lps

    # ------------------------------------------------------------------
    # Phase 3: Diagnostics — aggregated over all valid prompts.
    # Uses IS from the LAST inner iteration only (paper Phase-3 convention).
    # Each prompt contributes its own [N, T_i] token-level tensors;
    # we accumulate token counts and KL sums to get global statistics.
    # ------------------------------------------------------------------
    with torch.no_grad():
        if compute_diag and last_new_lps:
            clip_n = clip_lo_n = clip_hi_n = n_tok = 0.0
            kl_total = 0.0
            kl_q_pp = kl_q_pn = kl_q_np = kl_q_nn = 0.0
            kl_approx_sum = 0.0
            ratio_seq_all = []
            adv_all       = []
            adv_pos = adv_cnt = 0

            for lp in last_new_lps:
                new_lp    = lp['new_lp']
                old_lp    = lp['old_lp']
                comp_mask = lp['comp_mask']
                adv_sq    = lp['advantages'].squeeze(1)   # [N]

                ratio_tok = torch.exp(new_lp - old_lp)   # [N, T_i]
                ratio_seq = (ratio_tok * comp_mask).sum(1) \
                            / comp_mask.sum(1).clamp(min=1)

                cl = (ratio_tok < 1 - args.eps_low)
                cu = (ratio_tok > 1 + args.eps_high)
                nt = comp_mask.sum().item()

                clip_n   += ((cl | cu).float() * comp_mask).sum().item()
                clip_lo_n += (cl.float() * comp_mask).sum().item()
                clip_hi_n += (cu.float() * comp_mask).sum().item()
                n_tok    += nt

                kl_tok_i   = (ratio_tok - 1
                               - torch.log(ratio_tok.clamp(min=1e-8))) * comp_mask
                kl_seq     = kl_tok_i.sum(dim=1)          # [N]
                kl_sum     = kl_seq.sum().item()
                kl_total  += kl_sum
                kl_approx_sum += (kl_seq.mean()
                                  / comp_mask.sum(1).mean().clamp(min=1)).item()

                # Quadrant attribution — will normalise by kl_total at the end
                kl_q_pp += kl_seq[(adv_sq > 0)  & (ratio_seq > 1.0)].sum().item()
                kl_q_pn += kl_seq[(adv_sq > 0)  & (ratio_seq <= 1.0)].sum().item()
                kl_q_np += kl_seq[(adv_sq <= 0) & (ratio_seq > 1.0)].sum().item()
                kl_q_nn += kl_seq[(adv_sq <= 0) & (ratio_seq <= 1.0)].sum().item()

                ratio_seq_all.extend(ratio_seq.cpu().tolist())
                adv_all.extend(adv_sq.cpu().tolist())
                adv_pos += (adv_sq > 0).sum().item()
                adv_cnt += adv_sq.numel()

            kl_denom = max(kl_total, 1e-8)
            n_tok    = max(n_tok, 1)
            diag = {
                'train/clip_fraction':   clip_n   / n_tok,
                'train/clip_lower_frac': clip_lo_n / n_tok,
                'train/clip_upper_frac': clip_hi_n / n_tok,
                'train/ratio_mean':      float(np.mean(ratio_seq_all)),
                'train/ratio_std':       float(np.std(ratio_seq_all)),
                'train/kl_approx':       kl_approx_sum / B_valid,
                'train/adv_pos_frac':    adv_pos / max(adv_cnt, 1),
                'train/kl_q_pp':         kl_q_pp / kl_denom,
                'train/kl_q_pn':         kl_q_pn / kl_denom,
                'train/kl_q_np':         kl_q_np / kl_denom,
                'train/kl_q_nn':         kl_q_nn / kl_denom,
                '_ratio_list':           ratio_seq_all,
                '_adv_list':             adv_all,
            }
        else:
            diag = dict(_NAN_DIAG)

    all_adv = torch.cat([pd['advantages'].view(-1) for pd in prompt_data])
    return {
        'train/loss':         last_loss,
        'train/mean_reward':  rewards_all.mean().item(),
        'train/reward_std':   rewards_all.std().item(),
        'train/reward_max':   rewards_all.max().item(),
        'train/reward_min':   rewards_all.min().item(),
        'train/entropy':      last_entropy,
        'train/adv_abs_mean': all_adv.abs().mean().item(),
        'train/gen_seconds':  gen_seconds,
        '_rewards_list':      all_rewards,
        '_reward_std_groups': std_r_groups,
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
    batch_size = max(1, int(getattr(args, 'batch_size', 1)))
    use_buffer = bool(getattr(args, 'use_buffer', False))
    buffer_sample_size = int(getattr(args, 'buffer_sample_size', batch_size))
    buffer_expand_frac = float(getattr(args, 'buffer_expand_frac', 0.5))
    replay_buffer = (
        IndexBuffer(max_size=int(getattr(args, 'buffer_max_size', 4096)))
        if use_buffer else None
    )

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
        for start in range(0, len(indices), batch_size):
            if step >= args.max_steps:
                break
            base_indices = indices[start:start + batch_size]
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
                    f" train/kl_q_np={metrics['train/kl_q_np']:.4f}"
                    f" train/kl_q_nn={metrics['train/kl_q_nn']:.4f}"
                    f" train/kl_q_pp={metrics['train/kl_q_pp']:.4f}"
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
