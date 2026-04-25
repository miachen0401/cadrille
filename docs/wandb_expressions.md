# W&B Custom Expressions

Copy-paste these into W&B UI:
**Add panel → Line Plot → (y-axis) → Add expression**

---

## Loss


| Expression                                                            | Name                   | What it tells you                                                  |
| --------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------ |
| `${train/loss_contrib_neg_rew} + ${train/loss_contrib_pos_rew}`       | `loss_check`           | Should equal `loss` exactly — sanity check                         |
| `${train/loss_contrib_neg_rew} / (0 - ${train/loss_contrib_pos_rew})` | `penalty_reward_ratio` | >1 = bad seqs dominating; <1 = good seqs dominating; ~1 = balanced |


## Reward distribution


| Expression                                                       | Name                       | What it tells you                                         |
| ---------------------------------------------------------------- | -------------------------- | --------------------------------------------------------- |
| `1 - ${train/failure_rate}`                                      | `success_rate`             | Fraction of all B×G rollouts with reward ≥ 0              |
| `${average_reward} / (1 - ${train/failure_rate})`                | `conditional_reward`       | Mean reward given non-failure (reward ≥ 0 sequences only) |
| `${train/prompt_all_pos_frac} / (1 - ${train/fail_prompt_frac})` | `learnable_prompt_all_pos` | Of non-failing prompts, how many are fully successful     |


## KL health


| Expression                            | Name                | What it tells you                                                                                    |
| ------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| `${train/kl_q_pp} + ${train/kl_q_nn}` | `kl_healthy_frac`   | KL mass in correct directions (→1 = healthy)                                                         |
| `${train/kl_q_np} + ${train/kl_q_pn}` | `kl_unhealthy_frac` | KL mass in wrong directions (→1 = collapse risk)                                                     |
| `${train/kl_q_np}`                    | `collapse_signal`   | Negative adv + ratio>1: policy getting more likely to generate bad seqs — primary collapse indicator |


## Timing


| Expression                                                                                     | Name                 | What it tells you                            |
| ---------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------- |
| `${train/gen_seconds} + ${train/rew_seconds} + ${train/grad_seconds}`                          | `total_step_seconds` | Total wall time per step                     |
| `${train/gen_seconds} / (${train/gen_seconds} + ${train/rew_seconds} + ${train/grad_seconds})` | `gen_time_frac`      | Fraction of step spent on rollout generation |
| `${train/rew_seconds} / (${train/gen_seconds} + ${train/rew_seconds} + ${train/grad_seconds})` | `rew_time_frac`      | Fraction of step spent on reward computation |


---

## Metric reference

All metrics logged in `rl/algorithms/cppo.py` → `wandb.log(...)`:

### Core

- `loss` — CPPO surrogate loss (minimised); = `loss_contrib_neg_rew + loss_contrib_pos_rew`
- `average_reward` — mean reward across all B×G rollouts
- `train/reward_std` — mean per-prompt reward std (measures rollout diversity)
- `train/reward_max` / `train/reward_min` — global max/min across B×G

### Policy entropy

- `train/entropy` — per-token entropy after last batch_update (H = −mean log p(sampled token))
- `train/entropy_k0` — same but before any gradient update (k=0); tracks how entropy changes within a step

### KL & ratio

- `train/kl_approx` — approx KL(new‖old), token-count-weighted; = `kl_k3`
- `train/kl_k1` — k1 estimator: −log r (can be negative)
- `train/kl_k2` — k2 estimator: 0.5·(log r)² (always ≥ 0)
- `train/kl_k3` — k3 estimator: (r−1)−log r (always ≥ 0); = `kl_approx`
- `train/ratio_mean` / `train/ratio_std` — mean/std of per-sequence importance ratio
- `train/clip_fraction` — fraction of tokens where ratio was clipped
- `train/clip_lower_frac` / `train/clip_upper_frac` — clipped below (1−ε) vs above (1+ε)

### KL quadrants (fraction of total KL mass)

- `train/kl_q_pp` — adv>0 & ratio>1 : policy reinforcing good seqs ✓
- `train/kl_q_pn` — adv>0 & ratio<1 : policy weakening good seqs ✗
- `train/kl_q_np` — adv<0 & ratio>1 : policy strengthening bad seqs ✗ (collapse signal)
- `train/kl_q_nn` — adv<0 & ratio<1 : policy suppressing bad seqs ✓

### Advantage

- `train/adv_pos_frac` — fraction of top-N sequences with positive advantage
- `train/adv_abs_mean` — mean |advantage| (signal strength)
- `train/adv_mean_seq` / `train/adv_mean_tok` — signed mean advantage (seq-level vs tok-weighted)

### Loss contribution (sum = `loss` when B % mini_batch_size == 0)

- `train/loss_contrib_neg_rew` — loss contribution from reward<0 sequences (positive = pushing loss up)
- `train/loss_contrib_pos_rew` — loss contribution from reward≥0 sequences (negative = pulling loss down)
- `train/neg_rew_loss_frac` — of all sequences pushing loss up, fraction with reward<0

### Rollout distribution (based on top-N selected sequences)

- `train/failure_rate` — fraction of all B×G rollouts with reward<0
- `train/topN_neg_frac` — fraction of top-N selected sequences with reward<0
- `train/fail_prompt_frac` — prompts where ALL top-N rewards<0
- `train/prompt_all_pos_frac` — prompts where ALL top-N rewards>0
- `train/prompt_geq_half_pos` — prompts where >N/2 top-N rewards>0

### Timing & misc

- `train/avg_gen_len` — mean completion length in tokens
- `train/gen_seconds` / `train/rew_seconds` / `train/grad_seconds` — wall time per phase
- `train/pool_crashes` — reward worker pool crashes this step
- `train/lr` — current learning rate

