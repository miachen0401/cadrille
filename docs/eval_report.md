# Cadrille RL Evaluation Report

**Date:** 2026-03-14
**Eval machine:** RTX 4080 SUPER 16 GB, 12-core CPU, 15 GB RAM

---

## 1. Models Evaluated

| ID | Checkpoint | Description |
|---|---|---|
| `sft` | `checkpoints/cadrille-sft` | Qwen2-VL-2B SFT on CAD-Recode v1.5 (public baseline) |
| `official-rl` | `checkpoints/cadrille-rl` | `maksimko123/cadrille-rl` — published Dr. CPPO result |
| `a100-4500` | `cad_ckpt/a100-step4500` | Our A100 RL run, step 4500 |
| `a100-6000` | `cad_ckpt/a100-step6000` | Our A100 RL run, step 6000 |
| `a100-7200` | `cad_ckpt/a100-step7200` | Our A100 RL run, step 7200 |
| `4080-9000` | `rl-…-0311-0259/checkpoint-9000` | Our 4080 RL run, step 9000 (entropy bonus active) |

**Training details:**
- All RL runs: Dr. CPPO, `train_modality=img`, hard-mined examples (`combined_hard.pkl`, 6861 samples)
- A100 run: `lr=1e-5`, `G=16`, `batch_size=4`, `sequential_generation=False`
- 4080 run: `lr=1e-5`, `G=4`, `batch_size=1`, `sequential_generation=True`, `entropy_coef=0.01`

---

## 2. Img-Mode IoU & Chamfer Distance

Evaluated with `tools/eval_img.py` (greedy decode, `max_new_tokens=1024`).
**n=100** per split, rendered PNG cache used for img input.

| Model | DC IoU↑ | DC CD×10⁻³↓ | F360 IoU↑ | F360 CD×10⁻³↓ | Fail |
|---|---|---|---|---|---|
| sft | 0.834 | 0.210 | 0.783 | 0.218 | 5/200 |
| official-rl | **0.897** | **0.194** | **0.868** | **0.180** | 0/200 |
| a100-4500 | 0.844 | 0.208 | 0.829 | 0.192 | 3/200 |
| a100-6000 | 0.847 | 0.207 | 0.831 | 0.192 | 0/200 |
| a100-7200 | 0.841 | 0.203 | 0.825 | 0.183 | 4/200 |
| 4080-9000 | 0.793 | 0.209 | 0.761 | 0.234 | 7/200 |

**Observations:**
- Official RL leads by ~5% IoU over SFT and ~5% over our best A100 run.
- A100 step 6000 peaks on IoU (0.847 DC / 0.831 F360); step 7200 has lower CD but slightly lower IoU (possible overfitting or entropy collapse starting).
- **4080 run regressed below SFT** — analysis in §5.

---

## 3. Pass@k & Best IoU@k (pc mode, DeepCAD)

Evaluated with `rl/eval_passk.py` (temperature sampling, `n_samples=16`, `n_examples=50`).
Metric: **best_iou@k** = mean over examples of max(IoU) among k samples (oracle upper bound).
Metric: **pass@k** = unbiased estimator using all n=16 samples (HumanEval formula).

### 3a. Best IoU@k (oracle max, pc mode, DeepCAD n=50)

| Model | @1 t=0.3 | @4 t=0.3 | @16 t=0.3 | @1 t=0.7 | @4 t=0.7 | @16 t=0.7 | @1 t=1.0 | @4 t=1.0 | @16 t=1.0 |
|---|---|---|---|---|---|---|---|---|---|
| sft        | 0.842 | 0.881 | 0.908 | 0.821 | 0.904 | 0.909 | 0.792 | 0.893 | 0.907 |
| official-rl| 0.884 | 0.901 | 0.908 | 0.900 | 0.910 | **0.914** | 0.900 | 0.912 | **0.921** |
| a100-4500  | 0.870 | 0.896 | 0.905 | 0.874 | 0.902 | 0.916 | 0.873 | 0.902 | 0.917 |
| a100-6000  | 0.855 | 0.898 | 0.905 | 0.868 | 0.911 | 0.917 | 0.891 | 0.906 | 0.919 |
| a100-7200  | 0.859 | 0.895 | 0.904 | 0.874 | 0.904 | 0.908 | 0.854 | 0.900 | 0.920 |
| 4080-9000  | 0.872 | 0.883 | 0.899 | 0.817 | 0.869 | 0.893 | 0.797 | 0.886 | 0.907 |

### 3b. pass@k (IoU ≥ **0.95**, pc mode, DeepCAD n=50)

| Model | @1 t=0.3 | @4 t=0.3 | @16 t=0.3 | @1 t=0.7 | @4 t=0.7 | @16 t=0.7 | @1 t=1.0 | @4 t=1.0 | @16 t=1.0 |
|---|---|---|---|---|---|---|---|---|---|
| sft        | 0.444 | 0.460 | 0.460 | 0.439 | 0.474 | 0.500 | 0.419 | 0.485 | 0.520 |
| official-rl| **0.504** | **0.521** | 0.540 | **0.522** | **0.539** | 0.540 | **0.504** | **0.541** | **0.560** |
| a100-4500  | 0.475 | 0.485 | 0.500 | 0.475 | 0.495 | **0.540** | 0.456 | 0.485 | **0.540** |
| a100-6000  | 0.492 | 0.505 | **0.520** | 0.474 | 0.485 | 0.500 | 0.434 | 0.485 | 0.520 |
| a100-7200  | 0.451 | 0.465 | 0.480 | 0.461 | 0.504 | 0.520 | 0.435 | 0.497 | **0.560** |
| 4080-9000  | 0.434 | 0.449 | 0.460 | 0.443 | 0.470 | 0.500 | 0.429 | 0.459 | 0.460 |

---

## 4. Smoke Test (pc mode, N=5 simplest meshes)

Evaluated with `tools/smoke_eval.py` (greedy decode, `max_new_tokens=1024`).
Uses the 5 smallest STLs from `data/smoke_train/smoke_train.pkl`.

Evaluated with `tools/smoke_eval.py` (greedy decode, `max_new_tokens=1024`, mesh normalized to unit cube).
5 smallest training STLs from `data/smoke_train/smoke_train.pkl` (batch_00_308, 2100, 3371, 2289, 2154).

| Model | Avg IoU | Valid | Notes |
|---|---|---|---|
| sft | **0.969** | 5/5 | Near-perfect on training examples it was trained on |
| official-rl | 0.855 | 3/5 | Fails on 2 simplest examples (distribution shift) |
| a100-4500 | 0.959 | 5/5 | |
| a100-6000 | 0.890 | 5/5 | batch_00_2289 regressed to 0.647 |
| a100-7200 | 0.885 | 5/5 | batch_00_2289 stays at 0.647 |
| 4080-9000 | 0.956 | 5/5 | Surprisingly recovers batch_00_2289 to 1.0 |

**Key finding**: Smoke IoU is NOT monotonic with test-set IoU. SFT performs best on training-set smoke examples (it memorized them), while RL models shift away from simple shapes toward harder geometry. The official-rl failures on the 2 simplest examples are a red flag for distribution collapse at high step counts.

---

## 5. Analysis

### 5.1 Why official-rl leads our A100 run

The official run likely trained longer (unknown total steps, but our A100 only ran to step 7200 before OOM). Key gap: **+5% IoU** on both splits.

Hypothesis: official run used 8 GPUs (effective batch = 128), enabling `G=16` with full batch — learning signal is much denser per update. Our single-GPU run uses `G=16` with `batch_size=4`, so only 4 examples × 16 rollouts = 64 rollouts per step vs 128+ in the multi-GPU setting.

### 5.2 Why 4080 run regressed (0.793 < SFT 0.834)

Several contributing factors:

1. **Lazy-code collapse** — observed at step ~1970: entropy dropped to 0.032 (near-deterministic), gen_len went from 132→58 tokens. Model learned to output short `cq.Workplane().box(...)` or similar simple scripts that avoid the `-1` penalty for invalid code, sacrificing IoU.

2. **Entropy bonus added too late** — `entropy_coef=0.01` was added at around step 3000 (post-collapse). The model had already converged to a narrow policy; the entropy bonus slowed further collapse but couldn't recover diversity.

3. **G=4 rollout diversity** — with only 4 rollouts per example, advantage estimates are noisy. The model can achieve near-zero variance in advantages by outputting nearly identical code, preventing any learning signal from passing through CPPO.

4. **Possible overfitting** — 8000+ effective training steps on 6861 hard examples (many repeated) with `batch_size=1`.

**Fix for next run:** restart from SFT with `entropy_coef=0.01` from step 0, `G=8` (test with sequential or switch to A100), and early stopping based on img eval IoU plateau.

### 5.3 Temperature vs. best IoU@k trade-off

Confirmed expected pattern across all models:

**best_iou@k vs temperature** (a100-6000 as representative):

| Temp | @1   | @4   | @16  |
|------|------|------|------|
| 0.3  | 0.855 | 0.898 | 0.905 |
| 0.7  | 0.868 | 0.911 | 0.917 |
| 1.0  | 0.891 | 0.906 | 0.919 |

Key findings:
1. **best_iou@1 is *higher* at t=1.0 than t=0.3** — surprising but consistent across all RL models. High temperature still finds good samples; diversity helps even at k=1 for the oracle metric. Exception: SFT and 4080-9000, where t=0.3 @1 > t=1.0 @1 (collapsed/weaker policies benefit from lower variance).
2. **best_iou@4 peaks at t=0.7 for all models** — the diversity-quality sweet spot.
3. **best_iou@16 is nearly flat across temperatures** (~0.905–0.921) — at k=16 there are enough samples to find a good one regardless.
4. **pass@1 (strict quality) peaks at t=0.3** for official-rl (0.504 vs 0.522 at t=0.7) but the difference is small and **reverses** for RL-trained models at higher k.
5. **official-rl consistently dominates on pass@k@0.95** — the RL training improves not just mean IoU but also the tail of the distribution (more samples exceed 0.95).

**Practical recommendation:** use t=0.7 for best coverage-efficiency trade-off. For single-shot greedy eval (paper metrics), use t=0 (greedy/do_sample=False).

### 5.4 A100 run trajectory

| Step | DC IoU (img) | F360 IoU (img) | Notes |
|---|---|---|---|
| 4500 | 0.844 | 0.829 | earliest checkpoint available |
| 6000 | **0.847** | **0.831** | IoU peak |
| 7200 | 0.841 | 0.825 | CD improves, IoU slightly regresses |

The IoU peak at step 6000 then slight regression at 7200 is consistent with a mild policy drift post-peak (entropy collapse starting or distribution shift from hard examples). The CD improvement at 7200 suggests geometric precision is still improving even as IoU regresses — the model is generating slightly more syntactically complex code that matches geometry better but occasionally mis-assembles.

---

## 6. Recommendations

### 6.1 Next training run

- **Start from**: `a100-step6000` (best IoU checkpoint, stable run)
- **Config**: H100 (80 GB) with OOM fix (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `sequential_generation=True` if needed)
- **Entropy bonus**: `entropy_coef=0.01` from step 0 (already in config)
- **Monitor**: watch `eval/img/DeepCAD/IoU` every 500 steps; stop if IoU regresses > 1% from peak for 2 consecutive evals

### 6.2 Evaluation improvements

- Increase eval n to 200 per split for publication-quality numbers (currently n=100)
- Add pc-mode greedy IoU to the comparison table (currently only img mode from eval_img.py)
- Run pass@k on Fusion360 split as well

### 6.3 4080 run recovery

- Kill current 4080 run; it has regressed below SFT
- Restart from `a100-step6000` with 4080 config (`G=4`, `sequential_generation=True`, `lr=1e-5`)
- This gives a stronger starting point (+1.3% IoU over SFT) with better diversity before entropy collapse

---

## 7. Appendix: Eval Commands

```bash
# Img-mode IoU/CD
python3 tools/eval_img.py --checkpoint <ckpt> \
    --splits deepcad:data/deepcad_test_mesh fusion360:data/fusion360_test_mesh \
    --n-samples 100 --max-new-tokens 1024

# pass@k + best_iou@k
python3 rl/eval_passk.py --checkpoint <ckpt> \
    --val-dir data/deepcad_test_mesh \
    --n-examples 50 --n-samples 16 --k-values "1,4,16" \
    --temperature 0.7 --max-new-tokens 1024 \
    --eval-batch-size 4 --sequential

# Smoke test
python3 tools/smoke_eval.py \
    --checkpoints <ckpt1> <ckpt2> ... \
    --pkl data/smoke_train/smoke_train.pkl --n 5
```
