# CAD Process Reward Model — Design Discussion
**Context**: Direction 1 from `direction.md`. Building on Cadrille's existing `reward.py` and `rl_train.py`.

---

## 1. Reframing the Problem: Why Even Use a Reward "Model"?

The user observation is sharp and correct: **in CAD, the outcome is fully verifiable**. You execute the CadQuery program, get a mesh, and compute IoU against ground truth. There is no ambiguity, no human preference, no hallucination to detect. This is the opposite of RLHF scenarios (essay quality, helpfulness) where a learned reward model is *required* because the ground truth is inherently subjective.

So why consider a learned reward model at all?

The answer is not about verifiability of the **outcome** — it's about three structural problems in the **RL training process**:

### 1.1 The Credit Assignment Problem

CadQuery programs are **100–400 lines** long. The current reward (from `reward.py`) is delivered **once**, at the very end, for the entire sequence:

```
line 1:  wp = cq.Workplane("XY")          ← no signal
line 2:  wp = wp.circle(1.5)              ← no signal
...
line 47: r = r.fillet(0.05)               ← no signal
         [EXECUTE] → IoU = 0.82 → reward = 8.2   ← only signal
```

When a rollout gets reward 8.2 and another gets 3.1, GRPO assigns an advantage to the **entire sequence**. But which of the 400 tokens was responsible for the improvement? The model cannot tell. Dense intermediate signals would enable credit assignment at the operation level.

### 1.2 The Validity Cliff

The reward function has a massive discontinuity:
- **Valid code**: reward ∈ [0, 10] (IoU × 10)
- **Invalid code**: reward = −10 (hard penalty)

Early in RL training, many rollouts are invalid (IR = 7–10% at SFT initialization on CC3D). For all invalid rollouts, the gradient signal is the same: −10. The model learns "don't be invalid" but gets zero information about **which direction** to improve the code. A process reward that fires on syntactic correctness of each line, or on each partial execution attempt, would give gradient signal even for invalid completions.

### 1.3 The Pre-Solid Blindness

In CadQuery, geometry does not exist until a `solid-creating operation` is executed:

```python
wp = cq.Workplane("XY")          # workplane only — no geometry
wp = wp.rect(2.0, 3.0)           # 2D sketch — no solid
wp = wp.extrude(1.5)             # ← FIRST SOLID. Now we have geometry.
wp = wp.faces(">Z").circle(0.5)  # sketch on top face — no new solid yet
wp = wp.cutThruAll()             # ← SOLID MODIFICATION.
```

The "pre-solid" phase (workplane setup, sketch creation) can span 5–50 lines for complex parts. During this phase, there is **nothing** to verify geometrically. A learned reward model can predict "given this sketch setup, will the extruded solid match the target?" — a prediction that no rule-based oracle can make.

---

## 2. The Verifiability Spectrum in CadQuery

Not all reward signals are equally verifiable. Here is the full spectrum:

| Stage | Signal Type | Verifiable? | Cost |
|---|---|---|---|
| Post-complete-execution | Outcome IoU, CD, IR | **Exact** | 1× subprocess |
| Post-solid-op (partial) | Partial-solid IoU | **Exact** (but ambiguous meaning) | K× subprocess |
| Post-sketch (pre-solid) | 2D sketch area/topology | **Exact** for geometry, **heuristic** for quality | cheap |
| Any line | Syntactic validity | **Exact** (parse/compile) | negligible |
| Any line | Semantic plausibility | **Heuristic** (range checks, sequence validity) | negligible |
| Any prefix | Value prediction | **Approximated** (learned) | 1× small-model forward |

**The key insight**: CAD generation has a **verifiable upper layer** (outcome) and a **partially-unverifiable lower layer** (per-step credit). The learned reward model fills in the lower layer.

---

## 3. Four Implementation Methods

### Method A: Incremental Geometric Oracle (Rule-Based, Dense at Solid Ops)

**Concept**: Parse the generated code line by line. After each *solid-creating or solid-modifying operation*, execute the code prefix and compute a geometric score against the ground truth.

**Solid-creating operations in CadQuery:**
```python
SOLID_OPS = {
    'extrude', 'revolve', 'loft', 'sweep',    # create solid from sketch
    'union', 'cut', 'intersect', 'cutBlind',   # boolean ops
    'cutThruAll', 'shell', 'fillet', 'chamfer', # modifications
    'add',                                      # assembly-level
}
```

**Partial-solid reward formula:**

The partial solid at operation k should be compared to the GT using a *recall-biased* F1 metric to avoid penalizing "not yet added" geometry:

```
recall_k    = vol(partial_k ∩ GT) / vol(GT)
precision_k = vol(partial_k ∩ GT) / vol(partial_k)
r_partial_k = 2 * recall_k * precision_k / (recall_k + precision_k + ε)
```

This is identical to the final IoU formula — the partial solid is simply evaluated as if it were the final output. For early operations (k=1 of K), we expect low recall and decent precision if the model is on track.

**Step-level advantage accumulation:**
```
R_process(τ) = Σ_k γ^(K-k) · r_partial_k   (discounted sum, γ ≈ 0.9)
```

**Integration with existing `reward.py`:**
```python
def compute_reward_incremental(code_str, gt_mesh_path, gamma=0.9):
    lines = code_str.split('\n')
    prefixes = []
    for i, line in enumerate(lines):
        stripped = line.strip().split('(')[0].split('.')[-1]
        if stripped in SOLID_OPS:
            prefixes.append('\n'.join(lines[:i+1]))

    partial_rewards = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_execute_code_in_subprocess, p, gt_mesh_path)
                   for p in prefixes]
        for iou in [f.result() for f in futures]:
            partial_rewards.append((iou or 0.0) * 10.0)

    # Discounted sum + final outcome reward
    K = len(partial_rewards)
    process_reward = sum(gamma**(K-k) * r for k, r in enumerate(partial_rewards))
    outcome_reward = compute_reward(code_str, gt_mesh_path)  # existing
    return α * outcome_reward + (1-α) * process_reward
```

**Pros:**
- No training data required
- Geometrically exact (no approximation)
- Directly grounded in CAD semantics

**Cons:**
- **Expensive**: K subprocess calls per rollout (K ≈ 5–15 ops per program × G=16 rollouts = 80–240 extra executions per GRPO step). At ~0.5s/execution: adds **40–120s per step**. Currently a step is ~2–5s. This makes the RL loop 10–25× slower.
- Cannot reward the pre-solid phase (first N lines before first `extrude()`)
- Partial solid comparison is ambiguous for parts with multiple disconnected bodies

**Mitigation**: Only run partial executions for the **top-N selected rollouts** (N=4 in current CPPO), not all G=16. Cost: 4 × 10 = 40 extra executions per step → ~20s overhead → manageable.

---

### Method B: Learned Value Function — Offline PRM

**Concept**: Train a small model to predict `E[final_IoU | prefix]` from a dataset of (prefix, final_outcome) pairs collected from SFT rollouts. This is the classical Process Reward Model approach from math reasoning (Lightman et al. 2023, OmegaPRM).

**Training data collection:**
```python
# 1. For each of 5000 examples from DeepCAD/Fusion360:
#    Sample K completions from SFT model at temperature 1.0
# 2. Execute each completion → get final_IoU
# 3. For each completion, extract ALL prefixes at:
#    [25%, 50%, 75%, 100%] of token length
# 4. Label each prefix with the completion's final_IoU

# Dataset: ~5000 × 5 rollouts × 4 prefixes = 100,000 (prefix, IoU) pairs
```

**PRM Architecture:**

Option B1 — **Regression head on the policy LM itself:**
```python
# Add a scalar head to the existing Cadrille model
class CadrillePRM(Cadrille):
    def __init__(self, config):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward_value(self, input_ids, attention_mask, point_clouds, is_pc, is_img, ...):
        # Forward through LM backbone (no lm_head)
        hidden = self.model(input_ids=input_ids, ...)
        # Take last non-padding token's representation
        last_hidden = hidden[0][:, -1, :]  # [batch, hidden_size]
        return self.value_head(last_hidden).squeeze(-1)  # [batch]
```

Option B2 — **Separate small model (Qwen2-0.5B):**
```python
# Train a dedicated 0.5B model as PRM
# Input: (point_cloud_tokens || code_prefix_tokens)
# Output: scalar ∈ [0, 1] (predicted final IoU)
# Loss: MSE against actual final IoU from rollout labels
```

**Training:**
```python
loss = F.mse_loss(prm(prefix, input), torch.tensor(final_iou))
# Train for 2-3 epochs on 100K (prefix, IoU) pairs
# ~8 GPU-hours on 4080 for 0.5B model
```

**Integration into GRPO:**
```python
# Replace outcome reward with PRM reward during RL training
def compute_reward_prm(partial_code, full_code, gt_mesh_path, prm_model):
    step_rewards = []
    for prefix in extract_checkpoints(partial_code):
        step_rewards.append(prm_model(prefix, gt_mesh_path))
    outcome = compute_reward(full_code, gt_mesh_path)  # ground truth at end
    return α * outcome + (1-α) * mean(step_rewards)
```

**Pros:**
- Fast at inference: one small-model forward pass per checkpoint (no CadQuery execution)
- Works for pre-solid phase (PRM infers from sketch structure, not geometry)
- Can capture subtle correlations (e.g., certain sketch patterns correlate with bad outcomes)

**Cons:**
- Requires training data (but only ~8 GPU-hours to generate + train)
- **Distribution shift**: PRM trained on SFT rollouts may be miscalibrated for improved policy rollouts. Needs periodic retraining (every ~5k RL steps).
- Approximation error: PRM predicts an expectation, not the true value

---

### Method C: Monte Carlo Rollout Value Estimation (Gold Standard)

**Concept**: Instead of training a model, directly estimate `E[final_IoU | prefix_k]` by sampling M completions from the current policy and averaging their outcomes. This is the **exact** process reward — no approximation.

**Algorithm:**
```
For each rollout step k (checkpointed every Q lines):
    Sample M=4 completions from current policy starting at prefix_k
    Execute all M completions → get IoU_1, ..., IoU_M
    process_reward_k = mean(IoU_1, ..., IoU_M)
```

**Why it's the gold standard**: The estimated value is unbiased and tracks the current policy automatically (no distribution shift).

**Cost analysis:**
- G=16 rollouts, each 300 tokens, checkpoint every 100 tokens → 3 checkpoints per rollout
- M=4 completions per checkpoint
- Total extra executions per GRPO step: 16 × 3 × 4 = 192 completions
- At ~5s per generation + execution: 960s extra per step → **completely impractical**

**Practical variant**: Only apply MC estimation to **1 checkpoint per selected rollout** (the midpoint of the top-N completions):
- N=4 rollouts × 1 checkpoint × M=4 completions = 16 extra executions per step
- At ~5s each: +80s per step — still significant but manageable

**Better variant**: Use as a **validation metric** rather than a training signal:
- Every 1000 RL steps, estimate process rewards for 50 held-out examples
- Use this to **validate and calibrate** the learned PRM (Method B)
- No runtime overhead during training

---

### Method D: Rule-Based Syntactic + Semantic Rewards (Dense, Free)

**Concept**: Derive rewards from the code text itself, without any CadQuery execution. These rewards are computed line-by-line and capture structural properties.

**Sub-rewards (all rule-based, computable in <1ms):**

```python
def compute_syntactic_rewards(code_str, gt_info=None):
    rewards = {}

    # D1: Python compile check (is prefix valid Python?)
    try:
        compile(code_str, '<string>', 'exec')
        rewards['syntax_ok'] = 1.0
    except SyntaxError:
        rewards['syntax_ok'] = 0.0

    # D2: CadQuery API validity (are method names valid CQ calls?)
    used_methods = re.findall(r'\.(\w+)\(', code_str)
    valid_methods = KNOWN_CQ_METHODS  # pre-compiled set
    rewards['api_validity'] = len(set(used_methods) & valid_methods) / max(len(set(used_methods)), 1)

    # D3: Numeric range check (are values physically plausible?)
    numbers = [float(x) for x in re.findall(r'\b(\d+\.?\d*)\b', code_str)]
    if numbers:
        in_range = sum(0.001 < n < 100 for n in numbers) / len(numbers)
        rewards['numeric_plausibility'] = in_range

    # D4: Operation sequence validity (extrude must follow a sketch)
    rewards['sequence_valid'] = check_op_sequence(code_str)

    # D5: Variable reuse (r is assigned and reused; no orphan variables)
    rewards['variable_consistency'] = check_variable_reuse(code_str)

    # D6: Code length appropriateness (if we know complexity from input)
    if gt_info and 'complexity' in gt_info:
        expected_len = COMPLEXITY_TO_LENGTH[gt_info['complexity']]
        actual_len = len(code_str.split('\n'))
        rewards['length_appropriateness'] = max(0, 1 - abs(actual_len - expected_len) / expected_len)

    return rewards
```

**Per-token reward delivery:**
```python
# Assign reward to the token that completed the line/operation
# All other tokens: 0 reward
# This gives a sparse-but-dense signal (one reward per line ≈ 5-10 rewards per program)
```

**Pros:**
- Zero compute cost (no subprocess, no model)
- Fires even on invalid/pre-solid code
- Can be applied to every rollout in G=16 with no overhead

**Cons:**
- No geometric content — doesn't know if the shape is right, only if the code is structurally sound
- Limited signal richness; doesn't differentiate between geometrically correct and incorrect code

**Best use**: As a **baseline reward floor** — always applied, providing signal even when all rollouts are invalid. Prevents reward collapse during early RL training.

---

## 4. The Core Debate: Data-Driven vs. Rule-Based

Given that CAD outcomes **are** verifiable, let's directly address the user's question.

### When Rule-Based is Sufficient (and Better)

For the **final outcome**, rule-based is strictly better:
- IoU is exact, not approximated
- No training data needed
- No distribution shift
- Already implemented in `reward.py`

For **syntactic/semantic checking** (Method D), rule-based is also better:
- Zero cost, always available
- Perfect precision for Python syntax errors

For **post-solid incremental rewards** (Method A), rule-based is better IF cost is acceptable:
- Exact geometric signal
- No approximation
- The only cost is more CadQuery subprocesses

### When a Learned Model is Necessary

A learned reward model is **strictly necessary** in exactly one scenario: **predicting the value of pre-solid code prefixes** — code that has not yet produced any geometry.

Consider:
```python
# Line 1:
wp = cq.Workplane("XY")   # correct orientation for a flat part
vs.
wp = cq.Workplane("XZ")   # wrong orientation — will produce wrong geometry later
```

No rule-based oracle can tell you which is better at this point. Only a model that has seen thousands of (workplane_setup, final_IoU) pairs can learn that "XY" vs "XZ" matters. This is what the learned PRM captures.

**Quantifying the gap**: In the SFT model on CC3D, ~30% of errors originate in the first 10 tokens (wrong workplane, wrong initial scale). A PRM that signals the error at token 10 instead of token 400 gives 40× more gradient signal to fix the root cause.

### When the Verifiability Advantage Flips

There's a subtle point: **incremental execution (Method A) may give misleading rewards**.

Example: generating a complex bracket with 3 extrusions:
```
Step 1: extrude base → recall=0.4, precision=0.9 → partial IoU = 0.55 ✓
Step 2: extrude rib → recall=0.7, precision=0.75 → partial IoU = 0.72 ✓
Step 3: cut hole → recall=0.7, precision=0.85 → partial IoU = 0.76 ✓
Final: full execution → IoU = 0.91 ✓
```
vs. a wrong sequence:
```
Step 1: extrude wrong base → recall=0.45, precision=0.85 → partial IoU = 0.59 ✓ (looks OK!)
Step 2: extrude in wrong direction → recall=0.5, precision=0.6 → partial IoU = 0.55 ✓ (still OK!)
Step 3: ... → ...
Final: full execution → IoU = 0.32 ✗ (disaster)
```

The partial rewards in the wrong sequence look decent at steps 1 and 2. The oracle is **correctly verifying** the partial geometry, but that geometry is not predictive of the final quality. The learned PRM, trained on (prefix, final_outcome) pairs, would learn to distinguish these by memorizing that certain patterns lead to divergence.

**Conclusion**: Rule-based incremental rewards are necessary but not sufficient. A learned PRM adds information that the geometric oracle cannot: **which structural choices at early stages correlate with good outcomes at the end**.

---

## 5. Recommended Hybrid Architecture

Combining the best of all four methods:

```
R_total(τ) = λ₁ · R_outcome(τ)           [Method A/rule-based, end of sequence]
           + λ₂ · R_incremental(τ)        [Method A, only at solid-creating ops]
           + λ₃ · R_syntactic(τ)          [Method D, every line, zero cost]
           + λ₄ · R_PRM(τ)               [Method B, pre-solid + full sequence]
```

With curriculum schedule:
```
Phase 1 (steps 0-2k):   λ=(1.0, 0.0, 0.5, 0.0)   — outcome + syntax only
Phase 2 (steps 2k-5k):  λ=(1.0, 0.5, 0.2, 0.0)   — add incremental
Phase 3 (steps 5k+):    λ=(1.0, 0.5, 0.1, 0.5)   — add PRM
```

Rationale:
- Start simple: the model needs to first learn to produce valid code (outcome + syntax)
- Add geometric incrementality once validity rate exceeds ~90%
- Add learned PRM only after PRM is trained (requires collecting rollouts from Phase 1+2)

### Practical Implementation Priority

Given limited compute and a tight deadline:

**MVP (can implement in 1 week):**
1. Method D (syntactic rewards) — add to `reward.py` in ~50 lines, zero cost
2. Method A (incremental, but only for final top-N completions) — reuse existing `_execute_code_in_subprocess`
3. Ablation: outcome-only vs. +syntax vs. +incremental (3 training runs)

**Full paper version (weeks 2-4):**
4. Method B (learned PRM) — 2 days to collect rollouts + 1 day to train
5. Combine all four in hybrid reward
6. MC validation (Method C) as analysis tool, not training signal

---

## 6. Data Collection for Learned PRM (Method B)

### Step 1: Rollout Collection

```python
# collect_prm_data.py
# Run after SFT training; does NOT require any RL training

model = Cadrille.from_pretrained('checkpoints/sft-final')

dataset = DeepCADDataset(split='train', n=5000)  # 5K examples
rollouts = []

for item in dataset:
    for _ in range(5):  # 5 stochastic rollouts per item
        code = model.generate(item, temperature=1.0)
        iou = compute_reward(code, item['gt_mesh_path'])  # existing function

        # Extract checkpoints at 0%, 25%, 50%, 75%, 100% of code length
        tokens = tokenize(code)
        for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
            prefix = detokenize(tokens[:int(frac * len(tokens))])
            rollouts.append({
                'input': item,           # point cloud / image
                'prefix': prefix,        # partial code
                'label': iou / 10.0,     # target value ∈ [0, 1]
                'fraction': frac,
            })

# Dataset size: 5000 × 5 × 5 = 125,000 (prefix, IoU) pairs
# Generation time: ~1h on 4080 (5000 × 5 × 5s/completion)
# Storage: ~500MB
```

### Step 2: PRM Training

```python
# Option B1: Scalar head on existing Cadrille
# Fine-tune only the value_head (frozen backbone) for efficiency

prm = CadrillePRM.from_pretrained('checkpoints/sft-final')
for p in prm.parameters():
    p.requires_grad_(False)
for p in prm.value_head.parameters():
    p.requires_grad_(True)

# Train: ~4 GPU-hours for 10 epochs on 125K samples
optimizer = Adam(prm.value_head.parameters(), lr=1e-4)
for batch in dataloader:
    v = prm.forward_value(batch['prefix'], batch['input'])
    loss = F.mse_loss(v, batch['label'])
    loss.backward(); optimizer.step()
```

### Step 3: PRM Calibration Check

Before using PRM in RL training, verify calibration on held-out rollouts:
```python
# Expected: PRM(prefix at 50%) should correlate with final_IoU
# Metric: Spearman rank correlation ρ
# Acceptable: ρ > 0.6 at each fraction
# Red flag: ρ < 0.3 → PRM is not useful, abandon Method B
```

---

## 7. Key Design Decisions Summary

| Decision | Option A | Option B | Recommendation |
|---|---|---|---|
| Rule-based vs. learned | Rule-based at solid ops | Learned for pre-solid | **Both** (hybrid) |
| Incremental cost | High (K×G subprocesses) | Low (1× small model) | Start with top-N only (Method A) |
| Training data needed | No | Yes (~125K rollouts) | Collect in parallel with SFT |
| Handles pre-solid phase | No | Yes | PRM critical for this |
| Distribution shift | None | Yes (refresh every 5k RL steps) | Budget 1 extra training day |
| Partial IoU meaning | Clear (recall+precision) | Implicit (learned) | Document the formula explicitly |
| Integration complexity | Low (extend `reward.py`) | Medium (new PRM model) | Start with Method A+D, add B later |

---

## 8. Comparison with ReCAD's Reward Design (arXiv Dec 2025)

ReCAD (Qwen2.5-VL-7B, 8 A800 GPUs, GRPO G=8) introduces the richest RL reward design in the CAD literature to date. Understanding it is essential for positioning our CAD-PRM work.

### ReCAD Reward Formula

```
R_total = λ₁ · R_geometric + λ₂ · R_format
λ₁ = 0.1,  λ₂ = 0.9

R_geometric = min(IoU_best, DINOv2_similarity)
  where IoU_best  = best IoU across top-K executions
        DINOv2_similarity = cosine_sim(DINOv2(rendered_pred), DINOv2(rendered_gt)) > τ=0.55

R_format = 1.0 if response contains valid <think>...</think> + <answer>...</answer> blocks
           0.0 otherwise
```

**Key observation**: the format reward (λ₂=0.9) dominates the geometric reward (λ₁=0.1). This reflects their primary goal: inducing chain-of-thought reasoning. The geometric reward is a tie-breaker.

### ReCAD "Learn Under Guidance" Mechanism

For hard examples where `R_geometric < 0.8` after G=8 rollouts, ReCAD injects the *parameterized ground-truth code* as in-context guidance:

```
Prompt (normal):  [image] Generate CadQuery code for this shape.
Prompt (guided):  [image] Here is a parametric template: {gt_code_skeleton}
                  Complete the parameters to match the image.
```

The code skeleton has exact operations but numeric constants replaced by `<PARAM_i>` tokens. The model learns to fill in parameters rather than generate structure from scratch.

**Effect on RL**: For ~20% of examples (hard ones), the model sees a different, easier task. This effectively creates a curriculum — but it requires access to GT code skeletons, which are only available for synthetic datasets (DeepCAD, Fusion360, ABC), not for real-world scans.

### ReCAD SFT Design (for context)

| Aspect | ReCAD | Cadrille |
|--------|-------|----------|
| Base model | Qwen2.5-VL-7B | Qwen2-VL-2B |
| SFT data size | 359K samples | 2,810 samples |
| SFT data sources | UltraChat-85K + OpenCodeReasoning-20K + CAD-254K | CAD-Recode-2.8K |
| CAD API | Custom (Loops/Faces/Sketches/Extrude, 8-bit quantized coords) | CadQuery (Python) |
| Input modalities | Image only | Point cloud + image |
| Curriculum | Hierarchical (primitives → parts) | None |
| lr SFT | 1e-5 | 2e-4 |

### Comparison: ReCAD vs. Our Proposed Hybrid Reward

| Property | ReCAD Reward | Our CAD-PRM |
|-----------|-------------|-------------|
| Geometric signal | Outcome IoU + DINOv2 sim | Outcome IoU + incremental partial-solid IoU |
| Step-level credit | None (all outcome) | Yes — per solid-creating op |
| Pre-solid phase | Handled via format reward only | Handled via learned PRM (Method B) |
| Format/CoT reward | λ₂=0.9 chain-of-thought format | Not used (CadQuery doesn't have CoT) |
| Hard example handling | Guided injection (offline GT code skeleton) | Hard mining filter (mine_hard_examples.py) |
| Guidance type | Offline parameterized template | None (pure online RL) |
| Requires GT code | Yes (for guided injection) | No (only GT mesh needed) |
| Real-world scan compatible | No (guided injection needs GT code) | Yes (CD-based reward from mesh) |
| Visual similarity reward | DINOv2 (τ=0.55 threshold) | Not used (but could add as R_visual) |
| RL algorithm | GRPO G=8 | Dr. CPPO G=16, top-N=4 |
| Training scale | 8 A800, 359K RL samples | 8 H100, ~53K hard-mined samples |

### What ReCAD Teaches Us

1. **DINOv2 visual similarity as reward**: Effective for image-to-CAD tasks. Adds a visual consistency signal orthogonal to geometric IoU. Could be added to our reward for the `img` mode training samples: `R_visual = DINOv2_sim(render(pred_mesh), input_image)`.

2. **Format reward is powerful**: ReCAD uses λ₂=0.9 for CoT format, effectively using RL to teach *how to reason*, not just *what to output*. For CadQuery, the analog could be a code structure reward (enforcing variable naming, operation ordering conventions) — though less critical since CadQuery is already a constrained API.

3. **Learn Under Guidance vs. CAD-PRM**: These solve different problems:
   - Guided injection: helps when the model has the right knowledge but gets stuck on hard examples
   - CAD-PRM: helps with credit assignment — knowing *which* decision in the code was wrong
   - They are **complementary** and could be combined

4. **No step-level credit assignment remains open**: ReCAD explicitly does not address this. The CAD-PRM direction is therefore not preempted by ReCAD and remains novel.

5. **DINOv2 for OOD evaluation**: ReCAD shows that DINOv2 similarity correlates well with human visual judgment OOD (real scan results: 54.93% OOD CD improvement). This validates the visual similarity reward as a useful OOD generalization signal.

### Potential Additions to Our Reward

Based on ReCAD insights, consider adding to our hybrid reward formula:

```python
# Extended reward (img mode only):
R_total = λ₁ · R_outcome        # IoU × 10 (existing)
        + λ₂ · R_incremental     # partial solid IoU (Method A)
        + λ₃ · R_syntactic       # code structure (Method D)
        + λ₄ · R_PRM             # learned pre-solid predictor (Method B)
        + λ₅ · R_visual          # DINOv2(render(pred), input_image) — NEW from ReCAD

# For pc mode: λ₅ = 0 (no 2D image to compare against)
# For img mode: λ₅ = 0.1 (small but non-zero visual consistency signal)
```

---

## 9. Open Questions for Discussion

1. **Checkpoint frequency**: How often should we run incremental execution? Every solid-creating op (too frequent for simple shapes) or every N lines (simpler but arbitrary)?

2. **Discount factor γ**: Should later operations get more credit (γ < 1.0) or equal credit (γ = 1.0)? For CAD, the final `fillet()` is usually less important than the initial `extrude()`.

3. **PRM vs. dense reward directly**: Could we skip the PRM and instead just run partial executions at every `extrude()` as a fully rule-based dense reward? (This is Method A without any learned component.) Given compute budget, this might be simpler and equally effective.

4. **Invalid prefix handling**: When a prefix (e.g., first 50% of tokens) is syntactically incomplete (unclosed parentheses, etc.), should we: (a) complete the line before executing, (b) skip that checkpoint, or (c) assign a heuristic score?

5. **MC validation frequency**: How often should we run MC rollout estimation (Method C) to validate the PRM calibration? Every 2K RL steps seems reasonable.

6. **Add DINOv2 visual reward for img mode?** ReCAD shows this is effective. Cost: one DINOv2 forward pass per rollout (~5ms on GPU). Should we add `R_visual` for the 50% of training samples that use image input?

7. **Learn Under Guidance for our hardest examples?** Our hard mining filter (reward < 7.5) identifies ~53K hard examples. For the truly hardest (reward < 3.0, ~15% of RL set), should we inject a CadQuery code skeleton as a hint? Requires GT code (available for DeepCAD/Fusion360 synthetic data), not for CC3D real scans.
