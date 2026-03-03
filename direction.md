# NeurIPS 2026 Research Directions in CAD Reconstruction
**Deadline**: April 30, 2026 (full paper) | **Today**: Feb 28, 2026 | **Time**: ~2 months
**Constraint**: Limited compute (assume 2–4 A100/H100 GPUs max)
**Base**: Build on Cadrille (ICLR 2026) + CAD-Recode codebase

---

## Landscape Summary

| Paper | Venue | Gap Left |
|---|---|---|
| CAD-Recode | ICCV 2025 | Single-modal (PC only), sim-to-real gap |
| Cadrille | ICLR 2026 | Single-part only, no design intent, sparse RL reward |
| CADFusion | ICML 2025 | Text only, no geometric guarantee |
| CAD-Coder | NeurIPS 2025 | Text only, no real-world scans |
| CReFT-CAD | NeurIPS 2025 | Orthographic only, no closed-loop |
| GACO-CAD | arXiv Oct 2025 | No structured intent, conciseness only |
| CME-CAD | arXiv Dec 2025 | CoT heavy, expensive |
| ReCAD | arXiv Dec 2025 | Image-only (no PC), custom API (not CadQuery/B-Rep), outcome reward only, no step-level credit |

**Open gaps as of Feb 2026 (confirmed by literature):**
- No assembly-level reconstruction (all methods: single part)
- No design-intent / parametric editability guarantee
- No interactive iterative refinement loop
- Reward sparsity problem in RL (execute-or-nothing) — ReCAD partially addresses with DINOv2 visual similarity + format rewards, but no step-level credit assignment
- Real-world scan robustness remains poor (CC3D IoU ~67%)
- ReCAD's "Learn Under Guidance" injects code hints for hard examples but is offline, not a learned PRM

---

## Direction 1: Process Reward Models for CAD Code RL

### Motivation
All RL-based CAD methods (Cadrille, CAD-Coder, GACO-CAD, ReCAD) use **outcome rewards**: the program either executes and produces geometry (dense reward from IoU) or fails (−10 penalty). This is fundamentally sparse: every failed execution step gives zero signal for where the error occurred — a mismatched parenthesis on line 2 penalizes the same as a wrong extrusion on line 50. This causes:
- Slow RL convergence (most rollouts are invalid early in training)
- Poor exploration (model stays near SFT distribution)
- No credit assignment within the code

**Key insight**: Process Reward Models (PRMs), which assign rewards to *intermediate reasoning steps*, dramatically improve LLM RL in math (Lightman et al. 2023). CAD Python code has **natural intermediate checkpoints**: each CadQuery statement (`Workplane`, `circle`, `extrude`, `fillet`, etc.) can be executed incrementally. We can build a **CAD Process Reward Model (CAD-PRM)** that evaluates intermediate geometry at each statement checkpoint.

### Novelty vs. Prior Work
- Cadrille: outcome reward only (IoU of final mesh + invalidity penalty)
- GACO-CAD: adds conciseness reward (group length) but still outcome-level
- CAD-Coder: CoT decomposition is a heuristic, not a learned PRM
- ReCAD (arXiv Dec 2025): introduces two richer outcome signals — DINOv2 visual similarity reward (min(IoU, DINOv2_sim) with τ=0.55 threshold) and a CoT format reward (λ₁=0.1 geometric + λ₂=0.9 format). Also uses "Learn Under Guidance": for hard questions (reward < 0.8), injects parameterized code as in-context hint during RL rollouts. These are still **outcome-level** signals (no step credit); the guided hint is an offline heuristic, not a learned per-step model.
- **Ours**: first *learned* step-level reward model for CAD code generation — assigns credit at each CadQuery solid-creating operation, not just at final execution. Complements ReCAD's multi-signal approach by adding within-sequence density.

### Method
1. **Incremental Execution Engine**: Instrument CadQuery to checkpoint after each statement. Compute partial IoU against a voxelized ground truth at each checkpoint using Monte Carlo sampling of intermediate solid.
2. **PRM Training Data**: Sample 5K rollouts from the Cadrille SFT model on DeepCAD. For each rollout, record the per-statement geometry evolution and label each step with partial-geometry reward (partial Chamfer Distance to ground truth).
3. **CAD-PRM Architecture**: Finetune a small (0.5B) language model on `(code prefix, step label)` pairs via regression. This model takes a partial code and returns a quality score 0–1.
4. **GRPO + PRM Integration**: Replace Cadrille's outcome reward `R(τ)` with a weighted sum: `R(τ) = α·R_outcome(τ) + (1−α)·∑_k R_PRM(τ_{1:k})`. Use `α=0.7` with curriculum: start `α=1.0` (outcome only), anneal to `α=0.5` after convergence.
5. **Evaluation**: DeepCAD (IoU, CD, IR), Fusion360, CC3D. Compare convergence speed (steps to 90% of final IoU) and final IoU.

### Implementation Plan (8 weeks)
- **Week 1–2**: Implement incremental CadQuery execution engine. Verify partial geometry extraction. Generate 5K labeled rollouts.
- **Week 3**: Train CAD-PRM (0.5B LM finetune, ~4 A100 hours).
- **Week 4–5**: Integrate CAD-PRM into Cadrille's RL loop. Run ablation: outcome-only vs. PRM-only vs. hybrid.
- **Week 6**: Full training run on DeepCAD + Fusion360 RL data (≤ 7 days on 2 GPUs).
- **Week 7–8**: Evaluation + writing.

### Compute Budget
- PRM data generation: ~10 GPU-hours (inference only)
- PRM training: ~8 GPU-hours (0.5B model)
- RL training: 2–3 days × 2 A100 = ~120 GPU-hours total
- **Total: ~140 GPU-hours ≈ 6 days on 2 A100s**

### Expected Results
- 3–5% IoU improvement on DeepCAD (from 90.2% → ~93–95%)
- 2× faster convergence (fewer RL steps needed)
- Most impactful on CC3D (real-world) where outcome reward is sparse due to domain gap

### NeurIPS Pitch
*"We show that dense process-level rewards, derived from incremental CAD execution, dramatically improve RL convergence and final reconstruction quality — establishing the first Process Reward Model for structured code generation in geometry tasks."*

---

## Direction 2: Assembly-Level CAD Reconstruction

### Motivation
Every existing CAD reconstruction method (CAD-Recode, Cadrille, CADCrafter, CME-CAD, ...) operates on **single-part models**. Real-world engineering products are assemblies: a car engine has >200 parts, a LEGO brick has mating constraints, a cabinet has joints. Reconstructing an assembly from a point cloud or images of a scanned physical product is a completely open problem with enormous industrial value (digital twin creation, reverse engineering, design automation).

The gap is not merely scaling: assemblies require **spatial constraint reasoning** (part X is bolted to part Y, part Z slides along axis A) that current LLMs have no mechanism to produce. CadQuery natively supports assemblies via `cq.Assembly()`, but no dataset, method, or benchmark for assembly-level code generation exists.

### Novelty vs. Prior Work
- No prior work addresses assembly-level CAD reconstruction from scans
- AIDL (arXiv Feb 2025) uses a DSL with a constraint solver but is text-only and limited to simple geometry; no scan input
- CMT (ICCV 2025) generates B-Rep topology but for single parts

### Method
1. **Synthetic Assembly Dataset (ProceduralAssembly-100K)**: Write a procedural generator (extending the CAD-Recode generator) that creates multi-part assemblies: (a) generate N=2–5 single parts using existing generator, (b) place them with deterministic spatial relationships (butt joint, T-joint, concentric, face-to-face), (c) write CadQuery Assembly code with explicit constraint expressions.
   - Dataset size: 100K assemblies (2–5 parts each) — full generation in ~2 days on CPU.
2. **Point Cloud Segmentation Pre-Step**: Use an off-the-shelf 3D instance segmentation model (Mask3D, SA3D, or SAM3D) to segment the input point cloud into part-level clusters. Feed each cluster independently to Cadrille-SFT to reconstruct individual parts.
3. **Assembly LM**: Train a small (~1B param) LM to predict the Assembly block given: (a) individual part CadQuery codes from step 2, (b) relative 6-DoF pose estimates between parts (from ICP between part point clouds). Output: valid `cq.Assembly(...)` Python code with constraint declarations.
4. **Self-Consistency RL**: Use a novel reward: execute the assembly code, extract the assembled mesh, and compute Chamfer Distance against the original (unsegmented) point cloud. This reward is computable without any GT assembly annotation.
5. **Benchmark**: Introduce ProceduralAssembly-1K test set (held out), plus 50 hand-annotated real-world scans from ABC dataset for qualitative evaluation.

### Implementation Plan (8 weeks)
- **Week 1**: Build procedural assembly generator. Generate 100K training + 1K test assemblies.
- **Week 2**: Set up Mask3D-based segmentation pipeline. Measure part segmentation quality on synthetic data.
- **Week 3–4**: Fine-tune Cadrille for per-part reconstruction (use existing Cadrille weights + brief LoRA finetune on assembly part instances).
- **Week 5**: Train Assembly LM (1B model on 100K samples, ~48 GPU-hours).
- **Week 6**: RL fine-tuning with self-consistency reward on ~10K assemblies.
- **Week 7–8**: Evaluation + ablation + writing.

### Compute Budget
- Dataset generation: CPU only, ~2 days
- Assembly LM training: ~2 days × 2 A100 = ~96 GPU-hours
- RL fine-tuning: ~1 day × 2 A100 = ~48 GPU-hours
- **Total: ~150 GPU-hours ≈ 6 days on 2 A100s**

### Expected Results
- First published results on assembly reconstruction
- IoU on ProceduralAssembly test: ~65–75% (single-part methods inapplicable)
- Novel benchmark that will be adopted by the community

### NeurIPS Pitch
*"We introduce the first method and benchmark for assembly-level CAD reconstruction from point clouds, extending the CadQuery code paradigm to multi-part assemblies with spatial constraints. Our self-consistency reward enables annotation-free RL fine-tuning."*

---

## Direction 3: Iterative CAD Refinement via Multi-Turn RL

### Motivation
Current CAD reconstruction methods are **one-shot**: given an input, produce code. This mirrors how current LLMs work but not how engineers work: real CAD design is iterative — you sketch, check, adjust dimensions, re-check topology, adjust again. When the first reconstruction is wrong (e.g., a hole is in the wrong position), the user has no way to correct it without starting over.

An *interactive* CAD system would accept natural language feedback ("make the hole 2mm larger and shift it 5mm to the left") and update the Python code accordingly. This is a multi-turn **refinement** problem. Critically, this requires the model to:
1. Understand the *current* CAD code state
2. Interpret geometric feedback (natural language or visual diff)
3. Edit the code minimally (not regenerate from scratch)

No existing method addresses this. CAD-Assistant (ICCV 2025) attempts zero-shot tool use with FreeCAD, but produces fragile outputs and relies on GPT-4-level models. We propose a trained, lightweight refinement model.

### Novelty vs. Prior Work
- CAD-Assistant: zero-shot, no training, fragile, requires GPT-4
- ReCAD (arXiv Dec 2025): single-turn image+text to CAD; no iterative refinement
- Text2CAD / CAD-Coder: single-turn text to CAD
- **Ours**: first *trained* iterative CAD refinement model with multi-turn RL

### Method
1. **Refinement Dataset Generation**: Given existing CAD-Recode dataset (1M programs):
   - For each program τ*, sample a "degraded" prediction τ̃ (from Cadrille SFT at temperature 1.0)
   - Use GPT-4o-mini to generate a natural language description of the geometric difference between τ̃ and τ* (e.g., "the top face is missing a fillet," "the extrusion height is 3× too large")
   - Create triplets: (τ̃, feedback_text, τ*). Cost: ~$200 for 50K triplets using GPT-4o-mini
2. **Architecture**: Extend Cadrille to accept a "current code + feedback" input prompt in addition to the original modality (point cloud/image). The model outputs a corrected code. Finetune with LoRA.
3. **Multi-Turn RL**: Simulate iterative refinement loops using the model itself:
   - Start from Cadrille SFT prediction τ₀
   - At each turn t: compute reward R(τ_t), generate difference description (programmatically from geometry diff), feed as next-turn input
   - Use GRPO with per-turn reward; incentivize convergence in ≤3 turns
4. **Geometric Diff Oracle**: Instead of GPT-4o-mini at inference, train a small `GeoDiff` model (classify which face/region changed and by how much) using synthetic perturbations. This makes the system closed-loop at inference.
5. **Benchmark**: Introduce CADRefine-500 (500 test cases with 3-turn human correction sequences collected from 3 participants using a simple web UI).

### Implementation Plan (8 weeks)
- **Week 1**: Build programmatic geometry-diff descriptor (compare mesh face areas, volumes, topology changes). Validate it correlates with human descriptions.
- **Week 2**: Generate 50K refinement triplets (GPT-4o-mini API, ~$200 cost). Fine-tune Cadrille with LoRA on single-turn refinement (~24 GPU-hours).
- **Week 3–4**: Implement multi-turn RL loop. Run 5K simulated refinement trajectories. Train GRPO for 3-turn convergence.
- **Week 5**: Collect CADRefine-500 human benchmark (3 participants × 3 hours = minimal effort).
- **Week 6**: Evaluation: single-turn refinement accuracy, 3-turn convergence, comparison to zero-shot GPT-4o and CAD-Assistant.
- **Week 7–8**: Writing.

### Compute Budget
- SFT fine-tune (LoRA): ~24 GPU-hours
- Multi-turn RL: ~2 days × 2 A100 = ~96 GPU-hours
- GPT-4o-mini API: ~$200
- **Total: ~130 GPU-hours ≈ 5.5 days on 2 A100s**

### Expected Results
- After 3 refinement turns: IoU improves from ~90% (single-shot) to ~95% on DeepCAD
- Human study: users achieve 85%+ satisfaction vs. 45% for one-shot
- Strong baseline story: zero-shot GPT-4o does poorly; trained model is necessary

### NeurIPS Pitch
*"We introduce iterative multi-turn CAD refinement via RL, demonstrating that a compact trained model can surpass zero-shot GPT-4 in closed-loop correction — enabling practical interactive reverse engineering."*

---

## Direction 4: Structured CAD Code with Parametric Editability (Design-Intent RL)

### Motivation
A critical but understated failure mode of all current CAD code generation methods: the generated Python code reconstructs the *shape* but destroys the *design intent*. Consider a bracket with 4 symmetric bolt holes. A human CAD model would encode this as `for angle in [0, 90, 180, 270]: add_hole_at(angle)`. The model might instead generate 4 independent `circle()` calls with magic-number coordinates. Both produce the same mesh — but only the former is parametrically editable (change hole count → change one number).

This means generated CAD files are practically unusable for engineering modification, despite geometric correctness. The community has not addressed this because: (a) the standard benchmarks (DeepCAD, Fusion360) evaluate only geometric metrics (IoU, CD, IR), (b) there is no automated metric for parametric quality.

### Novelty vs. Prior Work
- All prior work measures IoU/CD/IR; none measures design intent / editability
- CADmium introduces topological metrics (Euler characteristic, sphericity) but not parametric quality
- **Ours**: first automated metric + RL reward for parametric editability; first method to optimize it

### Method
1. **Parametric Editability Score (PES)**: Define automated metrics:
   - *Symmetry consistency*: detect geometric symmetry in the reconstructed mesh; check if the code uses loops/arrays vs. repetitive constants
   - *Dimension parameterization*: what fraction of numeric constants in the code correspond to named variables (improves with variable extraction)?
   - *Refactorability*: does removing one line of code still produce a valid (if simpler) shape? (Measures over-specification)
   - *Edit sensitivity*: perturb a top-level variable by ±10%; how much does the mesh change?
2. **PES-Guided Code Generation**: Add PES as an auxiliary RL reward term alongside IoU. Key: PES is computed from the output code, not the mesh — no additional CAD execution needed.
3. **Structured Code Pretraining**: Generate a "style-regularized" dataset by rewriting the CAD-Recode dataset using GPT-4o-mini to introduce loops, variables, and symmetry annotations (cost: ~$300 for 20K rewrites). Fine-tune Cadrille with LoRA on this structured dataset first.
4. **PES-GRPO**: Integrate PES into the GRPO reward: `R(τ) = λ·IoU(τ) + (1−λ)·PES(τ)`. Anneal `λ` from 1.0 → 0.7 over training.
5. **Human Evaluation**: 20 engineers asked to "make a change to this CAD file." Measure: time to make change, success rate for 5 predefined modifications. Compare original Cadrille vs. PES-optimized code.

### Implementation Plan (8 weeks)
- **Week 1**: Implement PES metric suite. Validate on 100 human-annotated CAD programs (10 hours of annotation).
- **Week 2**: GPT-4o-mini rewriting of 20K CAD-Recode programs (~$300). LoRA fine-tune on structured dataset (~24 GPU-hours).
- **Week 3–4**: Implement PES reward computation in RL training loop. Run ablations on PES weight λ.
- **Week 5**: Full GRPO + PES training run on DeepCAD (2 days × 2 GPUs).
- **Week 6**: Human evaluation study (recruit engineers via university contacts, 20 participants).
- **Week 7–8**: Writing + ablations.

### Compute Budget
- LoRA SFT: ~24 GPU-hours
- GRPO training: ~2 days × 2 A100 = ~96 GPU-hours
- GPT-4o-mini API: ~$300
- **Total: ~130 GPU-hours ≈ 5.5 days on 2 A100s**

### Expected Results
- PES improves from baseline (random score) to ~0.65 (vs. human-authored: 0.85)
- No significant drop in IoU/CD (within 1%)
- Human study: 2× faster modification time on PES-optimized code
- New benchmark metric adopted by community

### NeurIPS Pitch
*"We define and optimize design-intent editability in LLM-generated CAD code, introducing the first Parametric Editability Score and demonstrating that RL can jointly optimize geometric fidelity and downstream modifiability — bridging the gap between reconstruction accuracy and engineering utility."*

---

## Direction 5: Noise-Robust CAD Reconstruction from Real-World Scans via Geometry-Aware Data Augmentation and Domain RL

### Motivation
The sim-to-real gap is the central unsolved problem in CAD reconstruction. Cadrille achieves 90%+ IoU on the clean synthetic DeepCAD but only 67.9% on CC3D (real-world scans). CAD-Recode achieves 60.5% IoU on CC3D. The gap is due to real scans having:
- Missing data (occluded surfaces → holes in point cloud)
- Non-uniform density (closer surfaces → denser sampling)
- Measurement noise (Gaussian + salt-and-pepper)
- Smoothed sharp edges (scanner aperture effect)
- Outlier points (multi-path reflections)

Current approaches handle this implicitly (RL fine-tuning on CC3D with real data) but there is no principled noise model or augmentation strategy specifically designed for CAD scan characteristics. Unlike natural 3D scenes, CAD scans have *piecewise smooth* surfaces with sharp feature curves — a domain-specific prior that existing augmentation strategies ignore.

### Novelty vs. Prior Work
- Cadrille: RL on DeepCAD + Fusion360 meshes improves CC3D to 67.9% IoU; no explicit noise modeling
- CAD-Recode: test-time sampling (10× inference) improves CC3D from 60.5 → 74%; expensive
- **Ours**: (a) CAD-specific noise augmentation for training, (b) geometry-aware feature extraction (sharp edge detection), (c) domain-adaptive RL using real CC3D scans without GT CAD models

### Method
1. **CAD Scan Noise Model**: Analyze CC3D point clouds statistically. Build a parametric noise model with 5 components:
   - Gaussian noise (σ sampled from Beta distribution fit to CC3D)
   - Missing region simulation (random convex-hull subtraction)
   - Density variation (distance-dependent subsampling)
   - Edge smoothing (Gaussian blur along surface normal)
   - Outlier contamination (Poisson-process outliers at 1–3%)
2. **Augmented SFT**: Re-train the point cloud encoder of Cadrille with aggressive augmentation using the above noise model on the CAD-Recode 1M dataset. The LLM backbone is frozen (LoRA only); only the point cloud encoder and projection layer are updated. This is extremely compute-efficient.
3. **Sharp-Edge Aware Encoding**: Add a Sharp-Edge Detection module (Edge Conditioned Convolution or simple curvature estimation) that detects CAD feature lines in the (noisy) point cloud and provides them as auxiliary tokens to the LLM. This gives the model access to topological priors about CAD geometry.
4. **Domain RL on CC3D**: Use CC3D training split (2000 scans, no GT CAD codes) for RL fine-tuning. The reward is computed by comparing the executed CAD model mesh against the CC3D scan using Chamfer Distance (point cloud to mesh).
5. **Consistency Regularization**: Add a consistency reward: for the same object under different augmentations (noise levels), the model should produce similar codes. This penalizes over-fitting to noise artifacts.

### Implementation Plan (8 weeks)
- **Week 1**: Statistical analysis of CC3D point clouds. Implement parametric noise simulator. Validate it reproduces CC3D distribution (FID-like metric on point cloud statistics).
- **Week 2**: Re-train Cadrille point encoder with noise augmentation (LoRA, ~32 GPU-hours).
- **Week 3**: Implement sharp-edge detection module. Integrate as auxiliary input tokens. Fine-tune end-to-end on DeepCAD + augmented CAD-Recode (~48 GPU-hours).
- **Week 4–5**: Domain RL on CC3D training split. Run GRPO with CD-based reward. Compare: baseline Cadrille RL vs. +noise aug vs. +edge module vs. full.
- **Week 6**: Ablation experiments. Evaluate on CC3D test, Fusion360 (also has real scan characteristics).
- **Week 7–8**: Writing.

### Compute Budget
- Point encoder re-training: ~32 GPU-hours
- Full SFT: ~48 GPU-hours
- Domain RL: ~2 days × 2 A100 = ~96 GPU-hours
- **Total: ~180 GPU-hours ≈ 7.5 days on 2 A100s**

### Expected Results
- CC3D IoU: from 67.9% (Cadrille) → ~78–80%
- CC3D IR: from 0.2% → near 0%
- DeepCAD maintained at 90%+ IoU (no regression)
- Novel: first explicit CAD-specific scan noise model + domain adaptation method

### NeurIPS Pitch
*"We show that a principled CAD-specific noise model, combined with sharp-edge-aware encoding and domain RL, closes 50% of the sim-to-real gap for CAD reconstruction — enabling practical reverse engineering from commodity scanners."*

---

## Comparison and Recommendation

| | Impact | Novelty | Feasibility | Compute | Recommendation |
|---|---|---|---|---|---|
| **Dir 1: Process Reward (CAD-PRM)** | High | Very High | High | ~140 GPU-h | **Start immediately** |
| **Dir 2: Assembly-Level CAD** | Very High | Extreme | Medium | ~150 GPU-h | Best long-term bet |
| **Dir 3: Iterative Refinement** | High | High | High | ~130 GPU-h | Good if Dir 1 succeeds fast |
| **Dir 4: Parametric Editability** | High | High | Medium | ~130 GPU-h | Strong human-study angle |
| **Dir 5: Noise-Robust Real Scans** | Very High | High | High | ~180 GPU-h | Clearest metric improvement |

**Recommended priority order for Apr 30 deadline:**
1. **Direction 1 (CAD-PRM)** — purely algorithmic, fastest to prototype, clear story
2. **Direction 5 (Noise-Robust)** — clear metric improvement, CC3D is a recognized benchmark
3. **Direction 2 (Assembly)** — highest impact but needs dataset generation; start procedural generator in parallel

**Avoid** Direction 3 (needs human annotation collection) and Direction 4 (needs engineer participants) unless you already have access to human evaluators — the logistical overhead may not fit 2 months.

---

## Shared Infrastructure Needed (Week 1, all directions)

```bash
# Install and validate CadQuery incremental execution
uv pip install cadquery

# Set up CAD rendering pipeline (for reward computation)
# - Point cloud sampling from B-Rep
# - Voxelization for IoU

# Clone and adapt Cadrille training loop
git clone https://github.com/col14m/cadrille
# Key files to modify:
#   reward.py       - add PRM/PES/noise reward
#   rl_train.py     - add multi-turn support
#   data/           - add assembly/noise generators
```

**Timeline Overview (2 months to April 30):**
```
Week 1-2:  Prototype + data generation
Week 3-4:  Training runs + ablations
Week 5-6:  Full experiments + evaluations
Week 7-8:  Paper writing + polishing
```
