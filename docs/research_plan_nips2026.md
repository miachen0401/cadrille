# Research Plan: Cadrille-3D — NeurIPS 2026

**Deadline**: ~50 days (submission ~May 2026)
**Current baseline**: SFT img/DeepCAD=86.1%, target=92.2% (Dr. CPPO paper)
**Observation**: RL training improves pc IoU (+2pp @ ckpt-2100) but img IoU stagnates/regresses
**Core problem**: Model does not effectively use visual information — it relies on code patterns rather than reading the 3D geometry from images

---

## Executive Summary

The Cadrille paper treats CAD reconstruction as a language task conditioned on images/point clouds. Our analysis shows the model's multimodal grounding is weak: img and pc IoU diverge after RL, and img rewards barely improve despite correct reward signal. We propose three research directions:

1. **Better RL signal for multimodal grounding** — reward shaping that explicitly requires visual understanding
2. **3D-native representation** — go beyond 2D multi-view to true 3D inputs
3. **Iterative refinement** — multi-turn generation with visual feedback

Each direction is a standalone paper; combined they make a strong systems paper.

---

## Idea 1: Visual Render-and-Compare Reward (VRC-Reward)

### Motivation
Current reward = IoU(generated_mesh, GT_mesh). This only measures geometric accuracy but gives no gradient signal about *why* the image and the generated shape differ visually. The model can achieve a good reward by memorizing code patterns without understanding the image.

### Core Idea
After generating CadQuery code → execute → render 4-view image → compare with input:

```
R_total = α × R_iou + β × R_visual

R_visual = CLIP_cosine_sim(render(pred), input_views)
         OR DINOv2_cosine_sim(render(pred), input_views)
```

This creates a **direct visual feedback loop**: the model must generate shapes that *look like* the input images, not just match the GT mesh geometrically.

### Why it works for multimodal capability
- Forces the vision encoder to extract geometry-relevant features
- Even when IoU reward is sparse (code failed partially), visual reward provides gradient
- Naturally handles the "plausible alternative" problem: IoU penalizes correct-looking shapes that differ in internal structure, but visual reward rewards them appropriately

### Implementation
1. Maintain a **fast renderer** in reward workers (already have open3d Visualizer)
2. Load CLIP ViT-B/32 once in each reward worker (330 MB)
3. Cache rendered views per unique (code_hash, gt_mesh_path) pair
4. Blend: `α=0.7, β=0.3` (IoU is still primary; visual is regularizer)

### Ablation plan
| Config | α | β | Expected Effect |
|--------|---|---|-----------------|
| IoU only (baseline) | 1.0 | 0.0 | Current behavior |
| Visual only | 0.0 | 1.0 | Sanity check |
| 70/30 blend | 0.7 | 0.3 | Main experiment |
| 50/50 blend | 0.5 | 0.5 | Ablation |

### Novelty for NeurIPS
- First RL method for CAD generation that closes the visual feedback loop
- **"Render-and-Compare"** is a clear, publishable contribution
- Related work: DreamBooth3D, Score Distillation Sampling in diffusion; this is the RL equivalent for procedural CAD

---

## Idea 2: Multi-View 3D Consistency Reward (MV3D)

### Motivation
The 4 input views are fixed (front-left, back-right, back-left, front-right). The model could cheat by generating shapes that look correct from these 4 angles but are hollow/degenerate from other views. A consistency reward checks novel views.

### Core Idea
```
Generate code → execute → render 8 views (4 input + 4 novel random)
R_consistency = CLIP_sim(render_novel(pred), render_novel(GT))
```

The novel view reward cannot be gamed by memorizing the 4 fixed views — the model must learn *true 3D structure*.

Additionally, this enables **view augmentation during training**:
- Randomly rotate the 4 training views (±30° noise on camera angles)
- Forces the visual encoder to learn rotation-invariant features
- Implemented as a wrapper around `render_img()` in `rl/dataset.py`

### Implementation
```python
# In render_img(), add optional view augmentation:
if augment_views:
    fronts = [rotate_camera(f, np.random.uniform(-30, 30)) for f in BASE_FRONTS]
else:
    fronts = BASE_FRONTS
```

### Key experiments
1. **View augmentation only** (no consistency reward): measure effect on img generalization
2. **Novel view consistency reward**: measure multimodal grounding improvement
3. **Combined**: full system

### Novelty
- View augmentation for RL training is underexplored in code generation tasks
- Novel view consistency as a reward is novel for CAD reconstruction
- Could significantly improve Fusion360 generalization (complex shapes with hidden features)

---

## Idea 3: Point Cloud + Image Joint Grounding

### Motivation
The current model trains on either `img` or `pc` mode separately. But humans reason about 3D shapes by combining visual appearance AND geometric structure. Joint training on both modalities simultaneously could produce better 3D grounding.

### Core Idea
Train on **paired (img, pc) inputs for the same shape**:

```
Input: [image_views, point_cloud] → both visible to model simultaneously
Output: CadQuery code
Reward: IoU(generated, GT)
```

This requires:
1. The model can already handle both modalities (Cadrille architecture is bimodal)
2. Training with both signals simultaneously as a new modality: `train_modality: both`

### Dual-modality advantage estimate
In CPPO with G=16, generate rollouts with:
- G/2 = 8 from image input
- G/2 = 8 from pc input
- Compute advantages within each group separately
- The best of both guides the gradient

This is the **"multi-modal advantage"** — the model learns from whichever modality provides a clearer signal for each shape.

### Expected benefit
- For simple shapes: pc provides cleaner geometry signal
- For complex shapes: image provides appearance signal
- Joint training should generalize better than either alone

### Novelty
- Novel "multi-modal advantage" aggregation in GRPO/CPPO framework
- Extends Dr. CPPO to heterogeneous input spaces
- Can be framed as **"Modality-Agnostic RL for 3D Code Generation"**

---

## Idea 4: Chain-of-Thought 3D Reasoning (CoT-3D)

### Motivation
Current model: [image] → [CadQuery code directly]
A human CAD engineer would: [image] → [describe shape] → [plan operations] → [write code]

### Core Idea
Add a structured reasoning prefix before the CadQuery code:

```
<think>
Shape analysis:
- Main body: rectangular prism ~40mm × 30mm × 20mm
- Feature 1: cylindrical hole, diameter ~8mm, through-hole on top face
- Feature 2: chamfered edges on top
- Coordinate system: bottom-left corner at origin, extrude along Z

CadQuery plan:
1. Create base box: box(40, 30, 20)
2. Cut cylinder: cylinder(r=4, h=21) at center (20, 15)
3. Chamfer top edges: chamfer(1.0)
</think>
import cadquery as cq
r = (cq.Workplane("XY")
    .box(40, 30, 20)
    ...
)
```

### Training strategy
1. **Phase 1** (SFT warmup): Generate (shape_description, cad_code) pairs using GPT-4V for reasoning annotations on training shapes
2. **Phase 2** (RL with CoT): RL reward on final CadQuery code; thinking tokens get no direct reward but improve code generation through better planning

### Why this boosts multimodal capability
- Forces the model to explicitly describe visual features before generating code
- The `<think>` section creates an interpretable reasoning chain
- Shape description error → code error: gradient flows through description to visual encoder
- Failure analysis becomes trivial: read the thinking to see where it went wrong

### Data generation for SFT
```
For each (STL, GT_code) pair:
  1. Render 4 views of STL
  2. Call GPT-4V with: "Describe this 3D CAD shape and write a step-by-step plan for CadQuery"
  3. Save as (views, reasoning, code) triplet
  4. Fine-tune model on these triplets to teach it to reason first
```

### Novelty
- First work applying chain-of-thought reasoning to CAD code generation
- Combines visual grounding (describe shape) with structural planning (CadQuery operations)
- Strong NeurIPS paper angle: "Does reasoning help code generation for 3D tasks?"

---

## Idea 5: Online Curriculum via Self-Play Difficulty (Dr. CPPO+)

### Motivation
Current hard examples are statically mined at R_th=0.75 using the SFT model. After K steps of RL training, many "hard" examples are now "easy" for the improved model. Continuing to train on these wastes compute and reduces sample diversity.

### Core Idea: Dynamic Difficulty Score (DDS)

Every K=500 steps, run fast evaluation on the training set and update per-example difficulty:

```python
DDS(x, step) = 1 - pass@1(model_step, x)   # 1 = completely unsolved, 0 = always solved

Sampling weight:
  w(x) ∝ DDS(x) × (1 - DDS(x))             # bell curve: avoid too-easy AND too-hard
         (= variance of Bernoulli(pass@1))   # peaks at DDS=0.5
```

This implements **"Zone of Proximal Development"** sampling — the model learns fastest from examples it can sometimes solve but not always.

### Implementation
```
Every 500 steps:
  1. Sample 200 examples from current training pool (fast subset)
  2. Run greedy decode on each (no temperature, fast)
  3. Update difficulty score: EMA of (1 - success)
  4. Reweight sampling distribution
```

### Key difference from curriculum
- Curriculum: static phases based on SFT difficulty
- DDS: **dynamic** — adapts to model's current capability
- DDS automatically extends to new hard examples as model improves

### Paper contribution
This is a **standalone NeurIPS methods paper**: "Self-Paced Reinforcement Learning for Code Generation via Dynamic Difficulty Scheduling"

---

## Idea 6: 3D Backbone Integration (3D-Cadrille)

### Motivation
Multi-view images are a lossy representation of 3D structure. Points clouds are sparser but 3D. What if we used **actual 3D features** as input?

### Core Idea
Replace or augment the Qwen2-VL vision encoder with a **3D-aware encoder**:

**Option A: PointNet++ prefix**
```
Input: point cloud → PointNet++ → 3D feature tokens
       image views → ViT → 2D feature tokens
       Concatenate → LLM → CadQuery code
```

**Option B: 3D occupancy grid**
```
Input: multi-view images → NeRF-like implicit reconstruction → occupancy features
       → LLM → CadQuery code
```

**Option C: Sparse Voxel Transformer**
```
Input: point cloud → Sparse3D / MinkowskiNet → voxel features
       → LLM → CadQuery code
```

### Why this is the most powerful approach
- 3D features directly encode geometry that multi-view images lose (e.g., interior structure, exact dimensions)
- PointNet++ is lightweight (~10M params) and can be trained end-to-end
- Enables accurate reconstruction of complex internal geometries (threaded holes, internal channels)

### Training
1. Pre-train PointNet++ on ShapeNet shape classification (standard, fast)
2. Integrate with frozen LLM as additional token prefix
3. Fine-tune with RL using both 3D and 2D inputs

### Novelty
- First LLM for CAD generation that uses **native 3D representations**
- True 3D understanding vs. view-projection approximation
- "3D-native multimodal LLM for procedural 3D generation"

---

## Idea 7: Compositional Shape Decomposition Reward

### Motivation
CadQuery builds shapes as compositions of operations (extrude, cut, fillet). A high IoU doesn't mean the model learned the right operations — it might use a different sequence that happens to produce similar geometry. But for NeurIPS we care about *learning correct CAD primitives*.

### Core Idea: Operation-level intermediate rewards

Instead of only rewarding the final shape, also reward individual operation correctness:

```
Given GT code with ops: [box(40,30,20), cut(cylinder(r=4)), chamfer(1.0)]
Generated code with ops: [box(38,29,21), cut(cylinder(r=3.8)), fillet(0.9)]

R_intermediate = Σ_i IoU(execute(ops[0:i]), GT_partial_shape[0:i]) × weight_i
```

This requires:
1. Parsing CadQuery code into a sequence of operations
2. Computing partial IoU at each step
3. Weighting later operations more (they require understanding all prior operations)

### Step-level reward implementation
```python
def compute_step_rewards(code_str, gt_code_str, gt_mesh_path):
    ops = parse_cadquery_ops(code_str)
    gt_ops = parse_cadquery_ops(gt_code_str)

    rewards = []
    for i, op in enumerate(ops[:len(gt_ops)]):
        partial_mesh = execute_partial(ops[:i+1])
        gt_partial = execute_partial(gt_ops[:i+1])
        partial_iou = compute_iou(partial_mesh, gt_partial)
        rewards.append(partial_iou × (i+1)/len(gt_ops))  # later ops weighted more

    return sum(rewards) / len(rewards)
```

### Novelty
- Dense reward signal for procedural 3D generation
- First work that rewards *intermediate* CAD operations, not just final shape
- Addresses the "reward hacking" problem in CAD RL (many ways to get high IoU with wrong operations)

---

## Priority Matrix for 50 Days

| Idea | Impact | Effort | NeurIPS Novelty | Start Week |
|------|--------|--------|-----------------|-----------|
| **VRC-Reward** (Idea 1) | High | Medium | High | Week 1 |
| **View Augmentation** (Idea 2, easy part) | Medium | Low | Medium | Week 1 |
| **Online Curriculum / DDS** (Idea 5) | High | Medium | High | Week 2 |
| **CoT-3D** (Idea 4) | Very High | High | Very High | Week 2-3 |
| **Joint img+pc training** (Idea 3) | Medium | Low | Medium | Week 3 |
| **3D Backbone** (Idea 6) | Very High | Very High | Very High | Week 4+ |
| **Step-level reward** (Idea 7) | High | High | High | Week 4+ |

### Recommended 50-day plan

```
Week 1 (Days 1-7):
  - Fix LR + eps (1 day): restart with lr=3.2e-6 → 8e-6, monitor clip_frac
  - Implement VRC-Reward (3 days): CLIP visual similarity in reward worker
  - View augmentation (1 day): random ±30° camera perturbation in render_img()
  - Run ablation: IoU-only vs VRC-Reward on 5k steps each

Week 2 (Days 8-14):
  - Online curriculum / DDS (3 days): fast greedy eval every 500 steps, reweight
  - CoT-3D data generation (3 days): GPT-4V annotations on 1k training shapes
  - Start long A100 run with VRC-Reward + curriculum

Week 3 (Days 15-21):
  - CoT-3D SFT fine-tuning (3 days): fine-tune on (views, reasoning, code) triplets
  - CoT-3D RL training (ongoing)
  - Layer-wise LR decay for vision encoder (1 day)
  - Eval: compare CoT vs non-CoT at 5k steps

Week 4-5 (Days 22-35):
  - If VRC-Reward works: implement Idea 3 (joint img+pc)
  - If not: pivot to 3D backbone (PointNet++ prefix)
  - Start writing paper draft
  - Ablation experiments: contribution of each component

Week 6-7 (Days 36-50):
  - Full evaluation on DeepCAD + Fusion360 + CC3D
  - Paper writing + figures
  - Submission
```

---

## Paper Title Candidates

1. **"Visual Render-and-Compare: Closing the Feedback Loop in RL for CAD Generation"**
2. **"3D-Grounded RLHF: Multimodal Reward Shaping for Procedural CAD Code Generation"**
3. **"Beyond IoU: Visual Consistency Rewards for Multimodal CAD Reconstruction"**
4. **"Cadrille-3D: Multi-View Visual Feedback RL for Parametric CAD Generation"** (systems paper)
5. **"Reasoning Before Coding: Chain-of-Thought 3D Analysis for CAD Code Generation"** (if CoT works)

---

## Quick Wins (can be done this week)

1. **LR sweep**: run 3 configs (1e-6, 3.2e-6, 8e-6) for 500 steps each, compare clip_frac
2. **View augmentation**: 1 line change in render_img(), free performance boost
3. **Curriculum**: already implemented, just turn on `curriculum: true`
4. **eps_high=0.2**: already in h100.yaml, allows bigger updates
5. **Layer-wise LR**: 30 min to implement in train.py optimizer init

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| VRC-Reward CLIP too slow for reward workers | Medium | Use CLIP-tiny, cache renders |
| CoT data quality from GPT-4V poor | Low | Use only high-quality examples |
| 3D backbone training unstable | Medium | Start with frozen backbone, fine-tune gradually |
| No clear improvement from any method | Low | Multiple independent ideas, fallback to ablation paper |
| Not enough time for full experiments | Medium | Focus on 2 best ideas; ablation table shows others |

---

*Last updated: 2026-03-19*
*Status: Planning phase*
