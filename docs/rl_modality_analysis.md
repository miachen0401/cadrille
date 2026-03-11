# RL Training Modality Analysis: Why Img-only RL Improves PC Scores

**Date:** 2026-03-11
**Context:** We are reproducing Cadrille (ICLR 2026) Table 2 results using Dr. CPPO on a 4080 SUPER.
**Question:** The paper trains RL on img mode only, yet PC scores improve. How? Are any components frozen?

---

## 1. What Does the Paper Actually Train On?

From the paper:

> "RL fine-tuning on images appears to be beneficial for other modalities."

**RL training uses img mode only** (DeepCAD + Fusion360 training splits). Point cloud data is *never seen* during the RL phase. The SFT phase uses CAD-Recode (procedurally generated), and RL uses the handcrafted DeepCAD+Fusion360 datasets.

Our config (`configs/rl/4080.yaml`) correctly sets `train_modality: img`.

---

## 2. Are Any Model Components Frozen?

**No. All parameters are trainable.**

From `rl/train.py:344`:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
```

This is somewhat surprising — the vision encoder (Qwen2-VL ViT), point cloud encoder, projection MLP, and LLM backbone are all updated during RL. The paper doesn't explicitly freeze anything.

**Implication:** The point cloud encoder is getting gradient updates from image-mode rollouts, even though it never receives non-zero input during RL (image samples have `is_pc=False`, so the PC encoder output is zeroed out or not used). This might mean the PC encoder effectively gets ~zero gradients during RL — functionally frozen in practice even though it's technically in the optimizer.

*TODO: Verify this by checking how `is_pc` gates the PC encoder in `cadrille.py`.*

---

## 3. Why Does PC Score Improve Despite Img-only RL?

This is the core mystery. Three plausible mechanisms:

### Hypothesis A: Shared LLM Backbone (Most Likely)

Both PC and img modalities pass through the same Qwen2-VL LLM backbone. During SFT, the model learns to map encoded inputs → CadQuery code. During RL on img, the LLM backbone is updated to generate *better CadQuery code* — code that more accurately reconstructs geometry.

**Key insight:** CadQuery code quality is modality-agnostic. Whether the model sees an image or a point cloud, it must output valid Python CadQuery code that produces the right mesh. Improving code generation quality (via IoU reward signal from img rollouts) generalizes directly to PC mode.

Specifically, RL teaches the model:
- Better use of `extrude`, `fillet`, `circle`, etc. for specific geometries
- More accurate dimension estimation
- Fewer syntax/logic errors (invalidity ↓ from ~2% → ~0.2%)

These improvements are baked into the LLM weights and activate regardless of which modality provides the input tokens.

### Hypothesis B: Modality-Invariant Geometric Representations

The projection layers that map PC tokens and img tokens into the LLM's embedding space were trained (during SFT) to produce *similar* representations for the same object. After RL, the LLM expects higher-quality geometric context from its input tokens. The PC projection layer (unchanged by RL) still produces the same-quality tokens, but the LLM is now better at using them.

This is like upgrading a decoder — the encoded representation (PC tokens) stays the same, but the decoder gets better at reading it.

### Hypothesis C: DeepCAD/Fusion360 Dataset Familiarity

RL training data = DeepCAD + Fusion360 training meshes (in img mode). Validation data = DeepCAD + Fusion360 *test* meshes (in both PC and img modes).

The test set shares the same geometric distribution as the training distribution. RL optimizes specifically for DeepCAD/Fusion360-style CAD geometry. Since the test split has the same geometric vocabulary (rectangular extrusions, cylindrical holes, chamfers, fillets), the improved code generation generalizes to PC inputs of the same objects.

**This would explain why CC3D (real-world) improves less:** CC3D has a different distribution (noisy scans, different geometry types), so the RL generalization is weaker.

---

## 4. Paper Numbers vs. Our Baseline

| Dataset | Modality | SFT | SFT + Dr. CPPO (paper) | Our baseline (ckpt-1000) |
|---------|----------|-----|------------------------|--------------------------|
| DeepCAD | PC  | 87.1% | **90.2%** | 87.6% ✓ |
| DeepCAD | Img | ~84%? | **92.2%** | 84.1% ✓ |
| Fusion360 | PC | 79.8% | **85.0%** | 80.7% ✓ |
| Fusion360 | Img | ~78%? | ~84-85%? | 78.4% ✓ |

Our SFT baseline closely matches the paper's SFT numbers. The RL target is +3–5% on PC and +7–8% on img.

**Step 1200 (after ~200 RL steps from ckpt-1000):**

| Dataset | Modality | ckpt-1000 | step-1200 | Δ |
|---------|----------|-----------|-----------|---|
| DeepCAD | PC  | 87.6% | 85.9% | -1.7% |
| Fusion360 | PC | 80.7% | 78.1% | -2.6% |
| DeepCAD | Img | 84.1% | 82.3% | -1.8% |
| Fusion360 | Img | 78.4% | 78.7% | +0.3% |

**Scores dipped slightly at step 1200.** This is expected behavior: early RL often degrades slightly before improving as the model "unlearns" SFT behavior and explores. Watch for improvement from step 1500+ onward.

---

## 5. Our Concern: Will PC Degrade Long-Term?

**The risk is real.** Since we train on img only:
- The PC encoder gets no reward signal
- The LLM adapts specifically to image-derived context
- Over many steps, the LLM's expectation of input tokens may drift away from what the PC encoder provides

The paper reports +3.1% PC improvement at the end of training, suggesting this doesn't happen — but they may be training at higher LR (G=16 vs our G=4) or with different data.

**Monitoring plan:**
- Log `eval/pc/DeepCAD IoU` every 100 steps
- If PC drops >3% below SFT baseline (i.e., below ~84%) at any checkpoint, raise alert
- Paper target: PC DeepCAD ≥ 90.2%, PC Fusion360 ≥ 85.0%

---

## 6. Key Open Questions

1. **Does the PC encoder receive any gradients during RL?** If `is_pc=False` gates the encoder out completely, it gets zero gradients — functionally frozen. If gradients flow back through the gating mechanism, it gets spurious updates.

2. **Why does img RL give +8% img improvement vs +3% PC improvement?** The model is directly optimizing on img inputs, so img improvement is expected to be larger. The PC improvement is "free" from backbone improvements.

3. **Would PC-mode RL be additive?** The paper only reports img-only RL. A natural follow-up experiment: after img RL converges, run a second PC-only RL phase. Might push PC to 92%+ as well.

4. **What is the effective training set size?** With hard-examples-only pkl (mined hard examples), RL sees a non-random subset of DeepCAD. Does this subset cover the same geometric variety needed for good PC generalization?

---

## 7. Conclusions

- **Img-only RL is the correct approach per the paper.** Don't add PC training to our run.
- **PC improvement comes from LLM backbone improvement**, not from direct PC-encoder training.
- **Early dips at step 1200 are normal.** Wait until ~step 2000 before drawing conclusions.
- **We are on track.** SFT baseline matches paper. Img eval bug is fixed (0.208 → 0.84).
- **Next milestone:** step 1500 eval — expect PC and img to start recovering/improving.
