# 3D Similarity Metrics — Beyond IoU and Chamfer

## Why this exists

IoU and Chamfer Distance both punish **trivially-equivalent geometry**. The
canonical failure mode: a CAD chair predicted 5% taller than GT. Volumetric
intersection collapses (legs no longer overlap → IoU≈0); surface CD spikes;
but a human says "those are the same chair". Cad_bench_722's diversified
track exposes this constantly — Cadrille-rl's 84% rotation-win rate on the
IoU-24 rescore was a first symptom.

This folder collects metrics that **soften** that brittleness — by trading
some geometric strictness for tolerance to:

- small isotropic / anisotropic scaling
- rigid-body rotation that preserves shape identity
- minor topology jitter (sub-feature missing, edge added)
- different but visually-equivalent surfaces (curved vs faceted of same silhouette)

## Per-case metrics (pairwise GT vs PRED, can drop straight into eval)

| metric | type | what it captures | costs | implemented here |
|---|---|---|---:|---:|
| **IoU** (current baseline) | volumetric, geometric | exact volumetric overlap | trimesh boolean (~1-3s) | yes (in `common.metrics`) |
| **Chamfer Distance** (current) | surface, geometric | mean point-to-surface distance | 8k surface samples (<0.1s) | yes |
| **F-score @ τ** | surface, geometric, *threshold-tolerant* | fraction of pred-surface points within τ of GT (and vice versa) → harmonic mean | 8k samples + KDTree (~0.1s) | **yes (`geom_metrics.py`)** |
| **DINOv2 multi-view cos** | learned, *self-supervised visual* | cosine of DINOv2 [CLS] embeddings on 4-view collage renders. Captures geometric+textural similarity that humans agree with. | 1 ViT-S forward / image (~50ms on GPU) | **yes (`image_metrics.py`)** |
| **CLIP multi-view cos** | learned, *image-text-aligned* | cosine of CLIP image embeddings. Tracks "do these look like the same kind of object". More semantic, less geometric than DINOv2. | 1 CLIP-B/32 forward / image (~30ms) | **yes** |
| **LPIPS** | learned, *perceptual* | distance in pretrained image-classifier feature space (AlexNet/VGG). Captures perceptual difference, not classification. Lower = more similar. | 1 forward / image (~10ms) | **yes** |
| **SSIM** | pixel, structural | luminance + contrast + structure agreement at every patch. Bounded [-1, 1]. | numpy (<10ms) | **yes** |
| **PSNR** | pixel, RMS-error | log of pixel-MSE. Coarse but free baseline. | numpy (<1ms) | **yes** |

**Implementation note on rendering:** all *image* metrics use the same 4-view
268×268 yellow-mesh collage that `common.meshio.render_img` produces (and
that the BenchCAD upstream `composite_png` ships). This means: no
re-rendering of GT, and the pred image goes through the exact same
camera/material/lighting pipeline as GT — any image-distance signal is
about geometry, not render style.

## Distribution-level metrics (model vs model, NOT per-case)

These can't go in a per-case grid — they need a *set* of samples to be meaningful.
Useful as a **single number per model** in the headline report.

| metric | what it captures | what to compute on |
|---|---|---|
| **FID** | distance between feature distributions (Inception-V3 pool features) of `{pred renders}` vs `{GT renders}` | all 720 successful + GT pairs |
| **KID** | unbiased version of FID; better with small samples (<10k) | same |
| **CLIP R-Precision** | "given a pred image, can CLIP retrieve the right GT among N candidates?" — measures cross-sample discriminability | all 720 |

Recommend: compute FID + CLIP R-Precision once per model on the full 720, add to the
existing baselines table in `docs/cad_bench_722_baselines.md`. Not in this folder yet
because it's a separate tool surface (mostly bookkeeping, no new physics).

## Recommendation for the cad project

Pick **two** companions to IoU, not five — more is noise.

1. **F-score @ τ=0.05** as the *geometry-with-tolerance* number. Standard in
   3D recon literature (Tatarchenko et al. 2019), one-line trimesh implementation,
   directly answers the chair-height case (a 5%-scaled chair would still hit
   F~0.7-0.9 instead of IoU=0). Cheap.
2. **DINOv2 multi-view cosine** as the *visual semantics* number. Self-supervised
   so no caption requirement; captures "they look like the same thing"
   without being vibes-y like CLIP. ~50ms per pair on GPU.

LPIPS, SSIM, PSNR are nice-to-have; LPIPS in particular is the next-best learned
perceptual metric if DINOv2 is too heavy. CLIP is a decent semantic-only signal
but its bias toward text-grounded categories makes it less informative on
synthetic CAD parts (most cad_bench_722 family names — "ball_knob",
"clevis_pin" — are in CLIP's training distribution as words but with very
different visual contexts).

## What this folder produces

- `image_metrics.py` — pairwise CLIP / DINOv2 / LPIPS / SSIM / PSNR on PIL images
- `geom_metrics.py` — F-score @ τ on trimesh meshes
- `score_grid_cases.py` — evaluates the 12 cases × 3 models on the full set,
  writes per-case scores JSON + extends `eval_outputs/cad_bench_722/grid.png`
  with a per-case score table below.
