# KL Quadrant Distribution Analysis — 2026-03-20

**Figure:** `kl_quadrants_0320.png`
**Script:** `tools/plot_kl_quadrants.py`
**Generated:** 2026-03-20

## Runs

| Run ID | Name | Steps | State |
|--------|------|-------|-------|
| [ecevm1vi](https://wandb.ai/hula-the-cat/cadrille-rl/runs/ecevm1vi) | rl-s3600-lr1e-5-G16-cppo-0320-0531 | ~75 | running |
| [s91its3w](https://wandb.ai/hula-the-cat/cadrille-rl/runs/s91its3w) | rl-s3600-lr1e-5-G16-cppo-0320-0524 | ~76 | running |
| [ymtm3hwo](https://wandb.ai/hula-the-cat/cadrille-rl/runs/ymtm3hwo) | rl-s3600-lr2e-5-G16-cppo-0320-0313 | ~48 | running |

## Quadrant Definitions

Each step's KL is partitioned by (advantage sign) × (ratio sign):

| Quadrant | adv | ratio | Meaning | Color |
|----------|-----|-------|---------|-------|
| pp | + | >1 | Model ↑ prob on good samples — ideal | Red |
| np | − | >1 | Model ↑ prob on bad samples — harmful | Yellow |
| pn | + | ≤1 | Model ↓ prob on good samples — bad | Blue |
| nn | − | ≤1 | Model ↓ prob on bad samples — benign | Gray |

Values are softmax-normalized per step so they sum to 100%.

## Observations

- **ecevm1vi / s91its3w (lr=1e-5)**: Nearly identical distributions — pp ~25%, np ~25%, pn ~20%, nn ~30%. Stable throughout. The np (yellow, harmful) share is concerningly high at ~25%, on par with pp (ideal).
- **ymtm3hwo (lr=2e-5)**: Slightly better profile — pp ~30%, np ~20%, pn ~20%, nn ~30%. Higher lr yields a larger ideal pp share and lower harmful np share, though the difference is modest.
- **Contrast with 0319**: Yesterday's longer runs (nixqqhdd, p0ui4ehg at 7k+ steps) showed nn dominating (~50%) with pp at ~20–25% and np subdued. These early-stage 0320 runs show a more balanced (less healthy) distribution — this may reflect early training instability before CPPO's clipping kicks in fully. Worth re-checking at 200+ steps.

## Reproducibility

```bash
uv run python tools/plot_kl_quadrants.py ecevm1vi s91its3w ymtm3hwo \
  --project cadrille-rl --smooth 3 --out docs/analysis/kl_quadrants_0320.png
```
