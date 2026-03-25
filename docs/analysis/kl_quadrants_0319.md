# KL Quadrant Distribution Analysis — 2026-03-19

**Figure:** `kl_quadrants_0319.png`
**Script:** `tools/plot_kl_quadrants.py`
**Generated:** 2026-03-19

## Runs

| Run ID | Name | Steps | State |
|--------|------|-------|-------|
| [ms9sml9y](https://wandb.ai/hula-the-cat/cadrille-rl/runs/ms9sml9y) | rl-s3600-lr3e-5-G16-cppo-0319-1438 | 30 | finished |
| [nixqqhdd](https://wandb.ai/hula-the-cat/cadrille-rl/runs/nixqqhdd) | rl-s50k-lr1e-5-G16-cppo-0311-1946 | ~7000 | killed |
| [p0ui4ehg](https://wandb.ai/hula-the-cat/cadrille-rl/runs/p0ui4ehg) | rl-s50k-lr3.2e-6-G16-cppo-0318-0205 | ~2400 | crashed |

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

- **nixqqhdd / p0ui4ehg**: nn (gray) dominates throughout (~50%), meaning most KL
  comes from suppressing bad samples — healthy CPPO behavior. pp (red) holds steady
  at ~20–25%. np (yellow) present but not dominant.
- **ms9sml9y**: Only 3 steps logged. Erratic early dynamics — step 20 is nearly all nn,
  step 30 sees a spike in pp to ~60%. Too few points to draw conclusions.

## Reproducibility

```bash
uv run python tools/plot_kl_quadrants.py ms9sml9y nixqqhdd p0ui4ehg \
  --project cadrille-rl --smooth 5 --out docs/analysis/kl_quadrants_0319.png
```
