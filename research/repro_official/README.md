# Official cadrille reproduction (transformers 4.50.3)

## Why this exists

The project's main eval pipeline (`eval/bench.py`, `eval/bench_stl.py`) is
built on **transformers 5.6.x** with a backbone-agnostic `Cadrille` mixin
(`common/model.py:_make_cadrille_class`). When that mixin loads weights from
the published `filapro/cadrille` checkpoint (= `cadrille-rl` here, an
RL-fine-tuned multi-modal cadrille trained against transformers 4.50.3),
the *weights load correctly* (verified byte-for-byte on representative
layers) but the **forward pass produces drift** — IoU on DeepCAD-300 falls
from the paper's 0.92 down to **0.14** (6× too low).

The drift was not present in the original cadrille code. Reproducing the
upstream environment exactly — `transformers==4.50.3` + the original
`cadrille.py` / `dataset.py` / `test.py` / `evaluate.py` from
[`col14m/cadrille`](https://github.com/col14m/cadrille) — recovers paper
numbers within sampling noise.

## Verified reproduction on cadrille_rl (this folder)

| dataset                | n   | exec   | mean IoU | paper (image-mode RL) |
|------------------------|-----|--------|----------|------------------------|
| DeepCAD test (random 300, seed=42)   | 300 | 100.0% | **0.915** | 0.922 |
| Fusion360 test (random 300, seed=42) | 300 | 99.7%  | **0.838** | 0.846 |
| `BenchCAD/cad_bench_722` (full 720)  | 720 |  91.2% | 0.075     | (not in paper; OOD diversified track) |

Within ≤1% absolute on the in-distribution sets — call it reproduced.

## How to run

### 1. Install the paper-era stack into the main venv

`pyproject.toml` pins `transformers>=5.6.0`, so `uv run` and `uv sync`
will fight a manual downgrade. Use the venv binary directly.

```bash
uv pip install \
    transformers==4.50.3 \
    tokenizers==0.21.0 \
    accelerate==0.34.2 \
    huggingface-hub==0.27.0
.venv/bin/python -c "import transformers; assert transformers.__version__.startswith('4.50.')"
```

When done with the repro, restore with `uv sync`.

### 2. Generate predictions + score

```bash
bash research/repro_official/run_official.sh
# → eval_outputs/repro_official/{deepcad_test_mesh,fusion360_test_mesh}_n300/score.txt
```

For `cad_bench_722` (which has `composite_png` directly on HF, no STL
files), use the inline scoring snippet in this folder (TODO: pull into
its own script if needed).

## Files

- `cadrille.py`, `dataset.py`, `test.py`, `evaluate.py` — verbatim copies
  from `col14m/cadrille@master`. Only modification: `attn_implementation`
  changed from `flash_attention_2` to `sdpa` (no flash-attn wheel for
  cp311), and `pytorch3d` import made lazy (only needed for PC mode).
- `run_official.sh` — drives the full DeepCAD + Fusion360 reproduction.

## Open: backbone-agnostic mixin in `common/model.py`

The `Cadrille_Qwen2VLForConditionalGeneration` class in `common/model.py`
needs a fix to behave equivalently to upstream's `Cadrille` under
transformers 5.x. Symptom: same weights, same render, same processor
settings, ~6× lower IoU. Suspected: subtle behaviour change in
`get_rope_index()` or vision-token replacement when running multi-modal
inputs through the 5.x model wrapper. Not yet root-caused.

For now: any cadrille_rl number we report from `eval/bench.py` /
`eval/bench_stl.py` is **broken**, and the numbers in this folder are
the ones that match the published paper.
