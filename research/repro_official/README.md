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

### 1. Bootstrap the dedicated paper-repro venv

`pyproject.toml` pins `transformers>=5.6.0` (training stack), but
filapro/cadrille and kulibinai/cadevolve-rl1 were trained on
`transformers==4.50.3` and **drift in 5.x** (DeepCAD IoU 0.92 → 0.14).
To keep the two stacks side-by-side without `uv sync` clobbering either,
this folder uses a separate venv at `.venv-eval/`:

```bash
bash scripts/setup_eval_env.sh           # idempotent — creates .venv-eval if missing
bash scripts/setup_eval_env.sh --rebuild # nuke + reinstall

source scripts/use_eval_env.sh           # activate (4.50.3 stack)
deactivate && source .venv/bin/activate  # back to main (5.x stack)
```

The eval venv ships:
- transformers 4.50.3 / tokenizers 0.21.0 / accelerate 0.34.2 / hf-hub 0.27.0
- torch 2.5.1 + cadquery (git) + trimesh + pyvista + datasets
- open3d source-built wheel (if `scripts/setup.sh` step [4] produced one),
  else PyPI `open3d-cpu==0.18.0` as fallback (the latter SIGSEGVs on a
  small fraction of meshes — wrap in subprocess + fall back to pyvista)

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
