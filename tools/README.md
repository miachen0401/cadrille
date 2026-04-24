# Tools

Infrastructure libraries used by `scripts/analysis/` and the rest of the repo.
Not CLI scripts — these are importable helpers. Analysis/one-off scripts live
in `scripts/analysis/` now.

| File | Purpose |
|------|---------|
| `parse_cq.py` | Regex-based feature extractor for CadQuery scripts. Imported by `scripts/analysis/{dataset_stats,failure_analysis,compare_evals}.py`. |
| `check_env/` | Post-install environment verification (torch, open3d, cadquery, dataset, model). Run after `scripts/setup.sh`. |

## Related packages

- **`scripts/analysis/`** — one-off research analysis scripts (error taxonomy,
  KL quadrants, mining health, failure analysis, training-curve plots, etc.).
- **`data_prep/`** — one-time dataset preparation.
- **`bench/`** — training throughput benchmarks.
- **`eval/bench.py`, `eval/bench_visualize.py`** — model evaluation on the
  BenchCAD benchmark.
- **`experiments/repair_lora/`** — repair-LoRA mini-experiment.
- **`experiments/data_prep_cadlib/`** — DeepCAD/Fusion360 mesh generation
  (needs `cadlib`, not pinned in pyproject).
