# Scripts

```
scripts/
  setup.sh                         install deps (optional: --data / --full)
  run_sft.sh run_rl.sh             training entry points
  run_eval.sh run_passk.sh         evaluation entry points
  mine_and_train.sh                mine hard examples then RL train
  pack_datasets.sh                 zip test meshes for HF push

  check_env/                       post-install environment verification
    check_{torch,open3d,cadquery,dataset,model}.py
    fetch_cad_recode.py            throttled HF downloader

  analysis/                        one-off research analyses
    parse_cq.py                      CadQuery regex feature extractor (local lib)
    analyze_errors.py                full-dataset inference + error taxonomy
    analyze_dim_errors.py            dim-error sub-classification
    analyze_sft_rl_delta.py          per-case IoU delta SFT vs RL
    mining_analysis.py               hard-example mining health
    plot_kl_quadrants.py             KL quadrant composition over training
    diag_resume_entropy.py           reproduce RL entropy/KL to debug resume
    render_comparison_grid.py        GT vs SFT-pred vs RL-pred grid
    render_singleview_grid.py        single-view contact sheet
    bench_compare_vis.py             visual diff of two bench eval runs
    compare_evals.py                 IoU/CD comparison across runs
    dataset_stats.py                 op freq / code-length / plane distributions
    failure_analysis.py              per-op failure rate, runtime error breakdown
    fillet_analysis.py               fillet-specific analysis
    training_dynamics.py             W&B training-curve plots
    run_new_eval.sh                  wrapper for a new eval on fresh ckpts
```

Everything runs from the repo root. Each analysis script has a full argparse
docstring — run with `--help`.
