# data/ — Dataset Directory

All datasets are downloaded automatically by `bash scripts/setup.sh --data`.
Large files are gitignored. Do not commit STL/PNG/pkl files.

---

## Directory Map

```
data/
├── deepcad_train_mesh/      # 84,526 STL + 84,526 PNG (pre-rendered)  [2.7 GB]  HF: Hula0401/deepcad_train_mesh
├── fusion360_train_mesh/    # 15,410 STL + 15,410 PNG (pre-rendered)  [582 MB]  HF: Hula0401/fusion360_train_mesh
├── deepcad_test_mesh/       #  8,046 STL +  8,046 PNG (pre-rendered)  [572 MB]  HF: Hula0401/deepCAD_test
├── fusion360_test_mesh/     #  1,725 STL +  1,725 PNG (pre-rendered)  [158 MB]  HF: Hula0401/fusion360_test_mesh
├── mined/                   # Hard-mined examples for RL training
│   ├── combined_hard.pkl            # merged hard examples (gt_mesh_path, file_name)
│   ├── deepcad_hard.pkl             # DeepCAD hard examples (pre-merge)
│   ├── fusion360_hard.pkl           # Fusion360 hard examples (pre-merge)
│   ├── deepcad_hard_scores.jsonl    # per-sample IoU scores from full DeepCAD scan
│   ├── fusion360_hard_scores.jsonl  # per-sample IoU scores from full Fusion360 scan
│   ├── iou_distribution.png         # IoU histogram (DeepCAD + Fusion360)
│   ├── hard_count_vs_threshold.png  # hard example count vs R_th threshold curve
│   └── upload/                      # STL zips staged for Hula0401/mine_CAD HF upload
│       ├── combined_hard_stls.zip
│       ├── deepcad_hard_stls.zip
│       └── fusion360_hard_stls.zip
├── cadrille_training/       # Symlinks only — no actual files
│   ├── deepcad  -> ../deepcad_train_mesh
│   └── fusion360 -> ../fusion360_train_mesh
│   NOTE: combined_hard.pkl paths go through these symlinks; do not remove without
│   also updating pkl gt_mesh_path entries.
├── cad-recode-v1.5/         # SFT training data (partial: ~3k pairs, full ~100k) [149 MB]
│   └── train/batch_00/ + batch_val/
├── smoke_train/             # 100 STL + smoke_train.pkl for smoke-test training   [416 KB]
├── deepcad_test_mini/       # 100-sample subset of deepcad_test_mesh (quick eval) [~4 MB]
├── deepcad_test_mini30/     #  30-sample subset of deepcad_test_mesh (quick eval) [~1 MB]
└── data.tar                 # Original DeepCAD source download (Columbia Univ, 199 MB)
                             # Contains cad_json.tar.gz + cad_vec.tar.gz + split JSON.
                             # Already converted to deepcad_train_mesh/ — safe to delete.
```

---

## HuggingFace Sources

| Directory | HuggingFace repo | Downloaded by |
|-----------|-----------------|---------------|
| `deepcad_train_mesh/` | `Hula0401/deepcad_train_mesh` | `setup.sh --full` |
| `fusion360_train_mesh/` | `Hula0401/fusion360_train_mesh` | `setup.sh --full` |
| `deepcad_test_mesh/` | `Hula0401/deepCAD_test` | `setup.sh --data` |
| `fusion360_test_mesh/` | `Hula0401/fusion360_test_mesh` | `setup.sh --data` |
| `mined/` | `Hula0401/mine_CAD` | `setup.sh --data` |
| `cad-recode-v1.5/` | `filapro/cad-recode-v1.5` | manual (SFT only) |

`--data`: minimum needed for RL training (~2 GB, ~5 min)
`--full`: also downloads train meshes for mining new hard examples (~4 GB extra)

---

## Notes

- All STL meshes are pre-normalised to [0, 1]³. No preprocessing needed after download.
- Pre-rendered PNGs (`{stem}_render.png`) live alongside their STLs. `render_img()` loads
  the PNG if present, otherwise falls back to on-the-fly rendering.
- `deepcad_train_mesh/` and `fusion360_train_mesh/` are the source for mining hard examples
  via `rl/mine.py`. The resulting `mined/combined_hard.pkl` is what RL training actually uses.
- `cad-recode-v1.5/` is only needed to retrain SFT from scratch. The public SFT checkpoint
  (`checkpoints/cadrille-sft`) makes this unnecessary for RL-only work.
- The original DeepCAD JSON source (`cad_json/`, 3.9 GB) and Fusion360 raw dataset
  (`_zips/`, 7.1 GB) have been deleted — already converted to STLs and uploaded to HF.
- `data.tar` and render zips (`deepcad_hard_renders.zip`, `fusion360_hard_renders.zip`) have
  been deleted. Renders live alongside their STLs as `{stem}_render.png` and are on HuggingFace.
- IoU score distributions from mining: `data/mined/iou_distribution.png`
  and `data/mined/hard_count_vs_threshold.png`
