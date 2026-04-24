#!/bin/bash
# run_new_eval.sh — Full evaluation of 5 checkpoints, 500 samples each split.
# Results written to work_dirs/new_eval/<name>/{deepcad,fusion360}/results.csv
# and summarised in new_eval.md after all runs complete.
set -euo pipefail

REPO=/workspace
LOG=/tmp/new_eval.log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "NEW EVAL — $(date)"
echo "============================================================"

cd "$REPO"

run_model() {
    local name=$1
    local ckpt=$2
    echo ""
    echo "------------------------------------------------------------"
    echo "[$name] checkpoint: $ckpt"
    echo "------------------------------------------------------------"
    python3 tools/eval_img.py \
        --checkpoint "$ckpt" \
        --splits deepcad fusion360 \
        --n-samples 500 \
        --out-dir "work_dirs/new_eval/$name" \
        --seed 42 \
        --batch-size 16 \
        --max-new-tokens 768
    echo "[$name] DONE — $(date)"
}

run_model "sft"          "checkpoints/cadrille-sft"
run_model "official-rl"  "checkpoints/cadrille-rl"
run_model "cad-4500"     "checkpoints/cad_ckpt/a100-step4500"
run_model "cad-6000"     "checkpoints/cad_ckpt/a100-step6000"
run_model "cad-7200"     "checkpoints/cad_ckpt/a100-step7200"

echo ""
echo "============================================================"
echo "ALL RUNS DONE — writing new_eval.md"
echo "============================================================"

python3 - <<'PYEOF'
import csv, os, math

REPO = "/workspace"
OUT = os.path.join(REPO, "new_eval.md")

models = [
    ("Official SFT",        "sft"),
    ("Official RL (paper)", "official-rl"),
    ("Ours a100-step4500",  "cad-4500"),
    ("Ours a100-step6000",  "cad-6000"),
    ("Ours a100-step7200",  "cad-7200"),
]
splits = [("DeepCAD",   "deepcad"),
          ("Fusion360", "fusion360")]

def read_csv(path):
    if not os.path.exists(path):
        return None, None, None
    rows = list(csv.DictReader(open(path)))
    n_total = len(rows)
    ious = [float(r['iou']) for r in rows if r.get('iou')]
    cds  = [float(r['cd'])  for r in rows if r.get('cd')]
    if not ious:
        return n_total, None, None
    return n_total, sum(ious)/len(ious), (sorted(cds)[len(cds)//2] * 1000 if cds else None)

lines = [
    "# New Evaluation Results\n",
    f"Seed=42, N=500 per split, img mode, evaluate.py (volumetric IoU + Chamfer Distance).\n",
    "\n## Paper Targets (Table 2)\n",
    "| Model | DeepCAD IoU | Fusion360 IoU |\n",
    "|-------|-------------|---------------|\n",
    "| cadrille SFT (paper) | 86.1% | 77.6% |\n",
    "| cadrille Dr. CPPO (paper) | **92.2%** | **84.6%** |\n",
    "\n## Our Results\n",
    "| Model | DeepCAD IoU | DeepCAD CD | DeepCAD IR | Fusion360 IoU | Fusion360 CD | Fusion360 IR |\n",
    "|-------|-------------|------------|------------|---------------|--------------|---------------|\n",
]

for label, name in models:
    row = [f"| {label}"]
    for _, split in splits:
        csv_path = os.path.join(REPO, "work_dirs/new_eval", name, split, "results.csv")
        n, iou, cd = read_csv(csv_path)
        if iou is None:
            row += ["| — | — | —"]
        else:
            ir = (1 - len([r for r in open(csv_path).readlines()[1:] if r.split(',')[2].strip()]) / max(n-1,1)) * 100
            # recompute IR properly
            rows = list(csv.DictReader(open(csv_path)))
            n_valid_cd = sum(1 for r in rows if r.get('cd'))
            ir_pct = (n - n_valid_cd) / n * 100 if n else 0
            row += [f"| **{iou*100:.1f}%** | {cd:.3f} | {ir_pct:.1f}%"]
    row.append(" |")
    lines.append("".join(row) + "\n")

lines += [
    "\n## Notes\n",
    "- IR = Invalid Rate (fraction of samples where CadQuery execution failed)\n",
    "- CD = median Chamfer Distance × 1000\n",
    "- All evals: img mode, seed=42, 500 random samples from official test sets\n",
    "- cad_ckpt = our RL checkpoints from A100 training run\n",
]

with open(OUT, 'w') as f:
    f.writelines(lines)
print(f"Written: {OUT}")
PYEOF

echo "Done: $(date)"
