#!/usr/bin/env bash
# Run upstream cadrille pipeline (transformers==4.50.3, original cadrille.py
# / dataset.py / test.py / evaluate.py) for proper baseline reproduction on
# DeepCAD + Fusion360. Bypasses the project's transformers 5.x environment
# because the upstream Cadrille subclass relies on transformers 4.x internals
# (self.model.embed_tokens etc.) that don't survive intact in 5.x.
#
# Prereq: `uv pip install transformers==4.50.3 tokenizers==0.21.0 \
#                          accelerate==0.34.2 huggingface-hub==0.27.0`
# and run via .venv/bin/python (NOT `uv run`, which auto-syncs back to 5.x).
#
# Usage:
#   bash research/repro_official/run_official.sh

set -uo pipefail

REPO=/home/hula0401/Projects/cadrille
cd "$REPO"

# Make sure we are NOT auto-resyncing to pyproject.toml's 5.x pin
.venv/bin/python -c "import transformers; assert transformers.__version__.startswith('4.50.'), \
                     f'Expected transformers 4.50.x, got {transformers.__version__}'"

CKPT=checkpoints/cadrille-rl
N_SAMPLES=300
SEED=42

run_one() {
    local SPLIT=$1
    local STL_DIR=$2
    local OUT=eval_outputs/repro_official/${SPLIT}_n${N_SAMPLES}
    mkdir -p "$OUT/py"
    rm -rf "$OUT/py/"*

    echo "=== ${SPLIT} (cadrille_rl, image mode, transformers 4.50.3) ==="
    echo "  STLs: $(ls $STL_DIR/*.stl | wc -l)"
    echo "  Sample size: $N_SAMPLES"

    PYTHONPATH=research/repro_official .venv/bin/python <<PY
import os, sys, random
sys.path.insert(0, 'research/repro_official')
import torch
from transformers import AutoProcessor
from torch.utils.data import DataLoader, ConcatDataset
from functools import partial
from cadrille import Cadrille, collate
from dataset import CadRecodeDataset

# Subsample STLs deterministically — write a tiny pkl-less iterator
import shutil
class SubsetCadRecode(CadRecodeDataset):
    def __init__(self, *a, indices=None, **kw):
        super().__init__(*a, **kw)
        self.annotations = [self.annotations[i] for i in indices]
        self.n_samples = len(self.annotations)

paths = sorted(p for p in os.listdir('${STL_DIR}') if p.endswith('.stl'))
rng = random.Random(${SEED}); rng.shuffle(paths)
keep = set(paths[:${N_SAMPLES}])

ds_full = CadRecodeDataset(
    root_dir='./data', split='${SPLIT}',
    n_points=256, normalize_std_pc=100, noise_scale_pc=None,
    img_size=128, normalize_std_img=200, noise_scale_img=-1,
    num_imgs=4, mode='img')
keep_idx = [i for i, ann in enumerate(ds_full.annotations)
            if os.path.basename(ann['mesh_path']) in keep]
print(f'  picked {len(keep_idx)}/{len(ds_full.annotations)} from {ds_full.split}')
ds_full.annotations = [ds_full.annotations[i] for i in keep_idx]

print('Loading model (cadrille-rl, sdpa) …', flush=True)
model = Cadrille.from_pretrained('${CKPT}',
    torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='auto')
processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')

loader = DataLoader(dataset=ConcatDataset([ds_full]*1),
    batch_size=8, num_workers=4,
    collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

counter = 0
total = len(ds_full)
for batch in loader:
    g = model.generate(
        input_ids=batch['input_ids'].to(model.device),
        attention_mask=batch['attention_mask'].to(model.device),
        point_clouds=batch['point_clouds'].to(model.device),
        is_pc=batch['is_pc'].to(model.device), is_img=batch['is_img'].to(model.device),
        pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos') is not None else None,
        video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw') is not None else None,
        max_new_tokens=768)
    trimmed = [o[len(i):] for i, o in zip(batch['input_ids'], g)]
    pys = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for stem, py in zip(batch['file_name'], pys):
        with open(os.path.join('${OUT}/py', f'{stem}+0.py'), 'w') as f:
            f.write(py)
        counter += 1
        if counter % 50 == 0:
            print(f'  generated {counter}/{total}')
print(f'done: {counter} preds in ${OUT}/py')
PY

    # Score with upstream evaluate.py
    echo "=== Scoring ${SPLIT} ==="
    .venv/bin/python research/repro_official/evaluate.py \
        --gt-mesh-path "$STL_DIR" \
        --pred-py-path "$OUT/py" \
        --n-points 8192 2>&1 | tee "$OUT/score.txt"
}

run_one deepcad_test_mesh data/deepcad_test_mesh
run_one fusion360_test_mesh data/fusion360_test_mesh

echo
echo "=== Final scores ==="
for sp in deepcad_test_mesh fusion360_test_mesh; do
    echo "  $sp:"
    grep "mean iou" eval_outputs/repro_official/${sp}_n${N_SAMPLES}/score.txt | head -1
done
