"""Quick diagnostic for step-7200 checkpoint garbled output.

Compares greedy vs sampled generation on the same 3 smoke-test examples.
"""
import os, sys, pickle, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor
from cadrille import Cadrille, collate

CKPT = './checkpoints/rl-s50k-lr1e-5-G16-cppo-0311-1946/checkpoint-7200'
PKL  = './data/mined/combined_hard.pkl'

print('Loading model...')
processor = AutoProcessor.from_pretrained(CKPT, min_pixels=256*28*28,
                                          max_pixels=1280*28*28, padding_side='left')
model = Cadrille.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                  attn_implementation='flash_attention_2',
                                  device_map='auto')
model.eval()
device = next(model.parameters()).device
print(f'Model on {device}')

from rl.dataset import RLDataset
dataset = RLDataset(PKL, modality='img')

valid_indices = [i for i in range(len(dataset.examples))
                 if os.path.exists(dataset.examples[i]['gt_mesh_path'])]
smoke = sorted(valid_indices,
               key=lambda i: os.path.getsize(dataset.examples[i]['gt_mesh_path']))[:3]
smoke = [(i, dataset[i]) for i in smoke]

def gen(example, do_sample, temperature, top_k, top_p, max_new_tokens=400):
    item = {k: v for k, v in example.items()
            if k not in ('_dataset_idx',) and not k.startswith('_')}
    batch = collate([item], processor=processor, n_points=256, eval=True)
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None
    kwargs = dict(
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        point_clouds=batch['point_clouds'].to(device),
        is_pc=batch['is_pc'].to(device),
        is_img=batch['is_img'].to(device),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if batch.get('pixel_values_videos') is not None:
        kwargs['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
        kwargs['video_grid_thw'] = batch['video_grid_thw'].to(device)
    if temperature is not None: kwargs['temperature'] = temperature
    if top_k      is not None: kwargs['top_k']      = top_k
    if top_p      is not None: kwargs['top_p']      = top_p
    with torch.no_grad():
        ids = model.generate(**kwargs)
    prompt_len = batch['input_ids'].shape[1]
    return processor.decode(ids[0, prompt_len:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)

from rl.reward import _get_worker_path

def score(code, gt_path):
    payload = json.dumps({'code_str': code, 'gt_mesh_path': gt_path, 'compute_chamfer': False})
    proc = subprocess.run([sys.executable, _get_worker_path()],
                          input=payload, capture_output=True, text=True, timeout=30)
    if proc.stdout.strip():
        d = json.loads(proc.stdout.strip())
        return d.get('iou'), d.get('error')
    return None, f'returncode={proc.returncode}'

configs = [
    ('greedy',          False, None,  None,  None),
    ('temp=0.3 top_k50',True,  0.3,   50,    1.0),
    ('temp=1.0 top_k50',True,  1.0,   50,    1.0),
    ('temp=0.3 top_k=0',True,  0.3,   0,     1.0),
]

for idx, ex in smoke:
    fname = ex.get('file_name', os.path.basename(ex.get('gt_mesh_path', '')))
    print(f"\n{'='*70}")
    print(f"Example: {fname}  (idx={idx}, size={os.path.getsize(ex['gt_mesh_path'])})")
    print(f"{'='*70}")
    for label, do_sample, temp, top_k, top_p in configs:
        code = gen(ex, do_sample, temp, top_k, top_p)
        iou, err = score(code, ex['gt_mesh_path'])
        snippet = code[:120].replace('\n', ' ↵ ')
        status = f'IoU={iou:.4f}' if iou is not None else f'FAIL({err})'
        print(f"  [{label:22s}]  {status:18s}  {snippet}")
