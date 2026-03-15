"""Standalone smoke test: run inference on the N smallest meshes and report IoU.

Replicates rl/train.py:_reward_smoke_test() as a CLI tool so it can be run
on any checkpoint without launching full training.

Usage
-----
python tools/smoke_eval.py \\
    --checkpoints checkpoints/cadrille-sft checkpoints/cadrille-rl \\
    --pkl data/smoke_train/smoke_train.pkl \\
    --n 5 \\
    --modality pc
"""

import argparse
import json
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

CKPT_SHORT = {
    'cadrille-sft':   'sft',
    'cadrille-rl':    'official-rl',
    'a100-step4500':  'a100-4500',
    'a100-step6000':  'a100-6000',
    'a100-step7200':  'a100-7200',
    'checkpoint-9000':'4080-9000',
}


def short_name(path):
    for k, v in CKPT_SHORT.items():
        if k in path:
            return v
    return os.path.basename(path.rstrip('/'))


def run_smoke(ckpt_path, examples, max_new_tokens, modality, device):
    from cadrille import Cadrille, collate
    from rl.reward import _get_worker_path
    from rl.dataset import render_img
    from transformers import AutoProcessor

    label = short_name(ckpt_path)
    print(f'\n{"="*60}')
    print(f'[smoke] {label}  ({ckpt_path})')
    print(f'{"="*60}')

    processor = AutoProcessor.from_pretrained(ckpt_path, use_fast=False)
    model = Cadrille.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None

    ious = []
    rows = []
    for i, ex in enumerate(examples):
        mesh_path = ex['gt_mesh_path']
        fname = ex.get('file_name', os.path.basename(mesh_path))

        # Build input
        if modality == 'img':
            item = render_img(mesh_path)
        else:
            import trimesh
            from dataset import mesh_to_point_cloud
            mesh = trimesh.load(mesh_path)
            # Normalize to [0,1] unit cube (training meshes are pre-normalized;
            # smoke_train meshes may be in raw CAD coords, so normalize here).
            mesh.apply_translation(-mesh.bounds[0])
            mesh.apply_scale(1.0 / mesh.extents.max())
            pc = mesh_to_point_cloud(mesh, 256)
            pc = (pc - 0.5) * 2
            item = {'point_cloud': pc}

        item['gt_mesh_path'] = mesh_path
        item['description'] = 'Generate cadquery code'
        item['file_name'] = fname
        collate_item = {k: v for k, v in item.items() if k != 'gt_mesh_path'}

        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None

        batch = collate([collate_item], processor=processor, n_points=256, eval=True)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                point_clouds=batch['point_clouds'].to(device),
                is_pc=batch['is_pc'].to(device),
                is_img=batch['is_img'].to(device),
                pixel_values_videos=(
                    batch['pixel_values_videos'].to(device)
                    if batch.get('pixel_values_videos') is not None else None),
                video_grid_thw=(
                    batch['video_grid_thw'].to(device)
                    if batch.get('video_grid_thw') is not None else None),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )

        prompt_len = batch['input_ids'].shape[1]
        code = processor.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        # Score
        payload = json.dumps({'code_str': code, 'gt_mesh_path': mesh_path, 'compute_chamfer': False})
        proc = subprocess.run(
            [sys.executable, _get_worker_path()],
            input=payload, capture_output=True, text=True, timeout=60)

        iou = None
        if proc.stdout.strip():
            data = json.loads(proc.stdout.strip())
            iou = data.get('iou')

        bar = ('█' * int((iou or 0) * 20)) if iou is not None else '✗' * 5
        status = f'IoU={iou:.4f} [{bar:<20}]' if iou is not None else 'FAILED'
        print(f'  [{i+1}/{len(examples)}] {fname:<30} {status}')
        if iou is not None:
            ious.append(iou)
        rows.append({'file_name': fname, 'iou': iou})

    avg = float(np.mean(ious)) if ious else 0.0
    print(f'\n  avg IoU = {avg:.4f}  ({len(ious)}/{len(examples)} valid)')

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return {'label': label, 'avg_iou': avg, 'n_valid': len(ious), 'n_total': len(examples), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--pkl',     default='data/smoke_train/smoke_train.pkl')
    parser.add_argument('--n',       type=int, default=5, help='Number of examples (smallest meshes)')
    parser.add_argument('--modality', default='pc', choices=['pc', 'img'])
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--out', default='work_dirs/smoke_eval/results.json')
    args = parser.parse_args()

    import pickle
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    # Filter valid paths, sort by file size (smallest = simplest geometry)
    valid = [ex for ex in data if os.path.exists(ex['gt_mesh_path'])]
    examples = sorted(valid, key=lambda ex: os.path.getsize(ex['gt_mesh_path']))[:args.n]
    print(f'Smoke examples ({args.n} smallest):')
    for ex in examples:
        sz = os.path.getsize(ex['gt_mesh_path'])
        print(f'  {ex.get("file_name","?"):30s}  {sz:>8} bytes')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []
    for ckpt in args.checkpoints:
        result = run_smoke(ckpt, examples, args.max_new_tokens, args.modality, device)
        all_results.append(result)

    # Summary table
    print(f'\n{"="*60}')
    print(f'SMOKE TEST SUMMARY  (modality={args.modality}, n={args.n})')
    print(f'{"="*60}')
    print(f'  {"Checkpoint":<20}  {"Avg IoU":>8}  {"Valid":>6}')
    print(f'  {"-"*40}')
    for r in all_results:
        print(f'  {r["label"]:<20}  {r["avg_iou"]:>8.4f}  {r["n_valid"]}/{r["n_total"]}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
