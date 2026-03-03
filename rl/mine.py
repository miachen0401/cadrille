"""Hard example mining for RL fine-tuning of Cadrille.

Run once before RL training. For each example in the dataset:
  1. Generate K completions from the SFT model
  2. Compute rewards via rl/reward.py
  3. Keep examples where mean(reward) < R_th (model struggles → hard)
  4. Save filtered dataset as a pickle for use by rl/train.py

Expected output scale: ~50k point cloud examples (DeepCAD) / ~3k image examples (Fusion360).

Usage:
    python rl/mine.py \\
        --checkpoint-path ./checkpoints/cadrille \\
        --data-path ./data \\
        --split deepcad_test_mesh \\
        --output ./data/rl_hard_examples.pkl \\
        --R-th 7.5 --K 3
"""

import os
import sys

# Allow standalone execution from repo root or rl/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
from transformers import AutoProcessor
from torch.utils.data import DataLoader

from cadrille import Cadrille, collate
from dataset import CadRecodeDataset
from rl.reward import compute_rewards_parallel


def mine(args):
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    model = Cadrille.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto')
    model.eval()

    dataset = CadRecodeDataset(
        root_dir=args.data_path,
        split=args.split,
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=None,   # no augmentation during mining
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=args.mode,
        n_samples=args.max_samples)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    hard_examples = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc='Mining hard examples')):
            file_name = batch['file_name'][0]
            gt_mesh_path = os.path.join(
                args.data_path, args.split, file_name + '.stl')

            if not os.path.exists(gt_mesh_path):
                continue

            # Expand batch K times so we generate K completions in one call.
            # All K copies share the same prompt — diversity comes from sampling.
            g_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    g_batch[k] = v.repeat(args.K, *([1] * (v.dim() - 1)))
                elif isinstance(v, list):
                    g_batch[k] = v * args.K

            generated_ids = model.generate(
                input_ids=g_batch['input_ids'].to(model.device),
                attention_mask=g_batch['attention_mask'].to(model.device),
                point_clouds=g_batch['point_clouds'].to(model.device),
                is_pc=g_batch['is_pc'].to(model.device),
                is_img=g_batch['is_img'].to(model.device),
                pixel_values_videos=(
                    g_batch['pixel_values_videos'].to(model.device)
                    if g_batch.get('pixel_values_videos') is not None else None),
                video_grid_thw=(
                    g_batch['video_grid_thw'].to(model.device)
                    if g_batch.get('video_grid_thw') is not None else None),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0)

            prompt_len = batch['input_ids'].shape[1]
            completion_ids = generated_ids[:, prompt_len:]
            code_strings = processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            rewards = compute_rewards_parallel(
                code_strings,
                [gt_mesh_path] * args.K,
                workers=args.reward_workers)

            mean_reward = float(np.mean(rewards))

            if mean_reward < args.R_th:
                # Hard example: store enough info to recreate the training sample.
                # We store the normalized point cloud (already computed) so
                # rl/train.py doesn't need to re-run FPS sampling.
                entry = {
                    'gt_mesh_path': gt_mesh_path,
                    'file_name': file_name,
                    'mode': args.mode,
                    'is_pc': bool(batch['is_pc'][0].item()),
                    'is_img': bool(batch['is_img'][0].item()),
                }
                if entry['is_pc']:
                    # point_clouds: [1, 256, 3]  →  store as numpy [256, 3]
                    entry['point_cloud'] = (
                        batch['point_clouds'][0].cpu().numpy())
                else:
                    # For image mode we store only the mesh path;
                    # images are re-rendered by RLDataset at training time.
                    entry['point_cloud'] = None

                hard_examples.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(hard_examples, f)

    print(f'\nHard examples: {len(hard_examples)} / {len(dataset)} '
          f'({100 * len(hard_examples) / max(len(dataset), 1):.1f}%)')
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mine hard examples for RL fine-tuning')
    parser.add_argument('--checkpoint-path', type=str,
                        default='maksimko123/cadrille')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='deepcad_test_mesh',
                        help='Dataset split (e.g. deepcad_test_mesh, fusion360_test_mesh)')
    parser.add_argument('--mode', type=str, default='pc',
                        choices=['pc', 'img'],
                        help='Input modality')
    parser.add_argument('--output', type=str,
                        default='./data/rl_hard_examples.pkl')
    parser.add_argument('--R-th', type=float, default=7.5,
                        help='Reward threshold: keep examples with mean reward < R_th')
    parser.add_argument('--K', type=int, default=3,
                        help='Number of completions to generate per example')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit dataset size (for smoke tests)')
    parser.add_argument('--reward-workers', type=int, default=4,
                        help='Number of parallel subprocess workers for reward computation')
    args = parser.parse_args()
    mine(args)
