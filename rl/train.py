"""RL fine-tuning of Cadrille (Dr. CPPO / GRPO or DPO).

All training settings live in a YAML config file.
Only a handful of flags can be overridden from the CLI.

Usage
-----
# Single GPU (RTX 4080, 16 GB)
python rl/train.py --config configs/rl/4080.yaml

# H100 (80 GB)
python rl/train.py --config configs/rl/h100.yaml

# 8× H100 (via torchrun)
torchrun --nproc_per_node=8 rl/train.py --config configs/rl/h100x8.yaml

# Quick smoke test
python rl/train.py --config configs/rl/4080.yaml --max-steps 3 --wandb-offline
"""

import os
import sys

# Allow execution from repo root or rl/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import argparse
import yaml
import torch
from transformers import AutoProcessor

from cadrille import Cadrille
from rl.config import load_yaml, resolve_args
from rl.dataset import MeshDataset, RLDataset, DPODataset
from rl.eval import load_val_examples
from rl.algorithms.cppo import train_cppo
from rl.algorithms.dpo import train_dpo

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def train(args, cfg_to_save=None):
    os.makedirs(args.output_dir, exist_ok=True)

    if cfg_to_save:
        cfg_path = os.path.join(args.output_dir, 'run_config.yaml')
        if not os.path.exists(cfg_path):
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_to_save, f, default_flow_style=False, sort_keys=True)
            print(f'Config snapshot saved → {cfg_path}')

    # W&B
    use_wandb = False
    if args.wandb_project:
        if not _WANDB_AVAILABLE:
            print('Warning: wandb not installed.')
        else:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or args.run_name,
                    entity=args.wandb_entity or None,
                    config=cfg_to_save or vars(args),
                    mode='offline' if args.wandb_offline else 'online',
                    resume='allow',
                    settings=wandb.Settings(console='off'),
                )
                try:
                    if wandb.run:
                        with open(os.path.join(args.output_dir, 'wandb_run.txt'), 'w') as f:
                            f.write(f"run_id: {wandb.run.id}\n")
                            f.write(f"run_url: {wandb.run.url}\n")
                            f.write(f"project: {wandb.run.project}\n")
                except Exception:
                    pass
                use_wandb = True
            except Exception as e:
                print(f'Warning: wandb.init() failed ({e}). Pass --wandb-offline for local logging.')

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

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
        print('Optimizer: Adam8bit (bitsandbytes)')
    except ImportError:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print('Optimizer: Adam (fp32) — bitsandbytes not found; may OOM on 16 GB GPU')

    # Validation
    val_modalities = tuple(m.strip() for m in args.val_modalities.split(','))
    val_examples = []
    for split_dir, n_samples in [
        (args.val_deepcad_dir,   args.val_samples_deepcad),
        (args.val_fusion360_dir, args.val_samples_fusion360),
    ]:
        if split_dir and os.path.isdir(split_dir) and n_samples > 0:
            val_examples += load_val_examples(split_dir, n_samples, modalities=val_modalities)
    if not val_examples:
        print('No validation dirs found; skipping validation.')

    if args.mode == 'cppo':
        if args.data_dir:
            dataset = MeshDataset(args.data_dir, noise_scale=0.01)
        elif args.hard_examples_pkl:
            dataset = RLDataset(args.hard_examples_pkl)
        else:
            raise ValueError('Provide data_dir (real meshes) or hard_examples_pkl in config')

        old_model = copy.deepcopy(model).cpu()
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad_(False)
        train_cppo(model, old_model, optimizer, dataset, processor,
                   val_examples, use_wandb, args)

    elif args.mode == 'dpo':
        if not args.dpo_dataset:
            raise ValueError('dpo_dataset is required for DPO mode')
        dataset = DPODataset(args.dpo_dataset)
        train_dpo(model, optimizer, dataset, processor,
                  val_examples, use_wandb, args)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RL fine-tuning of Cadrille. All settings in YAML config.')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config (e.g. configs/rl/4080.yaml)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Override run name from config')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Override checkpoint_path from config')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override max_steps from config (useful for quick tests)')
    parser.add_argument('--mode', type=str, default=None, choices=['cppo', 'dpo'],
                        help='Override mode from config')
    parser.add_argument('--wandb-offline', action='store_true',
                        help='Force W&B offline mode')
    parser.add_argument('--sequential-generation', action='store_true', default=None,
                        help='Force sequential rollout generation (overrides config)')

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    resolved_cfg = resolve_args(args, cfg)

    print(f'Run name : {args.run_name}')
    print(f'Output   : {args.output_dir}')

    train(args, cfg_to_save=resolved_cfg)
