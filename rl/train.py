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


def _preflight_check(args):
    """Fail fast before loading the model if checkpoint or CadQuery is broken."""
    import glob as _glob
    import subprocess
    import textwrap

    # ── 1. Checkpoint has actual weight files ─────────────────────────────────
    ckpt = args.checkpoint_path
    weights = (_glob.glob(os.path.join(ckpt, 'model*.safetensors')) +
               _glob.glob(os.path.join(ckpt, 'pytorch_model*.bin')))
    if not weights:
        raise RuntimeError(
            f"\n[preflight] No weight files found in: {ckpt}\n"
            f"  Expected model*.safetensors or pytorch_model*.bin.\n"
            f"  Re-download with: huggingface-cli download maksimko123/cadrille "
            f"--repo-type model --local-dir {ckpt}")
    print(f'[preflight] Checkpoint OK  ({len(weights)} weight file(s))')

    # ── 2. CadQuery subprocess produces a valid mesh ──────────────────────────
    # The reward worker does exec(code) then g['r'].val().tessellate(...)
    # If CadQuery is broken or OCP libs are missing, every reward will be -10.
    probe = textwrap.dedent("""\
        import cadquery as cq, json
        r = cq.Workplane('XY').box(1, 1, 1)
        v, f = r.val().tessellate(0.01)
        assert len(f) > 0
        print(json.dumps({'faces': len(f)}))
    """)
    proc = subprocess.run(
        [sys.executable, '-c', probe],
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"\n[preflight] CadQuery subprocess test failed — reward will be -10 for all outputs.\n"
            f"  returncode : {proc.returncode}\n"
            f"  stderr     :\n{proc.stderr}")
    print(f'[preflight] CadQuery subprocess OK  {proc.stdout.strip()}')


@torch.no_grad()
def _reward_smoke_test(model, dataset, processor, args, n=3):
    """Generate code for n examples and run the full reward pipeline.

    Prints the generated code and full subprocess stderr so you can see
    exactly why reward = -10 without waiting for training to get going.
    """
    import subprocess as _sp
    from cadrille import collate
    from rl.reward import _get_worker_path
    import json as _json

    print(f'\n{"="*60}')
    print(f'[smoke test] Generating + scoring {n} examples ...')
    print(f'{"="*60}')
    device = next(model.parameters()).device
    model.eval()

    for i in range(min(n, len(dataset))):
        example = dataset[i]
        collate_item = {k: v for k, v in example.items() if k != 'gt_mesh_path'}
        batch = collate([collate_item], processor=processor, n_points=256, eval=True)

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
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None)

        prompt_len = batch['input_ids'].shape[1]
        code = processor.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        print(f'\n─── example {i+1}: {example.get("file_name", "?")} ───')
        print(f'  generated ({len(code)} chars):')
        # Print first 400 chars of generated code, indented
        for line in code[:400].splitlines():
            print(f'    {line}')
        if len(code) > 400:
            print('    ...')

        # Run the reward subprocess with full stderr exposed
        payload = _json.dumps({
            'code_str': code,
            'gt_mesh_path': example['gt_mesh_path'],
            'compute_chamfer': False,
        })
        proc = _sp.run(
            [sys.executable, _get_worker_path()],
            input=payload, capture_output=True, text=True, timeout=30)

        if proc.stdout.strip():
            data = _json.loads(proc.stdout.strip())
            if data.get('iou') is not None:
                print(f'  → IoU = {data["iou"]:.4f}  reward = {data["iou"]*10:.2f}  ✓')
            else:
                print(f'  → FAILED  error: {data.get("error", "?")}')
        else:
            print(f'  → FAILED  returncode={proc.returncode}')

        if proc.stderr.strip():
            print(f'  subprocess stderr (last 400 chars):')
            for line in proc.stderr.strip()[-400:].splitlines():
                print(f'    {line}')

    print(f'\n{"="*60}\n')
    model.train()


def train(args, cfg_to_save=None):
    _preflight_check(args)
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
                # Resume the same W&B run ONLY when resuming a checkpoint.
                # Fresh training (start_step=0) always creates a new run so that
                # wandb.log(step=N) is never rejected as "step < last logged step"
                # from a previous crashed run that already had data at step N.
                wandb_id_file = os.path.join(args.output_dir, 'wandb_run_id.txt')
                stored_run_id = None
                if args.start_step > 0 and os.path.exists(wandb_id_file):
                    with open(wandb_id_file) as f:
                        stored_run_id = f.read().strip() or None

                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or args.run_name,
                    entity=args.wandb_entity or None,
                    id=stored_run_id,
                    config=cfg_to_save or vars(args),
                    mode='offline' if args.wandb_offline else 'online',
                    resume='allow',
                    settings=wandb.Settings(console='off'),
                )
                if wandb.run:
                    # Persist run ID so session restarts can resume the same W&B run
                    with open(wandb_id_file, 'w') as f:
                        f.write(wandb.run.id)
                    try:
                        run_url = wandb.run.url or wandb.run.get_url()
                    except Exception:
                        run_url = f'https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}'
                    # Print URL prominently so it's visible in Colab output
                    print(f'\n{"="*60}')
                    print(f'W&B run : {run_url}')
                    print(f'Run ID  : {wandb.run.id}')
                    print(f'{"="*60}\n')
                    with open(os.path.join(args.output_dir, 'wandb_run.txt'), 'w') as f:
                        f.write(f"run_id: {wandb.run.id}\n")
                        f.write(f"run_url: {run_url}\n")
                        f.write(f"project: {wandb.run.project}\n")
                use_wandb = True
            except Exception as e:
                print(f'Warning: wandb.init() failed ({e}). Pass --wandb-offline for local logging.')

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    model = Cadrille.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('Optimizer: Adam (fp32)')

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

        _reward_smoke_test(model, dataset, processor, args)

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
