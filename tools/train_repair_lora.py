"""LoRA SFT for action-conditioned CAD repair (Step 2).

Fine-tunes cadrille-rl with LoRA on synthetic corruption data.

Input modes (--input-mode):
  hstack   : single frame — GT and corrupt render side-by-side (536×268)
             prompt: "Left half: target. Right half: broken prediction."
  2frame   : two frames — GT render (frame 0) and corrupt render (frame 1)
             prompt: "First image: target shape. Second image: current broken shape."
  gt-only  : single frame — GT render only (268×268)
             prompt: "Target 3D shape is shown. Repair action: ..."

Input data: data/repair_sft/train.jsonl / val.jsonl
  Each record: stem, gt_code, corrupt_code, gt_render, corrupt_render, action

Usage
-----
  python3 tools/train_repair_lora.py
  python3 tools/train_repair_lora.py --input-mode 2frame --out checkpoints/repair-lora-2frame
  python3 tools/train_repair_lora.py --input-mode gt-only --out checkpoints/repair-lora-gtonly
  python3 tools/train_repair_lora.py --smoke-test   # 5 steps, no wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

_env_path = _REPO / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from cadrille import Cadrille, collate


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

PROMPTS = {
    'hstack': (
        "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).\n"
        "Repair action: {action} — the box() fallback must be replaced with a proper "
        "sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
    '2frame': (
        "First image: target 3D shape (4 views). Second image: current broken prediction (4 views).\n"
        "Repair action: {action} — the box() fallback must be replaced with a proper "
        "sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
    'gt-only': (
        "The target 3D shape is shown.\n"
        "Repair action: {action} — the box() fallback must be replaced with a proper "
        "sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
}


def _hstack(img1: Image.Image, img2: Image.Image) -> Image.Image:
    h = img1.height
    if img2.height != h:
        img2 = img2.resize((int(img2.width * h / img2.height), h), Image.LANCZOS)
    combined = Image.new('RGB', (img1.width + img2.width, h))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined


class RepairDataset(Dataset):
    def __init__(self, jsonl_path: str, input_mode: str = 'hstack'):
        assert input_mode in ('hstack', '2frame', 'gt-only'), \
            f"input_mode must be hstack/2frame/gt-only, got {input_mode}"
        self.input_mode = input_mode
        with open(jsonl_path) as f:
            self.records = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        gt_img   = Image.open(r['gt_render']).convert('RGB')
        prompt   = PROMPTS[self.input_mode].format(
                       action=r['action'], code=r['corrupt_code'])
        if self.input_mode == 'hstack':
            corr_img = Image.open(r['corrupt_render']).convert('RGB')
            video = [_hstack(gt_img, corr_img)]
        elif self.input_mode == '2frame':
            corr_img = Image.open(r['corrupt_render']).convert('RGB')
            video = [gt_img, corr_img]
        else:  # gt-only
            video = [gt_img]
        return {
            'video':       video,
            'description': prompt,
            'answer':      r['gt_code'],
            'file_name':   r['stem'],
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_val_loss(model, processor, val_loader, device, max_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch_items in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = collate(batch_items, processor=processor, n_points=256, eval=False)
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                point_clouds=batch['point_clouds'].to(device),
                is_pc=batch['is_pc'].to(device),
                is_img=batch['is_img'].to(device),
                pixel_values_videos=(batch['pixel_values_videos'].to(device)
                                     if batch.get('pixel_values_videos') is not None else None),
                video_grid_thw=(batch['video_grid_thw'].to(device)
                                if batch.get('video_grid_thw') is not None else None),
            )
            if out.loss is not None:
                losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float('nan')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = not args.smoke_test and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project='cadrille-rl',
            name=f'repair-lora-r{args.lora_rank}-lr{args.lr:.0e}-ep{args.epochs}-{args.input_mode}',
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = RepairDataset(str(_REPO / args.train_data), input_mode=args.input_mode)
    val_ds   = RepairDataset(str(_REPO / args.val_data),   input_mode=args.input_mode)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    def collate_fn(items): return items  # collate handles batching

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ------------------------------------------------------------------
    # Model + LoRA
    # ------------------------------------------------------------------
    print(f'Loading {args.checkpoint}...')
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28, padding_side='left')
    model = Cadrille.from_pretrained(
        str(_REPO / args.checkpoint), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')

    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        for batch_idx, batch_items in enumerate(train_loader):
            batch = collate(batch_items, processor=processor, n_points=256, eval=False)

            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                point_clouds=batch['point_clouds'].to(device),
                is_pc=batch['is_pc'].to(device),
                is_img=batch['is_img'].to(device),
                pixel_values_videos=(batch['pixel_values_videos'].to(device)
                                     if batch.get('pixel_values_videos') is not None else None),
                video_grid_thw=(batch['video_grid_thw'].to(device)
                                if batch.get('video_grid_thw') is not None else None),
            )

            loss = out.loss / args.grad_accum
            loss.backward()
            epoch_loss += out.loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f'  step {global_step}  loss {out.loss.item():.4f}  lr {lr_now:.2e}')
                    if use_wandb:
                        import wandb
                        wandb.log({'train/loss': out.loss.item(),
                                   'train/lr': lr_now,
                                   'train/step': global_step})

                if args.smoke_test and global_step >= 5:
                    print('Smoke test done.')
                    return

        # -- end of epoch --
        mean_train_loss = epoch_loss / max(n_batches, 1)
        val_loss = compute_val_loss(model, processor, val_loader, device)
        print(f'Epoch {epoch}/{args.epochs}  '
              f'train_loss={mean_train_loss:.4f}  val_loss={val_loss:.4f}')

        if use_wandb:
            import wandb
            wandb.log({'epoch/train_loss': mean_train_loss,
                       'epoch/val_loss': val_loss,
                       'epoch': epoch})

        # Save checkpoint
        ckpt_path = out_dir / f'epoch_{epoch:02d}'
        model.save_pretrained(str(ckpt_path))
        print(f'  Saved → {ckpt_path}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = out_dir / 'best'
            model.save_pretrained(str(best_path))
            print(f'  ✓ New best val_loss={val_loss:.4f} → {best_path}')

    if use_wandb:
        import wandb
        wandb.finish()

    print(f'\nDone. Best val_loss={best_val_loss:.4f}  checkpoints in {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',  default='checkpoints/cadrille-rl')
    parser.add_argument('--train-data',  default='data/repair_sft/train.jsonl')
    parser.add_argument('--val-data',    default='data/repair_sft/val.jsonl')
    parser.add_argument('--out',         default='checkpoints/repair-lora')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch-size',  type=int,   default=1)
    parser.add_argument('--grad-accum',  type=int,   default=8,
                        help='Gradient accumulation steps (effective batch = batch_size × grad_accum)')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--lora-rank',   type=int,   default=16)
    parser.add_argument('--lora-alpha',  type=int,   default=32)
    parser.add_argument('--input-mode',  default='hstack',
                        choices=['hstack', '2frame', 'gt-only'],
                        help='Visual input format: hstack=side-by-side, 2frame=two video frames, gt-only=GT only')
    parser.add_argument('--no-wandb',    action='store_true')
    parser.add_argument('--smoke-test',  action='store_true',
                        help='Run 5 steps only to verify setup')
    args = parser.parse_args()

    # Resource check
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True)
    if result.returncode == 0:
        free_mb = int(result.stdout.strip().split('\n')[0])
        print(f'GPU free VRAM: {free_mb} MB')
        if free_mb < 2000:
            print('WARNING: Less than 2 GB VRAM free — may OOM.')

    train(args)


if __name__ == '__main__':
    main()
