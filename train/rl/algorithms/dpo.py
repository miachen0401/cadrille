"""DPO algorithm for RL fine-tuning."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cadrille import Cadrille, collate
from train.rl.eval import run_validation, log_eval

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _collate_with_completion(item: dict, completion: str, processor) -> dict:
    training_item = {k: v for k, v in item.items()
                     if k not in ('gt_mesh_path', 'y_w', 'y_l',
                                  'ref_logp_w', 'ref_logp_l')}
    training_item['answer'] = completion
    return collate([training_item], processor=processor, n_points=256, eval=False)


def dpo_step(model, optimizer, item: dict, processor, args) -> dict:
    """One DPO update step.

    L = -log σ(β · [(log π_θ(y_w) - ref_w) - (log π_θ(y_l) - ref_l)])
    """
    device = next(model.parameters()).device
    ref_logp_w = torch.tensor(item['ref_logp_w'], device=device)
    ref_logp_l = torch.tensor(item['ref_logp_l'], device=device)

    base_item = {k: v for k, v in item.items()
                 if k not in ('y_w', 'y_l', 'ref_logp_w', 'ref_logp_l', 'gt_mesh_path')}
    batch_w = _collate_with_completion(base_item, item['y_w'], processor)
    batch_l = _collate_with_completion(base_item, item['y_l'], processor)

    model.train()

    def _seq_logprob(batch):
        out = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=None,
            point_clouds=batch['point_clouds'].to(device),
            is_pc=batch['is_pc'].to(device),
            is_img=batch['is_img'].to(device),
            pixel_values_videos=(
                batch['pixel_values_videos'].to(device)
                if batch.get('pixel_values_videos') is not None else None),
            video_grid_thw=(
                batch['video_grid_thw'].to(device)
                if batch.get('video_grid_thw') is not None else None),
        )
        return Cadrille.compute_sequence_logprob(
            out.logits, batch['labels'].to(device), mean_reduction=True).squeeze(0)

    log_p_w = _seq_logprob(batch_w)
    log_p_l = _seq_logprob(batch_l)
    margin = args.beta * ((log_p_w - ref_logp_w) - (log_p_l - ref_logp_l))
    loss   = -F.logsigmoid(margin)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'train/loss':          loss.item(),
        'train/reward_margin': margin.item(),
        'train/chosen_logp':   log_p_w.item(),
        'train/rejected_logp': log_p_l.item(),
    }


def train_dpo(model, optimizer, dataset, processor,
              val_examples, use_wandb, args):
    log_path = os.path.join(args.output_dir, 'log.txt')
    step  = 0
    epoch = 0

    if val_examples:
        print('\n[eval step=0 (pre-training baseline)]')
        val_metrics = run_validation(model, val_examples, processor, args)
        log_eval(val_metrics, step=0, log_path=log_path, use_wandb=use_wandb)
        model.train()

    pbar = tqdm(total=args.max_steps, desc='DPO')
    while step < args.max_steps:
        epoch += 1
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        for idx in indices:
            if step >= args.max_steps:
                break
            try:
                metrics = dpo_step(model, optimizer, dataset[idx], processor, args)
            except Exception as e:
                print(f'[step {step}] dpo_step error: {e}')
                continue

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{metrics['train/loss']:.3f}",
                margin=f"{metrics['train/reward_margin']:.2f}")

            if step % args.log_steps == 0:
                line = ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())
                with open(log_path, 'a') as f:
                    f.write(f'step={step} epoch={epoch} {line}\n')
                if use_wandb:
                    wandb.log({
                        'loss': metrics['train/loss'],
                        **{k: v for k, v in metrics.items() if k != 'train/loss'},
                    }, step=step)

            if val_examples and step % args.eval_steps == 0:
                print(f'\n[eval step={step}]')
                val_metrics = run_validation(model, val_examples, processor, args)
                log_eval(val_metrics, step=step, log_path=log_path, use_wandb=use_wandb)
                model.train()

            if step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

        if epoch % args.dpo_epochs_per_round == 0:
            print(f'[epoch {epoch}] DPO reference refresh point.')

    pbar.close()
    final_dir = os.path.join(args.output_dir, 'checkpoint-final')
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f'Training complete. Final checkpoint → {final_dir}')
