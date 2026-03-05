"""Validation and logging utilities for RL training."""

import os
import random
import numpy as np
import torch

from cadrille import Cadrille, collate
from rl.dataset import render_img
from rl.reward import compute_metrics, init_eval_pool

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_eval(val_metrics: dict, step: int, log_path: str, use_wandb: bool):
    """Write an eval result to log.txt and optionally W&B."""
    log_line = (
        f"step={step}"
        + ''.join(f" {k}={v:.4f}" for k, v in val_metrics.items())
    )
    with open(log_path, 'a') as f:
        f.write(log_line + '\n')
    if use_wandb:
        wandb.log(val_metrics, step=step)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def load_val_examples(split_dir: str, n_samples: int, n_points: int = 256,
                      modalities: tuple = ('pc',)) -> list:
    """Sample n_samples GT meshes from split_dir with a fixed seed.

    Returns one example dict per (mesh × modality) pair.
    Each dict has keys: point_cloud (pc mode) OR video (img mode),
    plus gt_mesh_path, file_name, _dataset_label, _modality.
    """
    import trimesh
    from dataset import mesh_to_point_cloud

    stl_files = sorted(f for f in os.listdir(split_dir) if f.endswith('.stl'))
    rng = random.Random(42)
    rng.shuffle(stl_files)

    base = []
    for fname in stl_files[:n_samples * 3]:
        if len(base) >= n_samples:
            break
        gt_mesh_path = os.path.join(split_dir, fname)
        try:
            mesh = trimesh.load(gt_mesh_path)
            pc = mesh_to_point_cloud(mesh, n_points)
            pc = (pc - 0.5) * 2
            base.append({
                '_gt_mesh_path': gt_mesh_path,
                '_file_name': fname[:-4],
                '_pc': pc,
            })
        except Exception:
            pass

    dataset_label = os.path.basename(os.path.normpath(split_dir))
    if 'deepcad' in dataset_label.lower():
        dataset_label = 'DeepCAD test'
    elif 'fusion' in dataset_label.lower():
        dataset_label = 'Fusion360 test'

    examples = []
    for b in base:
        for mod in modalities:
            if mod == 'pc':
                examples.append({
                    'point_cloud': b['_pc'],
                    'description': 'Generate cadquery code',
                    'file_name': b['_file_name'],
                    'gt_mesh_path': b['_gt_mesh_path'],
                    '_dataset_label': dataset_label,
                    '_modality': 'pc',
                })
            elif mod == 'img':
                try:
                    img_item = render_img(b['_gt_mesh_path'])
                    img_item.update({
                        'description': 'Generate cadquery code',
                        'file_name': b['_file_name'],
                        'gt_mesh_path': b['_gt_mesh_path'],
                        '_dataset_label': dataset_label,
                        '_modality': 'img',
                    })
                    examples.append(img_item)
                except Exception:
                    pass

    n_pc  = sum(1 for e in examples if e['_modality'] == 'pc')
    n_img = sum(1 for e in examples if e['_modality'] == 'img')
    print(f'Loaded {len(base)} meshes from {split_dir} → {n_pc} pc + {n_img} img examples')
    return examples


@torch.no_grad()
def eval_one_pass(model, examples: list, processor, max_new_tokens: int,
                  eval_batch_size: int = 8, reward_workers: int = 2,
                  eval_timeout: float = 120.0) -> dict:
    """Greedy eval on a list of examples; return per-modality/dataset metrics dict.

    Batches inference (eval_batch_size items per generate call) and runs
    compute_metrics via the warm eval pool (Fix 2) with the longer eval_timeout
    (Fix 1) and result cache (Fix 4).  reward_workers controls the
    ThreadPoolExecutor used to submit pool jobs concurrently (Fix 5).
    Returns W&B-ready dict.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    device = next(model.parameters()).device
    model.eval()

    # ── 1. Batched inference ──────────────────────────────────────────────────
    all_codes = []
    n = len(examples)
    for i in range(0, n, eval_batch_size):
        chunk = examples[i:i + eval_batch_size]
        collate_items = [{k: v for k, v in ex.items() if not k.startswith('_')}
                         for ex in chunk]
        batch = collate(collate_items, processor=processor, n_points=256, eval=True)

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
            temperature=None,
            top_p=None,
            top_k=None)

        prompt_len = batch['input_ids'].shape[1]
        for j in range(len(chunk)):
            code = processor.decode(generated_ids[j, prompt_len:], skip_special_tokens=True)
            all_codes.append(code)

    # ── 2. Parallel reward computation ───────────────────────────────────────
    buckets = defaultdict(lambda: {'ious': [], 'cds': [], 'failures': 0, 'total': 0})

    def _score(idx):
        ex = examples[idx]
        iou_reward, cd = compute_metrics(
            all_codes[idx], ex['gt_mesh_path'],
            timeout=eval_timeout,
            use_pool=True,   # warm pool + result cache (Fix 1/2/4)
        )
        return idx, iou_reward, cd

    with ThreadPoolExecutor(max_workers=reward_workers) as pool:
        futures = [pool.submit(_score, i) for i in range(n)]
        for fut in as_completed(futures):
            idx, iou_reward, cd = fut.result()
            ex    = examples[idx]
            mod   = ex.get('_modality', 'pc')
            label = ex.get('_dataset_label', 'DeepCAD test')
            key   = (mod, label)
            buckets[key]['total'] += 1
            if iou_reward <= -10.0:
                buckets[key]['failures'] += 1
            else:
                buckets[key]['ious'].append(iou_reward / 10.0)
                if cd is not None:
                    buckets[key]['cds'].append(cd)

    out = {}
    for (mod, label), b in buckets.items():
        ious = b['ious']
        cds  = b['cds']
        fail_frac  = b['failures'] / b['total'] if b['total'] > 0 else 0.0
        mean_iou   = float(np.mean(ious))   if ious else 0.0
        median_iou = float(np.median(ious)) if ious else 0.0
        mean_cd    = float(np.mean(cds))    if cds  else float('nan')
        median_cd  = float(np.median(cds))  if cds  else float('nan')
        prefix = f'eval/{mod}/{label}'
        out[f'{prefix}/IoU mean']          = mean_iou
        out[f'{prefix}/IoU median']        = median_iou
        out[f'{prefix}/CD mean']           = mean_cd
        out[f'{prefix}/CD median']         = median_cd
        out[f'{prefix}/Failures fraction'] = fail_frac
        print(f'  [{mod}/{label}] IoU={mean_iou:.3f}  CD={mean_cd:.2e}  Fail={fail_frac*100:.1f}%')

    return out


def run_validation(model, val_examples: list, processor, args) -> dict:
    """Greedy eval over all val_examples; returns W&B-ready metrics dict."""
    return eval_one_pass(
        model, val_examples, processor, args.max_new_tokens,
        eval_batch_size=getattr(args, 'eval_batch_size', 8),
        reward_workers=getattr(args, 'eval_workers', 2),   # Fix 5: separate eval_workers
        eval_timeout=getattr(args, 'eval_timeout', 120.0), # Fix 1: longer eval timeout
    )
