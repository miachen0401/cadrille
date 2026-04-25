"""Validation and logging utilities for RL training."""

import os
import sys
import random
import shutil
import subprocess
import csv
import numpy as np
import torch
from collections import defaultdict

from common.model import Cadrille, collate
from common.meshio import render_img
from common.metrics import compute_metrics

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    from common.datasets import mesh_to_point_cloud

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

    # Gradient checkpointing forces use_cache=False which breaks img generation:
    # prepare_inputs_for_generation sets pixel_values_videos=None for cache_position>0,
    # so the model loses the image after the first token.  Disable it for eval.
    had_gc = getattr(model, 'is_gradient_checkpointing', False)
    if had_gc:
        model.gradient_checkpointing_disable()

    try:
        # ── 1. Batched inference ──────────────────────────────────────────────
        # Sort by modality so pc and img examples are never mixed in the same batch.
        # Mixed batches cause heavy left-padding on pc examples to match the longer
        # img prompts (video tokens), which distorts img generation quality.
        examples = sorted(examples, key=lambda e: e.get('_modality', 'pc'))

        all_codes = [''] * len(examples)
        n = len(examples)
        for i in range(0, n, eval_batch_size):
            chunk = examples[i:i + eval_batch_size]
            collate_items = [{k: v for k, v in ex.items() if not k.startswith('_')}
                             for ex in chunk]
            batch = collate(collate_items, processor=processor, n_points=256, eval=True)

            # Reset cached rope_deltas before each generate() call.
            # Qwen2VL caches self.rope_deltas after the first img forward pass.
            # If stale (from a training step), get_rope_index() is skipped,
            # 2D spatial position IDs for video tokens are NOT computed,
            # and the model can't interpret images → img IoU ≈ 0.2.
            if hasattr(model, 'rope_deltas'):
                model.rope_deltas = None

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
                top_k=None,
                bad_words_ids=[[model.config.video_token_id]])

            prompt_len = batch['input_ids'].shape[1]
            for j in range(len(chunk)):
                code = processor.decode(generated_ids[j, prompt_len:], skip_special_tokens=True)
                all_codes[i + j] = code

        # ── 2. Parallel reward computation ───────────────────────────────────
        buckets = defaultdict(lambda: {'ious': [], 'cds': [], 'failures': 0, 'total': 0})

        def _score(idx):
            ex = examples[idx]
            iou_reward, cd = compute_metrics(
                all_codes[idx], ex['gt_mesh_path'],
                timeout=eval_timeout,
                use_pool=True,
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
                if iou_reward <= -1.0:
                    buckets[key]['failures'] += 1
                else:
                    buckets[key]['ious'].append(iou_reward)
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

    finally:
        if had_gc:
            model.gradient_checkpointing_enable()


def _run_img_eval_subprocess(model, processor, img_examples: list, args) -> dict:
    """Run img-mode eval via a fresh subprocess to avoid CUDA state contamination.

    The training process accumulates CUDA state (kernel warm-up, memory layout)
    that causes model.generate() to produce systematically different (worse) code
    than a fresh process loading the same checkpoint.  Running eval_img.py as a
    subprocess with a freshly loaded model reproduces the eval_img.py accuracy.

    Memory: saves model to a temp checkpoint, offloads params to CPU (freeing
    ~4.4 GB VRAM), runs the subprocess (~5 GB), then restores params to GPU.
    Optimizer states (~8.8 GB) stay on GPU throughout — total GPU peak ≈ 14 GB,
    within the 16 GB 4080 SUPER budget.
    """
    if not img_examples:
        return {}

    device = next(model.parameters()).device
    eval_img_py = os.path.join(_REPO_ROOT, 'tools', 'eval_img.py')
    output_dir  = getattr(args, 'output_dir', '/tmp')
    ckpt_dir    = os.path.join(output_dir, '_eval_tmp_checkpoint')
    out_root    = os.path.join(output_dir, '_eval_tmp_img')
    out = {}

    try:
        # 1. Save current model weights to a temp checkpoint.
        print('[eval/img] Saving temp checkpoint for subprocess eval ...', flush=True)
        model.save_pretrained(ckpt_dir)
        # processor is loaded from HF hub inside eval_img.py — no need to save it.

        # 2. Offload model params to CPU to free VRAM for the subprocess.
        model.cpu()
        torch.cuda.empty_cache()
        print('[eval/img] Model offloaded to CPU; VRAM freed.', flush=True)

        # 3. Group img examples by dataset (gt_dir as key).
        groups = {}
        for e in img_examples:
            gt_dir = os.path.dirname(e['gt_mesh_path'])
            label  = e.get('_dataset_label', 'DeepCAD test')
            if gt_dir not in groups:
                groups[gt_dir] = {'label': label, 'n': 0}
            groups[gt_dir]['n'] += 1

        # 4. Run eval_img.py once per dataset group.
        for gt_dir, info in groups.items():
            label     = info['label']
            n_samples = info['n']
            short     = 'deepcad' if 'deepcad' in label.lower() else 'fusion360'

            print(f'[eval/img] {label}: eval_img.py ({n_samples} samples) ...', flush=True)
            subprocess.run([
                sys.executable, eval_img_py,
                '--checkpoint',    ckpt_dir,
                '--splits',        f'{short}:{gt_dir}',
                '--n-samples',     str(n_samples),
                '--out-dir',       out_root,
                '--batch-size',    str(getattr(args, 'eval_batch_size', 8)),
                '--max-new-tokens', str(args.max_new_tokens),
                '--seed',          '42',
            ], timeout=3600)

            # 5. Parse results.csv produced by evaluate.py.
            results_csv = os.path.join(out_root, short, 'results.csv')
            if not os.path.exists(results_csv):
                print(f'[eval/img] WARNING: {results_csv} not found — skipping', flush=True)
                continue

            ious, cds, failures = [], [], 0
            with open(results_csv) as f:
                for row in csv.DictReader(f):
                    try:
                        iou = float(row['iou']) if row.get('iou') not in ('', 'None', None) else None
                        cd  = float(row['cd'])  if row.get('cd')  not in ('', 'None', None) else None
                    except (ValueError, KeyError):
                        iou, cd = None, None
                    if iou is None:
                        failures += 1
                    else:
                        ious.append(iou)
                        if cd is not None:
                            cds.append(cd)

            total  = len(ious) + failures
            prefix = f'eval/img/{label}'
            out[f'{prefix}/IoU mean']          = float(np.mean(ious))   if ious else 0.0
            out[f'{prefix}/IoU median']        = float(np.median(ious)) if ious else 0.0
            out[f'{prefix}/CD mean']           = float(np.mean(cds))    if cds  else float('nan')
            out[f'{prefix}/CD median']         = float(np.median(cds))  if cds  else float('nan')
            out[f'{prefix}/Failures fraction'] = failures / total        if total else 0.0
            print(f'  [img/{label}] IoU={out[f"{prefix}/IoU mean"]:.3f}  '
                  f'Fail={out[f"{prefix}/Failures fraction"]*100:.1f}%', flush=True)

    finally:
        # 6. Restore model params to GPU (optimizer states never left GPU).
        model.to(device)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(out_root,  ignore_errors=True)

    return out


def run_validation(model, val_examples: list, processor, args) -> dict:
    """Greedy eval over all val_examples; returns W&B-ready metrics dict.

    PC examples: evaluated inline via eval_one_pass.
    IMG examples: evaluated via _run_img_eval_subprocess (fresh subprocess with
    the correct Qwen2VL fast processor loaded from base model).  The cadrille-sft
    checkpoint ships a SLOW Qwen2VLImageProcessor whose __init__ always overwrites
    size.shortest_edge with the min_pixels kwarg (200704 → 32×32 tiles, 1024 tokens).
    The fast processor (loaded inside eval_img.py from Qwen2-VL-2B-Instruct) keeps
    the correct size.shortest_edge=3136 → 20×20 tiles, 400 tokens → correct IoU.
    """
    if not val_examples:
        return {}

    pc_examples  = [e for e in val_examples if e.get('_modality', 'pc') == 'pc']
    img_examples = [e for e in val_examples if e.get('_modality') == 'img']

    out = {}
    if pc_examples:
        out.update(eval_one_pass(
            model, pc_examples, processor, args.max_new_tokens,
            eval_batch_size=getattr(args, 'eval_batch_size', 8),
            reward_workers=getattr(args, 'eval_workers', 2),
            eval_timeout=getattr(args, 'eval_timeout', 120.0),
        ))
    if img_examples:
        out.update(_run_img_eval_subprocess(model, processor, img_examples, args))
    return out
