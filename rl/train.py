"""RL fine-tuning of Cadrille using Dr. CPPO (GRPO) — matches official rl branch.

The official implementation lives at col14m/cadrille (rl branch) in rl_finetune/.
This script mirrors that design for single-GPU usage while keeping the same
algorithm, hyperparameters, and reward function.

Algorithm: Dr. CPPO / GRPO
  Per step:
    1. Generate G completions for one prompt (temperature=1.0)
    2. Compute IoU rewards → advantages = r - mean(r)
    3. Select top_N completions by |advantage|
    4. Compute old-policy log probs (no grad)
    5. Do batch_updates gradient steps on the same rollout data (Adam)
       loss = -E[min(ratio*A, clip(ratio, 1-eps_low, 1+eps_high)*A)]

Official hyperparameters (from rl_finetune/train_cadrille_grpo.py):
  G=16, top_N=4, lr=3e-5, eps=0.1, batch_updates=3, max_new_tokens=400

W&B keys (match official cadrille rl branch naming)
----------------------------------------------------
  Per step (logged every log_steps):
    loss                         PPO clip loss (matches official)
    average_reward               mean of G rewards (matches official)
    train/reward_std             reward diversity (low → collapse)
    train/entropy                mean per-token entropy
    train/clip_fraction          fraction of ratios clipped
    train/ratio_mean             ratio diagnostics
    train/lr                     current learning rate

  Eval (logged every eval_steps):
    eval/pc/DeepCAD test/IoU mean        matches official dashboard exactly
    eval/pc/DeepCAD test/IoU median
    eval/pc/DeepCAD test/CD mean
    eval/pc/DeepCAD test/CD median
    eval/pc/DeepCAD test/Failures fraction

Usage
-----
# GRPO / Dr. CPPO using YAML config
python rl/train.py --config configs/rl/default.yaml --run-name cadrille-rl-v1

# Override specific values
python rl/train.py --config configs/rl/default.yaml --run-name test --max-steps 100

# 4080 single-GPU config
python rl/train.py --config configs/rl/4080.yaml
"""

import os
import sys

# Allow standalone execution from repo root or rl/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import copy
import json
import pickle
import random
import argparse
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from rl.reward import compute_reward, compute_rewards_parallel, compute_metrics

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _p(cli_val, cfg_val, default):
    """Resolve: CLI override > YAML config > hardcoded default. None = not set."""
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default


def _fmt_lr(lr: float) -> str:
    if lr == 0:
        return 'lr0'
    exp = int(math.floor(math.log10(abs(lr))))
    mantissa = lr / (10 ** exp)
    return f'lr{mantissa:.2g}e{exp}'


def _fmt_steps(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f'{n // 1000}k'
    return str(n)


def _auto_run_name_rl(mode, max_steps, lr, G) -> str:
    """Generate descriptive RL run name.

    Format: rl-s{steps}-{lr}-G{G}-{mode}-{MMDD}-{HHMM}
    Example: rl-s50k-lr3e-5-G16-cppo-0228-1045
    """
    ts = datetime.now().strftime('%m%d-%H%M')
    return f'rl-s{_fmt_steps(max_steps)}-{_fmt_lr(lr)}-G{G}-{mode}-{ts}'


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MeshDataset:
    """Load GT meshes directly from a directory of .stl files.

    This is the primary dataset for RL fine-tuning on real handcrafted meshes
    (e.g. DeepCAD train, Fusion360 train), matching the official implementation
    which trains on deepcad_fusion_train rather than hard-mined synthetic data.
    """

    def __init__(self, data_dir: str, n_points: int = 256,
                 noise_scale: float = 0.01, size: int = None):
        import trimesh
        from dataset import mesh_to_point_cloud

        stl_files = sorted(glob(os.path.join(data_dir, '**', '*.stl'), recursive=True)
                           + glob(os.path.join(data_dir, '*.stl')))
        if size is not None:
            rng = random.Random(42)
            rng.shuffle(stl_files)
            stl_files = stl_files[:size]

        self.examples = []
        print(f'Loading {len(stl_files)} meshes from {data_dir} ...')
        for path in tqdm(stl_files, desc='mesh→pc'):
            try:
                mesh = trimesh.load(path)
                pc = mesh_to_point_cloud(mesh, n_points)
                pc = (pc - 0.5) * 2  # match test-split normalisation
                if noise_scale > 0:
                    pc = pc + np.random.randn(*pc.shape).astype(np.float32) * noise_scale
                self.examples.append({
                    'point_cloud': pc,
                    'description': 'Generate cadquery code',
                    'file_name': os.path.splitext(os.path.basename(path))[0],
                    'gt_mesh_path': path,
                })
            except Exception:
                pass
        print(f'  → loaded {len(self.examples)} valid examples')

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class RLDataset:
    """Loads hard-mined examples from rl/mine.py output pkl."""

    def __init__(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            self.examples = pickle.load(f)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ex = self.examples[index]
        item = {
            'description': 'Generate cadquery code',
            'file_name': ex['file_name'],
            'gt_mesh_path': ex['gt_mesh_path'],
        }
        if ex.get('is_pc', True):
            item['point_cloud'] = ex['point_cloud']
        else:
            item.update(_render_img(ex['gt_mesh_path']))
        return item


class DPODataset:
    """Precomputed preference pairs for DPO training.

    JSONL: {"description", "point_cloud"|null, "file_name", "gt_mesh_path",
            "y_w", "y_l", "ref_logp_w", "ref_logp_l"}
    """

    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        item = {
            'description': rec['description'],
            'file_name': rec['file_name'],
            'gt_mesh_path': rec['gt_mesh_path'],
            'y_w': rec['y_w'],
            'y_l': rec['y_l'],
            'ref_logp_w': float(rec['ref_logp_w']),
            'ref_logp_l': float(rec['ref_logp_l']),
        }
        if rec.get('point_cloud') is not None:
            item['point_cloud'] = np.array(rec['point_cloud'], dtype=np.float32)
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_img(gt_mesh_path: str) -> dict:
    """Render 4-view image grid from a mesh path (image-mode examples)."""
    import trimesh
    import open3d
    from PIL import Image, ImageOps
    from dataset import mesh_to_image

    mesh = trimesh.load(gt_mesh_path)
    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = open3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
    o3d_mesh.compute_vertex_normals()
    fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
    imgs = [ImageOps.expand(mesh_to_image(o3d_mesh, camera_distance=-0.9,
                                          front=f, img_size=128),
                            border=3, fill='black')
            for f in fronts]
    combined = Image.fromarray(np.vstack((
        np.hstack((np.array(imgs[0]), np.array(imgs[1]))),
        np.hstack((np.array(imgs[2]), np.array(imgs[3]))))))
    return {'video': [combined]}


def expand_batch(batch: dict, G: int) -> dict:
    """Replicate every tensor/list in batch G times along the batch axis."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.repeat(G, *([1] * (v.dim() - 1)))
        elif isinstance(v, list):
            out[k] = v * G
        else:
            out[k] = v
    return out


def slice_batch(batch: dict, indices: torch.Tensor) -> dict:
    """Index into a batch dict along the first axis using *indices*."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[indices]
        elif isinstance(v, list):
            out[k] = [v[i] for i in indices.tolist()]
        else:
            out[k] = v
    return out


def make_labels(full_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    """Build label tensor: -100 for prompt tokens and post-EOS tokens."""
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    for i in range(full_ids.shape[0]):
        eos_pos = (full_ids[i, prompt_len:] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            labels[i, prompt_len + eos_pos[0].item() + 1:] = -100
    return labels


def _model_forward(model, full_ids, g_batch, device):
    """Forward pass returning model output."""
    return model(
        input_ids=full_ids.to(device),
        attention_mask=torch.ones_like(full_ids).to(device),
        labels=None,
        point_clouds=g_batch['point_clouds'].to(device),
        is_pc=g_batch['is_pc'].to(device),
        is_img=g_batch['is_img'].to(device),
        pixel_values_videos=(
            g_batch['pixel_values_videos'].to(device)
            if g_batch.get('pixel_values_videos') is not None else None),
        video_grid_thw=(
            g_batch['video_grid_thw'].to(device)
            if g_batch.get('video_grid_thw') is not None else None),
    )


def compute_policy_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean per-token entropy over unmasked (completion) positions."""
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100)
    log_p = F.log_softmax(shift_logits, dim=-1)
    token_entropy = -(torch.exp(log_p) * log_p).sum(dim=-1)
    return (token_entropy * mask.float()).sum() / mask.float().sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def load_val_examples(split_dir: str, n_samples: int, n_points: int = 256) -> list:
    """Sample n_samples GT meshes from split_dir with a fixed seed."""
    import trimesh
    from dataset import mesh_to_point_cloud

    stl_files = sorted(f for f in os.listdir(split_dir) if f.endswith('.stl'))
    rng = random.Random(42)
    rng.shuffle(stl_files)

    examples = []
    for fname in stl_files[:n_samples * 2]:
        if len(examples) >= n_samples:
            break
        gt_mesh_path = os.path.join(split_dir, fname)
        try:
            mesh = trimesh.load(gt_mesh_path)
            pc = mesh_to_point_cloud(mesh, n_points)
            pc = (pc - 0.5) * 2
            examples.append({
                'point_cloud': pc,
                'description': 'Generate cadquery code',
                'file_name': fname[:-4],
                'gt_mesh_path': gt_mesh_path,
            })
        except Exception:
            pass

    print(f'Loaded {len(examples)} validation examples from {split_dir}')
    return examples


@torch.no_grad()
def run_validation(model, val_examples: list, processor, args) -> dict:
    """Greedy decode one completion per val example; return IoU + CD metrics.

    Returns dict with keys matching official W&B dashboard naming:
      eval/pc/DeepCAD test/IoU mean
      eval/pc/DeepCAD test/IoU median
      eval/pc/DeepCAD test/CD mean
      eval/pc/DeepCAD test/CD median
      eval/pc/DeepCAD test/Failures fraction
    """
    device = next(model.parameters()).device
    model.eval()
    ious = []
    cds = []
    failures = 0

    for item in val_examples:
        batch = collate([item], processor=processor, n_points=256, eval=True)
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
            do_sample=False)

        prompt_len = batch['input_ids'].shape[1]
        code = processor.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        iou_reward, cd = compute_metrics(code, item['gt_mesh_path'], timeout=30.0)

        if iou_reward <= -10.0:
            failures += 1
        else:
            ious.append(iou_reward / 10.0)   # convert reward back to [0,1] IoU
            if cd is not None:
                cds.append(cd)

    n = len(val_examples)
    failure_frac = failures / n if n > 0 else 0.0
    mean_iou    = float(np.mean(ious))    if ious else 0.0
    median_iou  = float(np.median(ious))  if ious else 0.0
    mean_cd     = float(np.mean(cds))     if cds  else float('nan')
    median_cd   = float(np.median(cds))   if cds  else float('nan')

    print(f'  IoU mean={mean_iou:.3f}  CD mean={mean_cd:.4f}  Failures={failure_frac*100:.1f}%')

    return {
        'eval/pc/DeepCAD test/IoU mean':          mean_iou,
        'eval/pc/DeepCAD test/IoU median':        median_iou,
        'eval/pc/DeepCAD test/CD mean':           mean_cd,
        'eval/pc/DeepCAD test/CD median':         median_cd,
        'eval/pc/DeepCAD test/Failures fraction': failure_frac,
    }


# ---------------------------------------------------------------------------
# Dr. CPPO / GRPO step  (matches official grpo_mm.py)
# ---------------------------------------------------------------------------

def cppo_step(model, old_model, optimizer, item, processor, args) -> dict:
    """One Dr. CPPO update: rollout G completions, select top_N, do batch_updates
    gradient steps.  Matches the official grpo_mm.py train_with_grpo_mm logic.

    Returns metrics dict from the last gradient update.
    """
    G = args.G
    N = min(args.top_N, G)
    device = next(model.parameters()).device

    # --- 1. Prepare prompt ---
    batch = collate([item], processor=processor, n_points=256, eval=True)
    gt_mesh_path = item['gt_mesh_path']

    # --- 2. Generate G completions one at a time (avoids batch-G peak memory) ---
    # Generating batch_size=G simultaneously requires G × activation memory, which
    # OOMs on 16 GB. Sequential generation uses batch_size=1, trading speed for memory.
    single_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    model.eval()
    generated_ids_list = []
    with torch.no_grad():
        for _ in range(G):
            ids = model.generate(
                input_ids=single_batch['input_ids'],
                attention_mask=single_batch['attention_mask'],
                point_clouds=single_batch['point_clouds'],
                is_pc=single_batch['is_pc'],
                is_img=single_batch['is_img'],
                pixel_values_videos=(
                    single_batch['pixel_values_videos']
                    if single_batch.get('pixel_values_videos') is not None else None),
                video_grid_thw=(
                    single_batch['video_grid_thw']
                    if single_batch.get('video_grid_thw') is not None else None),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=50)
            generated_ids_list.append(ids.cpu())  # move to CPU to free GPU cache

    # Pad all sequences to the same length before stacking
    prompt_len = batch['input_ids'].shape[1]
    max_len = max(ids.shape[1] for ids in generated_ids_list)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    padded = []
    for ids in generated_ids_list:
        if ids.shape[1] < max_len:
            pad = torch.full((1, max_len - ids.shape[1]), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=1)
        padded.append(ids)
    generated_ids = torch.cat(padded, dim=0)  # [G, max_len] on CPU

    code_strings = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)

    # --- 3. Rewards and advantages ---
    rewards = compute_rewards_parallel(
        code_strings, [gt_mesh_path] * G, workers=args.reward_workers)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std()
    advantages_raw = rewards_t - mean_r  # unnormalised, used for top-N selection

    # --- 4. Select top_N by |advantage| (matches official top_samples selection) ---
    _, top_idx = torch.topk(advantages_raw.abs(), N)

    eos_id = processor.tokenizer.eos_token_id
    sel_ids_cpu = generated_ids[top_idx]  # [N, max_len] stays on CPU for now
    sel_labels_cpu = make_labels(sel_ids_cpu, prompt_len, eos_id)
    # Build the prompt-only batch for the N selected items (still on CPU)
    g_batch = expand_batch(batch, G)   # used only for slicing, all CPU
    sel_g_batch_cpu = slice_batch(g_batch, top_idx)
    advantages = (advantages_raw[top_idx] / (std_r + 1e-8))

    # --- 5. Old-policy log probs (no grad; old_model on CPU, move to GPU briefly) ---
    torch.cuda.empty_cache()  # free KV cache from generation before loading old model
    sel_ids = sel_ids_cpu.to(device)
    sel_labels = sel_labels_cpu.to(device)
    sel_g_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in sel_g_batch_cpu.items()}
    advantages = advantages.to(device)

    old_model.to(device)
    old_model.eval()
    with torch.no_grad():
        old_out = _model_forward(old_model, sel_ids, sel_g_batch, device)
        old_log_probs = Cadrille.compute_sequence_logprob(
            old_out.logits, sel_labels, mean_reduction=True).detach()
    old_model.cpu()
    torch.cuda.empty_cache()  # free old_model activations before backward

    # --- 6. batch_updates gradient steps on the same rollout data ---
    last_metrics = {}
    for _ in range(args.batch_updates):
        model.train()
        new_out = _model_forward(model, sel_ids, sel_g_batch, device)
        new_log_probs = Cadrille.compute_sequence_logprob(
            new_out.logits, sel_labels, mean_reduction=True)
        entropy = compute_policy_entropy(new_out.logits, sel_labels)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - args.eps_low, 1.0 + args.eps_high)
        loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_metrics = {
            'train/loss':          loss.item(),
            'train/mean_reward':   mean_r.item(),
            'train/reward_std':    std_r.item(),
            'train/entropy':       entropy.item(),
            'train/clip_fraction': ((ratio < 1 - args.eps_low) |
                                    (ratio > 1 + args.eps_high)).float().mean().item(),
            'train/ratio_mean':    ratio.mean().item(),
            'train/ratio_std':     ratio.std().item(),
        }

    return last_metrics


# ---------------------------------------------------------------------------
# DPO step
# ---------------------------------------------------------------------------

def _collate_with_completion(item, completion, processor):
    training_item = {k: v for k, v in item.items()
                     if k not in ('gt_mesh_path', 'y_w', 'y_l',
                                  'ref_logp_w', 'ref_logp_l')}
    training_item['answer'] = completion
    return collate([training_item], processor=processor, n_points=256, eval=False)


def dpo_step(model, optimizer, item, processor, args) -> dict:
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
    loss = -F.logsigmoid(margin)

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


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args, cfg_to_save=None):
    os.makedirs(args.output_dir, exist_ok=True)

    # Save resolved config for traceability
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
                    name=args.wandb_run_name or getattr(args, 'run_name', None),
                    entity=args.wandb_entity or None,
                    config=cfg_to_save or vars(args),
                    mode='offline' if args.wandb_offline else 'online',
                    resume='allow',
                    settings=wandb.Settings(console='off'),
                )
                # Save run URL alongside checkpoint for traceability
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

    # 8-bit Adam: reduces optimizer memory from ~16 GB (fp32) to ~2 GB (int8).
    # Required to fit RL training on 16 GB GPU alongside model + activations.
    # Falls back to standard Adam if bitsandbytes unavailable.
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
        print('Optimizer: Adam8bit (bitsandbytes)')
    except ImportError:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print('Optimizer: Adam (fp32) — bitsandbytes not found; may OOM on 16 GB GPU')

    # Validation
    val_examples = []
    if args.val_split_dir and os.path.isdir(args.val_split_dir):
        val_examples = load_val_examples(args.val_split_dir, args.val_samples)
    else:
        print('No --val-split-dir provided; skipping validation.')

    if args.mode == 'cppo':
        if args.data_dir:
            dataset = MeshDataset(args.data_dir, noise_scale=0.01)
        elif args.hard_examples_pkl:
            dataset = RLDataset(args.hard_examples_pkl)
        else:
            raise ValueError('Provide --data-dir (real meshes) or --hard-examples-pkl')

        old_model = copy.deepcopy(model).cpu()
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad_(False)
        _train_cppo(model, old_model, optimizer, dataset, processor,
                    val_examples, use_wandb, args)

    elif args.mode == 'dpo':
        if not args.dpo_dataset:
            raise ValueError('--dpo-dataset is required for DPO mode')
        dataset = DPODataset(args.dpo_dataset)
        _train_dpo(model, optimizer, dataset, processor,
                   val_examples, use_wandb, args)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    if use_wandb:
        wandb.finish()


def _train_cppo(model, old_model, optimizer, dataset, processor,
                val_examples, use_wandb, args):
    log_path = os.path.join(args.output_dir, 'log.txt')
    step = 0
    indices = list(range(len(dataset)))

    pbar = tqdm(total=args.max_steps, desc='Dr. CPPO')
    while step < args.max_steps:
        np.random.shuffle(indices)
        for idx in indices:
            if step >= args.max_steps:
                break
            try:
                metrics = cppo_step(
                    model, old_model, optimizer, dataset[idx], processor, args)
            except Exception as e:
                print(f'[step {step}] cppo_step error: {e}')
                continue

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{metrics['train/loss']:.3f}",
                reward=f"{metrics['train/mean_reward']:.2f}",
                H=f"{metrics['train/entropy']:.2f}")

            if step % args.log_steps == 0:
                # Log to file with official-style key names
                log_line = (
                    f"step={step}"
                    f" loss={metrics['train/loss']:.4f}"
                    f" average_reward={metrics['train/mean_reward']:.4f}"
                    f" train/reward_std={metrics['train/reward_std']:.4f}"
                    f" train/entropy={metrics['train/entropy']:.4f}"
                    f" train/clip_fraction={metrics['train/clip_fraction']:.4f}"
                    f" train/lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                with open(log_path, 'a') as f:
                    f.write(log_line + '\n')

                if use_wandb:
                    wandb.log({
                        # Official top-level key names (match paper dashboard)
                        'loss':           metrics['train/loss'],
                        'average_reward': metrics['train/mean_reward'],
                        # Enhanced diagnostics
                        'train/reward_std':    metrics['train/reward_std'],
                        'train/entropy':       metrics['train/entropy'],
                        'train/clip_fraction': metrics['train/clip_fraction'],
                        'train/ratio_mean':    metrics['train/ratio_mean'],
                        'train/lr':            optimizer.param_groups[0]['lr'],
                    }, step=step)

            if val_examples and step % args.eval_steps == 0:
                print(f'\n[eval step={step}]')
                val_metrics = run_validation(model, val_examples, processor, args)
                # Log to file
                log_line = (
                    f"step={step}"
                    + ''.join(f" {k}={v:.4f}" for k, v in val_metrics.items())
                )
                with open(log_path, 'a') as f:
                    f.write(log_line + '\n')
                if use_wandb:
                    wandb.log(val_metrics, step=step)
                model.train()

            # Sync old policy
            if step % args.K_update == 0:
                old_model.load_state_dict(
                    {k: v.cpu() for k, v in model.state_dict().items()})

            if step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

    pbar.close()
    final_dir = os.path.join(args.output_dir, 'checkpoint-final')
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f'Training complete. Final checkpoint → {final_dir}')


def _train_dpo(model, optimizer, dataset, processor,
               val_examples, use_wandb, args):
    log_path = os.path.join(args.output_dir, 'log.txt')
    step = 0
    epoch = 0

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
                log_line = (
                    f"step={step}"
                    + ''.join(f" {k}={v:.4f}" for k, v in val_metrics.items())
                )
                with open(log_path, 'a') as f:
                    f.write(log_line + '\n')
                if use_wandb:
                    wandb.log(val_metrics, step=step)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RL fine-tuning of Cadrille (Dr. CPPO / GRPO or DPO). '
                    'All settings can be defined in a YAML config; CLI flags override.')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (e.g. configs/rl/default.yaml)')

    # Run identity
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name → saved to {checkpoints-dir}/{run-name}/. '
                             'Overrides run_name from config.')
    parser.add_argument('--checkpoints-dir', type=str, default=None,
                        help='Root directory for checkpoints (default: ./checkpoints)')

    # Common
    parser.add_argument('--mode', type=str, default=None, choices=['cppo', 'dpo'])
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (official: 3e-5 with Adam)')
    parser.add_argument('--log-steps', type=int, default=None)
    parser.add_argument('--save-steps', type=int, default=None)

    # Validation
    parser.add_argument('--val-split-dir', type=str, default=None)
    parser.add_argument('--val-samples', type=int, default=None)
    parser.add_argument('--eval-steps', type=int, default=None)

    # W&B
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-offline', action='store_true')

    # CPPO / GRPO
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory of GT .stl meshes for RL (real handcrafted, '
                             'e.g. ./data/deepcad_test_mesh). Preferred over --hard-examples-pkl.')
    parser.add_argument('--hard-examples-pkl', type=str, default=None,
                        help='Hard-mined examples pkl from rl/mine.py (legacy)')
    parser.add_argument('--G', type=int, default=None,
                        help='Number of rollout completions per step (official: 16)')
    parser.add_argument('--top-N', type=int, default=None,
                        help='Select top-N by |advantage| for gradient update (official: 4)')
    parser.add_argument('--eps-high', type=float, default=None,
                        help='PPO clip upper bound (official: 0.1)')
    parser.add_argument('--eps-low', type=float, default=None,
                        help='PPO clip lower bound (official: 0.1)')
    parser.add_argument('--batch-updates', type=int, default=None,
                        help='Gradient steps per rollout (official: 3)')
    parser.add_argument('--K-update', type=int, default=None,
                        help='Copy new → old policy every K_update steps')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help='Max generation length (official: 400)')
    parser.add_argument('--reward-workers', type=int, default=None)

    # DPO
    parser.add_argument('--dpo-dataset', type=str, default=None)
    parser.add_argument('--beta', type=float, default=None,
                        help='DPO KL coefficient (official Table 13: 0.3)')
    parser.add_argument('--dpo-epochs-per-round', type=int, default=None)

    args = parser.parse_args()

    # Load YAML config if provided
    cfg = _load_yaml(args.config) if args.config else {}

    # Resolve all params first (needed for auto run name)
    args.mode             = _p(args.mode,             cfg.get('mode'),             'cppo')
    args.checkpoint_path  = _p(args.checkpoint_path,  cfg.get('checkpoint_path'),  'maksimko123/cadrille')
    args.max_steps        = _p(args.max_steps,        cfg.get('max_steps'),        50000)
    args.lr               = _p(args.lr,               cfg.get('lr'),               3e-5)
    args.log_steps        = _p(args.log_steps,        cfg.get('log_steps'),        100)
    args.save_steps       = _p(args.save_steps,       cfg.get('save_steps'),       5000)
    args.val_split_dir    = _p(args.val_split_dir,    cfg.get('val_split_dir'),    None)
    args.val_samples      = _p(args.val_samples,      cfg.get('val_samples'),      50)
    args.eval_steps       = _p(args.eval_steps,       cfg.get('eval_steps'),       500)
    args.wandb_project    = _p(args.wandb_project,    cfg.get('wandb_project'),    None)
    args.wandb_entity     = _p(args.wandb_entity,     cfg.get('wandb_entity'),     None)
    args.wandb_offline    = args.wandb_offline or bool(cfg.get('wandb_offline', False))
    args.data_dir         = _p(args.data_dir,         cfg.get('data_dir'),         None)
    args.hard_examples_pkl = _p(args.hard_examples_pkl, cfg.get('hard_examples_pkl'), None)
    # Official values: G=16, top_N=4, batch_updates=3, max_new_tokens=400
    # Defaults below are reduced for 16 GB GPU (RTX 4080). Override via YAML or CLI.
    args.G                = _p(args.G,                cfg.get('G'),                8)
    args.top_N            = _p(args.top_N,            cfg.get('top_N'),            4)
    args.eps_high         = _p(args.eps_high,         cfg.get('eps_high'),         0.1)
    args.eps_low          = _p(args.eps_low,          cfg.get('eps_low'),          0.1)
    args.batch_updates    = _p(args.batch_updates,    cfg.get('batch_updates'),    1)
    args.K_update         = _p(args.K_update,         cfg.get('K_update'),         10)
    args.max_new_tokens   = _p(args.max_new_tokens,   cfg.get('max_new_tokens'),   256)
    args.reward_workers   = _p(args.reward_workers,   cfg.get('reward_workers'),   4)
    args.dpo_dataset      = _p(args.dpo_dataset,      cfg.get('dpo_dataset'),      None)
    args.beta             = _p(args.beta,             cfg.get('beta'),             0.3)
    args.dpo_epochs_per_round = _p(args.dpo_epochs_per_round, cfg.get('dpo_epochs_per_round'), 10)

    # Resolve run identity — auto-generate from hyperparams if not specified
    checkpoints_dir = _p(args.checkpoints_dir, cfg.get('checkpoints_dir'), './checkpoints')
    run_name_base   = _p(args.run_name,         cfg.get('run_name'),         None)
    if run_name_base:
        run_name = run_name_base
    else:
        run_name = _auto_run_name_rl(args.mode, args.max_steps, args.lr, args.G)
    args.output_dir       = os.path.join(checkpoints_dir, run_name)
    args.wandb_run_name   = _p(args.wandb_run_name, cfg.get('wandb_run_name'), run_name)
    args.run_name         = run_name

    print(f'Run name : {run_name}')
    print(f'Output   : {args.output_dir}')

    # Resolved config dict for saving alongside the checkpoint
    resolved_cfg = {k: v for k, v in vars(args).items()
                    if not k.startswith('_') and k != 'config'}
    resolved_cfg['config_file'] = args.config

    train(args, cfg_to_save=resolved_cfg)
