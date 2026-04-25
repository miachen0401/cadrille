"""Online IoU eval during SFT — wandb metrics mirror the RL training loop.

At every `eval_steps` tick, greedy-generate on a fixed subset of each eval
dataset (benchcad val + deepcad test + fusion360 test), exec each code,
compute IoU / CD, and log per-dataset metrics under the same key names as
train/rl/eval.py:

    eval/img/{dataset}/IoU mean
    eval/img/{dataset}/IoU median
    eval/img/{dataset}/CD mean
    eval/img/{dataset}/CD median
    eval/img/{dataset}/Failures fraction

Wired via a TrainerCallback so the HF Trainer's default `eval_loss` path is
untouched — this runs on the same schedule but emits IoU-based metrics
alongside.
"""
from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from common.model import collate
from common.metrics import compute_metrics


_SUBSETS_DEFAULT = {
    'benchcad_val':       'data/benchcad/val',
    'deepcad_test':       'data/deepcad_test_mesh',
    'fusion360_test':     'data/fusion360_test_mesh',
}


def _seeded_sample(items, n, seed=42):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[:n]


def _load_benchcad_subset(root: str, n: int, seed: int) -> list[dict]:
    """Load N fixed benchcad val items as examples ({gt_mesh_path, video, ...})."""
    from PIL import Image
    pkl = Path(root).parent / 'val.pkl'
    if not pkl.exists():
        return []
    with pkl.open('rb') as f:
        rows = pickle.load(f)
    rows = _seeded_sample(rows, n, seed)
    root_p = Path(root).parent
    out = []
    for r in rows:
        png = root_p / r['png_path']
        stl = root_p / r['mesh_path']
        if not (png.exists() and stl.exists()):
            continue
        out.append({
            '_modality': 'img',
            '_dataset_label': 'BenchCAD val',
            'file_name': r['uid'],
            'gt_mesh_path': str(stl),
            'video': [Image.open(png).convert('RGB')],
            'description': 'Generate cadquery code',
        })
    return out


def _load_stl_dir_subset(root: str, label: str, n: int, seed: int) -> list[dict]:
    """Load N fixed DeepCAD/Fusion360 test items (img modality via _render.png)."""
    from PIL import Image
    rootp = Path(root)
    stls = sorted(rootp.glob('*.stl'))
    stls = _seeded_sample(stls, n, seed)
    out = []
    for stl in stls:
        png = stl.with_name(stl.stem + '_render.png')
        if not png.exists():
            continue
        out.append({
            '_modality': 'img',
            '_dataset_label': label,
            'file_name': stl.stem,
            'gt_mesh_path': str(stl),
            'video': [Image.open(png).convert('RGB')],
            'description': 'Generate cadquery code',
        })
    return out


def load_online_eval_subsets(n_per: int = 30, seed: int = 42,
                             subsets: dict | None = None) -> list[dict]:
    """Return combined example list; dataset_label distinguishes buckets."""
    subsets = subsets or _SUBSETS_DEFAULT
    bc = _load_benchcad_subset(subsets['benchcad_val'], n_per, seed)
    dc = _load_stl_dir_subset(subsets['deepcad_test'],   'DeepCAD test',   n_per, seed)
    f3 = _load_stl_dir_subset(subsets['fusion360_test'], 'Fusion360 test', n_per, seed)
    return bc + dc + f3


@torch.no_grad()
def run_online_eval(model, processor, examples: list[dict],
                    eval_batch_size: int = 8,
                    reward_workers: int = 8,
                    max_new_tokens: int = 768,
                    eval_timeout: float = 30.0) -> dict:
    """Greedy eval → per-(modality, dataset) IoU / CD / Failures. wandb-ready dict."""
    if not examples:
        return {}
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # Avoid GC interference with generate() image handling (mirror RL eval).
    had_gc = getattr(model, 'is_gradient_checkpointing', False)
    if had_gc:
        model.gradient_checkpointing_disable()

    try:
        all_codes = [''] * len(examples)
        n = len(examples)
        for i in range(0, n, eval_batch_size):
            chunk = examples[i:i + eval_batch_size]
            collate_items = [{k: v for k, v in ex.items() if not k.startswith('_')
                              and k != 'gt_mesh_path'} for ex in chunk]
            batch = collate(collate_items, processor=processor, n_points=256, eval=True)
            if hasattr(model, 'rope_deltas'):
                model.rope_deltas = None

            gen_kw = dict(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                point_clouds=batch['point_clouds'].to(device),
                is_pc=batch['is_pc'].to(device),
                is_img=batch['is_img'].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                bad_words_ids=[[model.config.video_token_id]],
            )
            if batch.get('pixel_values_videos') is not None:
                gen_kw['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
                gen_kw['video_grid_thw']       = batch['video_grid_thw'].to(device)

            generated_ids = model.generate(**gen_kw)
            prompt_len = batch['input_ids'].shape[1]
            for j in range(len(chunk)):
                all_codes[i + j] = processor.decode(
                    generated_ids[j, prompt_len:], skip_special_tokens=True)

        buckets = defaultdict(lambda: {'ious': [], 'cds': [], 'failures': 0, 'total': 0})

        def _score(idx):
            ex = examples[idx]
            iou_reward, cd = compute_metrics(
                all_codes[idx], ex['gt_mesh_path'],
                timeout=eval_timeout, use_pool=True)
            return idx, iou_reward, cd

        with ThreadPoolExecutor(max_workers=reward_workers) as pool:
            futures = [pool.submit(_score, i) for i in range(n)]
            for fut in as_completed(futures):
                idx, iou_reward, cd = fut.result()
                ex = examples[idx]
                key = (ex.get('_modality', 'img'), ex.get('_dataset_label', '?'))
                buckets[key]['total'] += 1
                if iou_reward <= -1.0:
                    buckets[key]['failures'] += 1
                else:
                    buckets[key]['ious'].append(iou_reward)
                    if cd is not None:
                        buckets[key]['cds'].append(cd)

        out = {}
        for (mod, label), b in buckets.items():
            ious, cds = b['ious'], b['cds']
            fail_frac  = b['failures'] / b['total'] if b['total'] else 0.0
            exec_rate  = 1.0 - fail_frac
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
            out[f'{prefix}/exec_rate']         = exec_rate
            print(f'  [{mod}/{label}] IoU={mean_iou:.3f}  '
                  f'exec={exec_rate*100:.1f}%  (n={b["total"]})', flush=True)
        return out

    finally:
        if had_gc:
            model.gradient_checkpointing_enable()
        if was_training:
            model.train()


class OnlineIoUEvalCallback(TrainerCallback):
    """Run IoU eval every `eval_steps` on the same schedule as HF Trainer's loss eval.

    Uses HF Trainer's on_evaluate hook so metrics land in the *same* log dict that
    eval_loss goes into — wandb picks them up without an extra call.
    """

    def __init__(self, processor, n_per_dataset: int = 20, seed: int = 42,
                 subsets: dict | None = None, eval_batch_size: int = 8,
                 reward_workers: int = 8, max_new_tokens: int = 768,
                 eval_timeout: float = 30.0):
        self.processor = processor
        self.n_per_dataset = n_per_dataset
        self.seed = seed
        self.subsets = subsets
        self.eval_batch_size = eval_batch_size
        self.reward_workers = reward_workers
        self.max_new_tokens = max_new_tokens
        self.eval_timeout = eval_timeout
        self._examples = None

    def _ensure_examples(self):
        if self._examples is None:
            self._examples = load_online_eval_subsets(
                n_per=self.n_per_dataset, seed=self.seed, subsets=self.subsets)
            print(f'[online-eval] loaded {len(self._examples)} examples across datasets',
                  flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after HF Trainer's default eval_loss pass. Append our IoU metrics."""
        if not state.is_world_process_zero:
            return
        if metrics is None:
            return
        model = kwargs.get('model')
        if model is None:
            return
        self._ensure_examples()
        print(f'[online-eval] step={state.global_step} running IoU eval ...', flush=True)
        iou_metrics = run_online_eval(
            model, self.processor, self._examples,
            eval_batch_size=self.eval_batch_size,
            reward_workers=self.reward_workers,
            max_new_tokens=self.max_new_tokens,
            eval_timeout=self.eval_timeout,
        )
        # Inject into the same metrics dict Trainer will log — wandb sees them.
        metrics.update(iou_metrics)
