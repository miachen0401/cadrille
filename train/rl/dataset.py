"""RL-specific dataset classes.

Shared mesh→input helpers (`render_img`, `MeshDataset`) live in `common.meshio`.
RL training loads hard-mined examples (pkl) / preference pairs (jsonl) via the
three classes below.
"""
import os
import json
import pickle

import numpy as np

from common.meshio import render_img  # used by RLDataset.__getitem__ img branch

__all__ = ['RLDataset', 'CurriculumRLDataset', 'DPODataset']


class RLDataset:
    """Loads hard-mined examples from rl/mine.py output pkl.

    modality='img' (default): renders 4-view image on every __getitem__ call.
    modality='pc': generates point cloud on every __getitem__ call (lazy, no upfront cost).
    """

    def __init__(self, pkl_path: str, modality: str = 'img', n_points: int = 256):
        with open(pkl_path, 'rb') as f:
            self.examples = pickle.load(f)
        self.modality = modality
        self.n_points = n_points

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ex = self.examples[index]
        item = {
            'description': 'Generate cadquery code',
            'file_name': ex['file_name'],
            'gt_mesh_path': ex['gt_mesh_path'],
            'modality': self.modality,
        }
        if self.modality == 'pc':
            import trimesh
            from dataset import mesh_to_point_cloud
            mesh = trimesh.load(ex['gt_mesh_path'])
            pc = mesh_to_point_cloud(mesh, self.n_points)
            pc = (pc - 0.5) * 2
            item['point_cloud'] = pc
        else:
            item.update(render_img(ex['gt_mesh_path']))
        return item


def _load_scores(score_jsonl_paths: list) -> dict:
    """Load SFT IoU scores from mine.py output jsonl files → {file_name: mean_reward}."""
    scores = {}
    for path in score_jsonl_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    scores[d['file_name']] = d['mean_reward']
                except Exception:
                    pass
    return scores


class CurriculumRLDataset(RLDataset):
    """RLDataset with difficulty-based curriculum sampling.

    Enriches the pkl with SFT IoU scores from _hard_scores.jsonl files, then
    progressively expands the active training pool from easy to hard examples
    as training advances.

    Curriculum phases (based on SFT mean_reward from rl/mine.py):
      Phase 1 (steps 0 → phase2_step):       easy-hard only  (SFT IoU ∈ [0.5, 0.75])
      Phase 2 (phase2_step → phase3_step):   + medium-hard   (SFT IoU ≥ 0.3)
      Phase 3 (phase3_step → end):            full dataset    (all SFT IoU < R_th)

    If score files are not found, falls back to uniform sampling over all examples.

    Args:
        pkl_path:        Path to combined_hard.pkl.
        score_paths:     List of paths to *_hard_scores.jsonl files from rl/mine.py.
        phase2_step:     Training step at which Phase 2 begins.
        phase3_step:     Training step at which Phase 3 begins.
        easy_threshold:  Upper IoU bound for "easy-hard" examples (default 0.75).
        medium_threshold: Lower IoU bound for Phase 2 examples (default 0.3).
        modality, n_points: Passed through to RLDataset.
    """

    def __init__(
        self,
        pkl_path: str,
        score_paths: list,
        phase2_step: int = 5000,
        phase3_step: int = 15000,
        easy_threshold: float = 0.75,
        medium_threshold: float = 0.3,
        modality: str = 'img',
        n_points: int = 256,
    ):
        super().__init__(pkl_path, modality=modality, n_points=n_points)
        self.phase2_step     = phase2_step
        self.phase3_step     = phase3_step
        self.easy_threshold  = easy_threshold
        self.medium_threshold = medium_threshold
        self._current_step   = 0

        # Join pkl examples with SFT IoU scores
        scores = _load_scores(score_paths)
        for ex in self.examples:
            ex['_sft_iou'] = scores.get(ex['file_name'], None)

        # Pre-compute tier indices (sorted by ascending difficulty = descending SFT IoU)
        easy   = [i for i, ex in enumerate(self.examples)
                  if ex['_sft_iou'] is not None and
                  self.medium_threshold <= ex['_sft_iou'] < self.easy_threshold]
        medium = [i for i, ex in enumerate(self.examples)
                  if ex['_sft_iou'] is not None and
                  0.0 <= ex['_sft_iou'] < self.medium_threshold]
        full   = list(range(len(self.examples)))   # includes invalid (-1) and unscored

        self._tiers = {
            'easy':   easy,
            'medium': easy + medium,
            'full':   full,
        }
        self._active_indices = self._tiers['easy'] if easy else full
        n_easy = len(easy)
        n_med  = len(easy) + len(medium)
        print(f'CurriculumRLDataset: {len(full)} total  '
              f'| Phase1 easy={n_easy}  Phase2 easy+med={n_med}  Phase3 full={len(full)}')
        print(f'  Curriculum: Phase2@step{phase2_step}  Phase3@step{phase3_step}')

    def set_step(self, step: int) -> str:
        """Update active pool based on current training step. Returns phase name."""
        self._current_step = step
        if step < self.phase2_step:
            tier = 'easy'
        elif step < self.phase3_step:
            tier = 'medium'
        else:
            tier = 'full'
        pool = self._tiers[tier]
        self._active_indices = pool if pool else self._tiers['full']
        return tier

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, index: int) -> dict:
        real_idx = self._active_indices[index % len(self._active_indices)]
        return super().__getitem__(real_idx)


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
