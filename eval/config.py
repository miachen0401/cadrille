"""EvalConfig — unified config for all Cadrille evaluations.

YAML schema
-----------
tag: "my-eval"

checkpoints:
  - path: checkpoints/cadrille-sft
    label: "SFT"

datasets:
  deepcad:
    path: data/deepcad_test_mesh
    n_samples: 500          # null = full test set
  fusion360:
    path: data/fusion360_test_mesh
    n_samples: 200

modalities: [img, pc]
max_new_tokens: 768
base_model: Qwen/Qwen2-VL-2B-Instruct

# ── resource tuning ─────────────────────────────────────────────────────────
# img and pc have different GPU memory profiles; tune separately.
# 4080 SUPER (16 GB): batch_size_img=4, batch_size_pc=16, score_workers=4
# H100 (80 GB):       batch_size_img=32, batch_size_pc=64, score_workers=8
resources:
  batch_size_img: 4           # GPU batch size for image-mode inference
  batch_size_pc: 16           # GPU batch size for point-cloud-mode inference
  score_workers: 4            # subprocess workers for CadQuery scoring
  prep_threads_img: 2         # threads for img prep (PNG load + transform)
  prep_threads_pc: 4          # threads for pc prep (mesh load + sample)
  queue_size: 32              # bounded prep→GPU queue depth
  save_code: true             # write {case_id}.py for each case
  save_stl: true              # write {case_id}.stl for successful cases

pass_k:
  enabled: false
  k: [1, 5, 10]
  n_samples: 16               # samples drawn per case
  temperature: 0.8
  iou_threshold: 0.95

render:
  enabled: true
  n: 20
  strategy: failures          # failures | low_iou | random | all

out_dir: eval_outputs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CheckpointConfig:
    path: str
    label: str

    def resolved_path(self) -> Path:
        return Path(self.path)


@dataclass
class DatasetConfig:
    path: str
    n_samples: Optional[int] = None

    def resolved_path(self) -> Path:
        return Path(self.path)


@dataclass
class ResourceConfig:
    batch_size_img: int = 4
    batch_size_pc: int = 16
    score_workers: int = 4
    prep_threads_img: int = 2
    prep_threads_pc: int = 4
    queue_size: int = 32
    save_code: bool = True
    save_stl: bool = True


@dataclass
class PassKConfig:
    enabled: bool = False
    k: list[int] = field(default_factory=lambda: [1, 5])
    n_samples: int = 8
    temperature: float = 0.8
    iou_threshold: float = 0.95


@dataclass
class RenderConfig:
    enabled: bool = True
    n: int = 20
    strategy: str = 'failures'


@dataclass
class EvalConfig:
    tag: str
    checkpoints: list[CheckpointConfig]
    datasets: dict[str, DatasetConfig]
    modalities: list[str] = field(default_factory=lambda: ['img', 'pc'])
    max_new_tokens: int = 768
    base_model: str = 'Qwen/Qwen2-VL-2B-Instruct'
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    pass_k: PassKConfig = field(default_factory=PassKConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    out_dir: str = 'eval_outputs'

    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.tag

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'EvalConfig':
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> 'EvalConfig':
        checkpoints = [
            CheckpointConfig(**{**c}) for c in d.get('checkpoints', [])
        ]
        datasets = {
            name: DatasetConfig(path=v['path'], n_samples=v.get('n_samples'))
            for name, v in d.get('datasets', {}).items()
        }

        r = d.get('resources', {})
        resources = ResourceConfig(
            batch_size_img=r.get('batch_size_img', 4),
            batch_size_pc=r.get('batch_size_pc', 16),
            score_workers=r.get('score_workers', 4),
            prep_threads_img=r.get('prep_threads_img', 2),
            prep_threads_pc=r.get('prep_threads_pc', 4),
            queue_size=r.get('queue_size', 32),
            save_code=r.get('save_code', True),
            save_stl=r.get('save_stl', True),
        )

        pk = d.get('pass_k', {})
        pass_k = PassKConfig(
            enabled=pk.get('enabled', False),
            k=pk.get('k', [1, 5]),
            n_samples=pk.get('n_samples', 8),
            temperature=pk.get('temperature', 0.8),
            iou_threshold=pk.get('iou_threshold', 0.95),
        )

        rnd = d.get('render', {})
        render = RenderConfig(
            enabled=rnd.get('enabled', True),
            n=rnd.get('n', 20),
            strategy=rnd.get('strategy', 'failures'),
        )

        return cls(
            tag=d['tag'],
            checkpoints=checkpoints,
            datasets=datasets,
            modalities=d.get('modalities', ['img', 'pc']),
            max_new_tokens=d.get('max_new_tokens', 768),
            base_model=d.get('base_model', 'Qwen/Qwen2-VL-2B-Instruct'),
            resources=resources,
            pass_k=pass_k,
            render=render,
            out_dir=d.get('out_dir', 'eval_outputs'),
        )

    def to_yaml(self) -> str:
        import dataclasses
        return yaml.dump(dataclasses.asdict(self), default_flow_style=False, allow_unicode=False)
