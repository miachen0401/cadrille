import os
import json
import math
import pickle
from pathlib import Path
import yaml

# transformers 5.6+ requires torch>=2.6 for torch.load() of optimizer.pt during
# resume_from_checkpoint. We're on torch 2.5.1; bypass the check since we
# generate + store optimizer state locally (no untrusted source).
# Need to patch BOTH source module and trainer.py's local rebinding.
try:
    import transformers.utils.import_utils as _tf_iu
    _tf_iu.check_torch_load_is_safe = lambda: None
    import transformers.trainer as _tf_tr
    _tf_tr.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# torch 2.5 + transformers 5.x: rng_state.pth contains numpy globals that
# weights_only=True rejects. Allowlist them since we own the file.
try:
    import torch as _torch
    import numpy as _np
    _safe_globals = [
        _np._core.multiarray._reconstruct,
        _np.ndarray,
        _np.dtype,
    ]
    # Newer numpy keeps dtype subclasses under _np.dtypes (UInt32DType, etc.)
    if hasattr(_np, 'dtypes'):
        for _name in dir(_np.dtypes):
            _attr = getattr(_np.dtypes, _name, None)
            if isinstance(_attr, type):
                _safe_globals.append(_attr)
    _torch.serialization.add_safe_globals(_safe_globals)
except Exception:
    pass
from datetime import datetime
from functools import partial
from argparse import ArgumentParser

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from common.model import Cadrille, collate, get_cadrille_class
from common.datasets import Text2CADDataset, CadRecodeDataset, BenchCadDataset, CadRecode20kDataset


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
    """Format learning rate compactly: 2e-4 → '2e-4', 1.5e-4 → '1.5e-4'."""
    if lr == 0:
        return 'lr0'
    exp = int(math.floor(math.log10(abs(lr))))
    mantissa = lr / (10 ** exp)
    m_str = f'{mantissa:.2g}'   # '2', '1.5', etc.
    return f'lr{m_str}e{exp}'


def _fmt_steps(n: int) -> str:
    """Format step count: 600 → '600', 10000 → '10k', 120000 → '120k'."""
    if n >= 1000 and n % 1000 == 0:
        return f'{n // 1000}k'
    return str(n)


def _auto_run_name(prefix, max_steps, learning_rate, batch_size, accum_steps, mode) -> str:
    """Generate a descriptive run name from key hyperparameters.

    Format: {prefix}-s{steps}-{lr}-b{bs}a{acc}-{mode}-{MMDD}-{HHMM}
    Example: sft-s600-lr2e-4-b2a2-pc_img-0228-1045
    """
    ts = datetime.now().strftime('%m%d-%H%M')
    lr_str = _fmt_lr(learning_rate)
    steps_str = _fmt_steps(max_steps)
    return f'{prefix}-s{steps_str}-{lr_str}-b{batch_size}a{accum_steps}-{mode}-{ts}'


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        # transformers 5.x lets `logging_dir` default to None when not passed.
        # Don't crash — only mkdir when an explicit path was set.
        if state.is_world_process_zero and args.logging_dir:
            os.makedirs(args.logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


def _build_callbacks(processor, seed, hf_upload_repo, hf_upload_private,
                     online_eval_n_per=20, mix_weights=None,
                     max_iou_k=8, max_iou_temperature=1.0,
                     max_iou_every_n_evals=1):
    from train.sft.online_eval import OnlineIoUEvalCallback
    cbs = [
        PrintToFileCallback(),
        WandbRunSaverCallback(),
        OnlineIoUEvalCallback(
            processor, n_per_dataset=online_eval_n_per, seed=seed,
            mix_weights=mix_weights,
            max_iou_k=max_iou_k,
            max_iou_temperature=max_iou_temperature,
            max_iou_every_n_evals=max_iou_every_n_evals,
        ),
    ]
    if hf_upload_repo:
        from train.sft.hf_uploader import HFCheckpointUploadCallback
        cbs.append(HFCheckpointUploadCallback(
            repo_id=hf_upload_repo, private=hf_upload_private))
        print(f'[callbacks] HF ckpt upload to {hf_upload_repo} '
              f'(private={hf_upload_private})', flush=True)
    return cbs


class WandbRunSaverCallback(TrainerCallback):
    """Write W&B run URL to output_dir/wandb_run.txt at training start."""

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            try:
                import wandb
                if wandb.run:
                    with open(os.path.join(args.output_dir, 'wandb_run.txt'), 'w') as f:
                        f.write(f"run_id: {wandb.run.id}\n")
                        f.write(f"run_url: {wandb.run.url}\n")
                        f.write(f"project: {wandb.run.project}\n")
                        f.write(f"name: {wandb.run.name}\n")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Weighted-sampler trainer (enforces sft_mix_weights across ConcatDataset)
# ---------------------------------------------------------------------------

class LengthGroupedWeightedSampler(torch.utils.data.Sampler):
    """WeightedRandomSampler + mega-batch length sort.

    Draw indices per `sample_weights` (with replacement), then within every
    `batch_size * mega_batch_mult` chunk sort by length. Similar spirit to
    HF's `LengthGroupedSampler` but wraps our mix sampler so both coexist.

    Mega-batches mean items of similar length land in the same mini-batches
    → less padding waste. Text2CAD's long descriptions vs. BenchCAD's short
    metadata otherwise share batches and inflate seq_len for the whole batch.
    """

    def __init__(self, weights, num_samples: int, lengths: list,
                 batch_size: int, mega_batch_mult: int = 100):
        self.weights = torch.as_tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.lengths = lengths
        self.batch_size = max(1, batch_size)
        self.mega_batch_size = self.batch_size * mega_batch_mult

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        indices = torch.multinomial(self.weights, self.num_samples,
                                    replacement=True).tolist()
        # Shuffle already randomised; sort inside each mega-batch.
        out = []
        for i in range(0, len(indices), self.mega_batch_size):
            chunk = indices[i:i + self.mega_batch_size]
            chunk.sort(key=lambda idx: self.lengths[idx], reverse=True)
            out.extend(chunk)
        return iter(out)


class _StepTracker:
    """Mutable shared step counter so callback + sampler agree on global_step."""
    def __init__(self, step: int = 0):
        self.step = step


class CurriculumWeightedSampler(torch.utils.data.Sampler):
    """Curriculum-weighted sampler — switch sampling weights at phase boundaries.

    `phases` is a list of `(start_step, weights_tensor)` tuples sorted by
    start_step (must include start_step=0). The active weight set at any
    given step S is the phase whose start_step is the largest value <= S.

    Weights are re-evaluated on every `__iter__` call (i.e. every dataloader
    epoch); HF Trainer creates the dataloader once but re-iterates it per
    epoch, so phase boundaries take effect at the next epoch boundary
    after the threshold step. With our typical 113 k-item ConcatDataset
    and effective batch 32 (~3.5 k steps/epoch on a 20 k run), expect a
    ~1 k-step lag between phase trigger and observed weight switch.

    Same length-grouping behaviour as LengthGroupedWeightedSampler.
    """

    def __init__(self, phases, num_samples: int, lengths: list,
                 batch_size: int, mega_batch_mult: int = 100,
                 step_tracker: _StepTracker | None = None):
        if not phases:
            raise ValueError('curriculum requires at least one phase')
        sorted_phases = sorted(phases, key=lambda p: p[0])
        if sorted_phases[0][0] != 0:
            raise ValueError(
                f'first phase must start at step 0 (got {sorted_phases[0][0]})'
            )
        self.phases = sorted_phases
        self.num_samples = num_samples
        self.lengths = lengths
        self.batch_size = max(1, batch_size)
        self.mega_batch_size = self.batch_size * mega_batch_mult
        self.step_tracker = step_tracker or _StepTracker()
        self._last_active_start = -1   # debounce phase-change print

    def _active_weights(self):
        active = self.phases[0][1]
        active_start = 0
        for start, w in self.phases:
            if self.step_tracker.step >= start:
                active = w
                active_start = start
        return active, active_start

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        weights, active_start = self._active_weights()
        weights_t = torch.as_tensor(weights, dtype=torch.float)
        if active_start != self._last_active_start:
            print(f'[curriculum] step={self.step_tracker.step} → '
                  f'switched to phase that started at step {active_start} '
                  f'({len(self.phases)} phases total)', flush=True)
            self._last_active_start = active_start
        indices = torch.multinomial(weights_t, self.num_samples,
                                    replacement=True).tolist()
        out = []
        for i in range(0, len(indices), self.mega_batch_size):
            chunk = indices[i:i + self.mega_batch_size]
            chunk.sort(key=lambda idx: self.lengths[idx], reverse=True)
            out.extend(chunk)
        return iter(out)


class CurriculumStepCallback(TrainerCallback):
    """Push state.global_step into the sampler's StepTracker on every step.

    The sampler reads this on each new __iter__ (epoch boundary) to decide
    which phase's weights to use. Lightweight — no I/O, just an int copy.
    """
    def __init__(self, step_tracker: _StepTracker):
        self.step_tracker = step_tracker

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_tracker.step = int(state.global_step)


def _expand_mix_to_sample_weights(mix: dict, sources: dict) -> list[float]:
    """Convert a {source: weight} mix dict into a flat per-sample weight list,
    aligned with the order of `sources` (a dict preserving insertion order).

    Each source contributes `[mix[src] / len(ds)] * len(ds)` so the total
    sampling mass per source equals its configured mix weight.
    """
    out = []
    for src, ds in sources.items():
        n = len(ds)
        w = float(mix.get(src, 0.0))
        if n > 0 and w > 0:
            out.extend([w / n] * n)
        else:
            out.extend([0.0] * n)
    return out


def _collect_lengths(dataset) -> list:
    """Concat per-dataset .lengths lists; fall back to uniform if a source has none."""
    if isinstance(dataset, ConcatDataset):
        out = []
        for d in dataset.datasets:
            out.extend(getattr(d, 'lengths', [1] * len(d)))
        return out
    return getattr(dataset, 'lengths', [1] * len(dataset))


class WeightedSamplerTrainer(Trainer):
    """HF Trainer with optional WeightedRandomSampler + length-grouped batches.

    When `sample_weights` is provided, uses it instead of the default
    RandomSampler, so sft_mix_weights (e.g. recode:text2cad:benchcad = 2:1:2)
    actually shapes batch composition. When `group_by_length=True` AND lengths
    are available, also sorts within mega-batches to cut padding waste on
    mixed-length corpora.
    """

    def __init__(self, *args, sample_weights=None, group_by_length=False,
                 curriculum_phases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
        # `group_by_length` was a TrainingArguments kwarg in transformers <5.0
        # but was removed in 5.x. We carry our own copy to drive
        # LengthGroupedWeightedSampler regardless of transformers version.
        self._group_by_length = group_by_length
        # curriculum_phases is a list of (start_step, sample_weights_list).
        # When set, takes priority over sample_weights — the sampler picks
        # active weights from the current step.
        self.curriculum_phases = curriculum_phases
        if curriculum_phases:
            self._curriculum_step_tracker = _StepTracker(0)
            self.add_callback(CurriculumStepCallback(self._curriculum_step_tracker))
            print(f'[curriculum] {len(curriculum_phases)} phases registered: '
                  + ', '.join(f'step={s}' for s, _ in curriculum_phases),
                  flush=True)

    def _get_eval_sampler(self, eval_dataset):
        # HF Trainer tries to infer lengths from 'input_ids' when
        # group_by_length=True, which fails for our custom datasets. Always
        # use a plain sequential sampler for eval — group_by_length is only
        # a training-throughput trick.
        return torch.utils.data.SequentialSampler(eval_dataset)

    def _get_train_sampler(self, *args, **kwargs):
        # Curriculum path takes priority. Phase weights already expanded to
        # per-sample form by run().
        if self.curriculum_phases:
            n = len(self.train_dataset)
            lengths = _collect_lengths(self.train_dataset)
            if len(lengths) != n:
                lengths = [1] * n
            print(f'[sampler] CurriculumWeightedSampler: n={n}, '
                  f'batch={self.args.train_batch_size}, '
                  f'phases={[s for s, _ in self.curriculum_phases]}',
                  flush=True)
            return CurriculumWeightedSampler(
                phases=self.curriculum_phases,
                num_samples=n,
                lengths=lengths,
                batch_size=self.args.train_batch_size,
                step_tracker=self._curriculum_step_tracker,
            )
        if self.sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        if self.args.world_size > 1:
            import warnings
            warnings.warn(
                'sft_mix_weights: using WeightedRandomSampler per rank '
                '(no DistributedSampler wrapper). Mix ratios hold in '
                'expectation but rank shards are not disjoint.'
            )
        n = len(self.train_dataset)
        if self._group_by_length:
            lengths = _collect_lengths(self.train_dataset)
            if len(lengths) == n and len(set(lengths)) > 1:
                print(f'[sampler] LengthGroupedWeightedSampler: '
                      f'n={n}, batch={self.args.train_batch_size}, '
                      f'length range [{min(lengths)}, {max(lengths)}]',
                      flush=True)
                return LengthGroupedWeightedSampler(
                    weights=self.sample_weights, num_samples=n,
                    lengths=lengths,
                    batch_size=self.args.train_batch_size,
                )
            print('[sampler] group_by_length set but lengths unavailable/uniform; '
                  'falling back to plain WeightedRandomSampler', flush=True)
        return WeightedRandomSampler(
            weights=self.sample_weights, num_samples=n, replacement=True,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(data_path, output_dir, mode, use_text, max_steps, batch_size_override,
        accum_steps_override, learning_rate, warmup_steps, lr_scheduler_type,
        dataloader_workers, log_steps,
        save_steps, eval_steps, wandb_project, eval_on_start,
        bf16, tf32, gradient_checkpointing, optim,
        seed=42, max_code_len=None, sft_mix_weights=None,
        base_model='Qwen/Qwen2-VL-2B-Instruct',
        resume_from_checkpoint=None, cfg_to_save=None,
        hf_upload_repo=None, hf_upload_private=True,
        group_by_length=False,
        save_only_model=True, save_total_limit=1,
        online_eval_n_per=20,
        backbone='qwen2_vl',
        max_iou_k=8, max_iou_temperature=1.0, max_iou_every_n_evals=1,
        curriculum_phases=None,
        benchcad_train_pkl=None, cad_iso_106_train_pkl=None,
        benchcad_simple_train_pkl=None,
        holdout_families=None, holdout_families_v2=None,
        total_train_dp=None, sft_pool_rows=None):

    os.makedirs(output_dir, exist_ok=True)

    # Resolve "latest" checkpoint shortcut
    # Resume + save_only_model are incompatible — warn on mismatch.
    if resume_from_checkpoint and save_only_model:
        print('[WARN] resume_from_checkpoint is set but save_only_model=True. '
              'Optimizer/LR state will not be restored — the resume will load '
              'model weights only. Set save_only_model=false in your config '
              'if you want exact-state resume (costs ~13 GB / checkpoint).',
              flush=True)

    if resume_from_checkpoint == 'latest':
        ckpt_dirs = sorted(
            [d for d in os.listdir(output_dir)
             if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))],
            key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
        if ckpt_dirs:
            resume_from_checkpoint = os.path.join(output_dir, ckpt_dirs[-1])
            print(f'Resuming from checkpoint: {resume_from_checkpoint}')
        else:
            print('No checkpoint found for --resume-from-checkpoint latest; starting from scratch.')
            resume_from_checkpoint = None

    # Save resolved config snapshot for traceability (skip if already exists)
    if cfg_to_save:
        cfg_path = os.path.join(output_dir, 'run_config.yaml')
        if not os.path.exists(cfg_path):
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_to_save, f, default_flow_style=False, sort_keys=True)
            print(f'Config snapshot saved → {cfg_path}')

    cad_recode_path = os.path.join(data_path, 'cad-recode-v1.5')
    # Source-labeled datasets; keeps per-source lengths so WeightedRandomSampler
    # can convert sft_mix_weights (source-level ratios) into per-sample weights.
    sources = {}
    if os.path.isdir(cad_recode_path) and os.path.exists(os.path.join(cad_recode_path, 'train.pkl')):
        # Schema-detect: original CAD-Recode bundle has 'mesh_path' (STL); our
        # img-only render of filapro/cad-recode-v1.5 has 'png_path'. Route to
        # the matching loader so both bundle formats can live at the same path
        # name. Img-only corpus → CadRecode20kDataset; PC-capable corpus with
        # STLs → CadRecodeDataset.
        with open(os.path.join(cad_recode_path, 'train.pkl'), 'rb') as _f:
            _probe = pickle.load(_f)
        _img_only = bool(_probe) and 'mesh_path' not in _probe[0] and 'png_path' in _probe[0]
        if _img_only and mode != 'pc':
            sources['recode'] = CadRecode20kDataset(
                root_dir=cad_recode_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img')
        elif not _img_only:
            sources['recode'] = CadRecodeDataset(
                root_dir=cad_recode_path,
                split='train',
                n_points=256,
                normalize_std_pc=100,
                noise_scale_pc=0.01,
                img_size=268,
                normalize_std_img=200,
                noise_scale_img=-1,
                num_imgs=4,
                mode=mode,
                max_code_len=max_code_len)
        # else (img-only corpus + pc mode) → silently skip, no STLs available
    batch_size = batch_size_override or 28
    accumulation_steps = accum_steps_override or 1

    if use_text:
        # Legacy text2cad — only load if local data is still present (deleted
        # in v3 prep on 2026-04-28). Without this guard, train.py crashes when
        # the legacy folder has been removed but use_text is still True for
        # text2cad_bench_text below.
        text2cad_legacy_path = os.path.join(data_path, 'text2cad')
        if (os.path.isdir(text2cad_legacy_path)
                and os.path.exists(os.path.join(text2cad_legacy_path, 'train.pkl'))):
            sources['text2cad'] = Text2CADDataset(
                root_dir=text2cad_legacy_path,
                split='train',
                max_code_len=max_code_len)
        batch_size = batch_size_override or 8
        accumulation_steps = accum_steps_override or 4

    benchcad_path = os.path.join(data_path, 'benchcad')
    benchcad_pkl = benchcad_train_pkl or 'train.pkl'
    if os.path.isdir(benchcad_path) and os.path.exists(os.path.join(benchcad_path, benchcad_pkl)):
        sources['benchcad'] = BenchCadDataset(
            root_dir=benchcad_path,
            split='train',
            n_points=256,
            normalize_std_pc=100,
            noise_scale_pc=0.01,
            img_size=268,
            normalize_std_img=200,
            noise_scale_img=-1,
            num_imgs=4,
            mode=mode,
            max_code_len=max_code_len,
            pkl_filename=benchcad_pkl)

    recode20k_path = os.path.join(data_path, 'cad-recode-20k')
    if os.path.isdir(recode20k_path) and os.path.exists(os.path.join(recode20k_path, 'train.pkl')):
        # Img-only corpus; silently skip if mode is 'pc' (loader will raise).
        if mode != 'pc':
            sources['recode20k'] = CadRecode20kDataset(
                root_dir=recode20k_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img')

    # Bench-style variants (T8): same dataset class, different root dirs.
    # Code in these dirs has been v2-rewritten to BenchCAD shell style;
    # PNG geometry is identical to the raw recode (rewrite is AST-only).
    recode_bench_path = os.path.join(data_path, 'cad-recode-bench')
    if os.path.isdir(recode_bench_path) and os.path.exists(os.path.join(recode_bench_path, 'train.pkl')):
        if mode != 'pc':
            sources['recode_bench'] = CadRecode20kDataset(
                root_dir=recode_bench_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img')

    # Phase F (T8 follow-up): BenchCAD/cad_simple_ops_100k imported via
    # data_prep/import_benchcad_simple.py. Native BenchCAD shell style (no
    # rewrite needed); image+code, no description. Same on-disk layout as
    # recode_bench so we reuse CadRecode20kDataset.
    benchcad_simple_path = os.path.join(data_path, 'benchcad-simple')
    benchcad_simple_pkl = benchcad_simple_train_pkl or 'train.pkl'
    if os.path.isdir(benchcad_simple_path) and os.path.exists(os.path.join(benchcad_simple_path, benchcad_simple_pkl)):
        if mode != 'pc':
            sources['benchcad_simple'] = CadRecode20kDataset(
                root_dir=benchcad_simple_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img',
                pkl_filename=benchcad_simple_pkl)

    # Phase F (T8 follow-up): BenchCAD/cad_iso_106 imported via the same
    # data_prep/import_benchcad_simple.py pipeline (different upstream
    # generator config — pipe_elbow / pipe_flange / ball_knob / industrial
    # parts). 170k items, BenchCAD shell style, only dataset that has
    # fillet (~19% of items) — primary source for rare-op coverage.
    cad_iso_106_path = os.path.join(data_path, 'cad-iso-106')
    cad_iso_106_pkl = cad_iso_106_train_pkl or 'train.pkl'
    if os.path.isdir(cad_iso_106_path) and os.path.exists(os.path.join(cad_iso_106_path, cad_iso_106_pkl)):
        if mode != 'pc':
            sources['cad_iso_106'] = CadRecode20kDataset(
                root_dir=cad_iso_106_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img',
                pkl_filename=cad_iso_106_pkl)

    # v4 source: BenchCAD/benchcad-easy (~109k items, 55 shards). Same
    # `simple_*` family taxonomy as benchcad/benchcad-simple but ~10× the
    # size of benchcad. Re-rendered this session via cadquery → 4-view
    # 268×268 PNG (looser tessellation tolerance for thumbnails).
    benchcad_easy_path = os.path.join(data_path, 'benchcad-easy')
    if os.path.isdir(benchcad_easy_path) and os.path.exists(os.path.join(benchcad_easy_path, 'train.pkl')):
        if mode != 'pc':
            sources['benchcad_easy'] = CadRecode20kDataset(
                root_dir=benchcad_easy_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img')

    # text2cad-bench available in TWO modes (separate sources, separate weights):
    #   * text2cad_bench_img: image-conditioned (uses png_path, code) via CadRecode20kDataset
    #   * text2cad_bench_text: text-conditioned (uses description, code) via Text2CADDataset
    # Same train.pkl (each row has uid, code, description, png_path), but different
    # __getitem__ produces different conditioning input. User policy 2026-04-28:
    # do NOT mix img+text on the same sample — pick exactly one per training example.
    text2cad_bench_path = os.path.join(data_path, 'text2cad-bench')
    if os.path.isdir(text2cad_bench_path) and os.path.exists(os.path.join(text2cad_bench_path, 'train.pkl')):
        if mode != 'pc':
            sources['text2cad_bench_img'] = CadRecode20kDataset(
                root_dir=text2cad_bench_path,
                split='train',
                img_size=268,
                max_code_len=max_code_len,
                mode='img')
        if use_text:
            sources['text2cad_bench_text'] = Text2CADDataset(
                root_dir=text2cad_bench_path,
                split='train',
                max_code_len=max_code_len)

    # Optional: cap the unique-row pool to a fixed total dp count,
    # subsampling each source proportionally to its mix weight. Default off
    # (None / 0) → use full source sizes.
    #
    # Use case: enforce identical "data volume" across configs that mix
    # different source compositions (§7 v2 5-line ablation), so the only
    # confound is content not pool-size diversity.
    #
    # Pool-size control — TWO modes (sft_pool_rows takes priority):
    #
    # 1. Explicit (`sft_pool_rows: {src: N, …}`):
    #    train.py truncates each source to exactly the listed count. Use
    #    this when the yaml has been "frozen" with computed counts via
    #    data_prep/compute_v2_pool_rows.py — gives perfectly reproducible
    #    per-source row choices across machines.
    #
    # 2. Implicit (`total_train_dp: N`, no sft_pool_rows):
    #    Saturate-and-redistribute: per-source target = weight × N /
    #    total_weight, capped at availability; deficit redistributed to
    #    non-saturated sources. Iterative.
    #
    # Both modes share the same per-source rng seed
    # (sha256(f'{base_seed}:{src_name}')[:4]) so smaller target = prefix
    # of larger for the same source.
    if sft_pool_rows:
        # Mode 1 — explicit per-source counts (frozen)
        if total_train_dp:
            print('[sft_pool_rows] both sft_pool_rows AND total_train_dp set; '
                  'sft_pool_rows takes priority (set total_train_dp: null to silence).',
                  flush=True)
        import random as _rnd
        import hashlib as _hl
        base_seed = int(seed) if isinstance(seed, int) else 42
        actual_total = sum(sft_pool_rows.values())
        print(f'[sft_pool_rows] explicit per-source pool: '
              f'{actual_total:,} rows across {len(sft_pool_rows)} sources',
              flush=True)
        for src_name, ds in sources.items():
            target = int(sft_pool_rows.get(src_name, 0))
            if target <= 0:
                continue
            if not hasattr(ds, 'annotations'):
                print(f'  {src_name:24s}  (no .annotations attr — skip)')
                continue
            cur = len(ds.annotations)
            if target >= cur:
                print(f'  {src_name:24s}  {cur:,} kept  (target {target:,} ≥ available)')
                continue
            _digest = _hl.sha256(f'{base_seed}:{src_name}'.encode()).digest()
            src_seed = int.from_bytes(_digest[:4], 'big')
            _rng = _rnd.Random(src_seed)
            idx = list(range(cur))
            _rng.shuffle(idx)
            idx = sorted(idx[:target])
            ds.annotations = [ds.annotations[i] for i in idx]
            if hasattr(ds, 'lengths') and ds.lengths is not None and \
                    len(ds.lengths) == cur:
                ds.lengths = [ds.lengths[i] for i in idx]
            print(f'  {src_name:24s}  {cur:,} → {target:,}  (src_seed={src_seed})')
    elif total_train_dp:
        # Removed: saturate-redistribute path. The rounding made targets
        # off by 1-2 from total_train_dp (CodeRabbit flag), and we've
        # converged on natural-pool sampling (Plan A) for §7 v2. If you
        # want a precise pool cap, use explicit `sft_pool_rows:` —
        # generated once via data_prep/compute_v2_pool_rows.py and
        # committed for provenance.
        raise ValueError(
            'total_train_dp is set without sft_pool_rows. The saturate-'
            'redistribute path was removed (was inexact and dead code in '
            'practice). Either drop total_train_dp from your config '
            '(natural pool, Plan A) or generate explicit sft_pool_rows '
            'via data_prep/compute_v2_pool_rows.py.'
        )

    # Defensive filter — remove any row whose uid appears in the eval set.
    # The 90/10 split at data prep already guarantees disjointness, but we
    # apply this filter unconditionally so that no eval uid CAN possibly
    # leak into training, even if a future data-prep script regresses or a
    # custom pkl is dropped in. Reads data/_eval_uids/v2_eval_uids.json
    # (regenerate via `uv run python scripts/dump_eval_uids.py`).
    #
    # FAIL-CLOSED: any error here aborts the run. This is the last line of
    # defense for paper provenance — silently continuing without filtering
    # could leak eval uids into training. If you intentionally want to skip
    # the filter, delete data/_eval_uids/v2_eval_uids.json (the `exists()`
    # gate then short-circuits).
    eval_uids_path = Path('data/_eval_uids/v2_eval_uids.json')
    if eval_uids_path.exists():
        _eval_uids: set[str] = set()
        for _bucket_uids in json.loads(eval_uids_path.read_text()).values():
            _eval_uids.update(_bucket_uids)
        for src_name, ds in sources.items():
            if not hasattr(ds, 'annotations'):
                continue
            cur = len(ds.annotations)
            # Index-aligned filter so ds.lengths stays positionally
            # consistent with ds.annotations even when dropped rows aren't
            # at the end (e.g. group_by_length sampler reads ds.lengths[i]
            # to bucket batches).
            keep_idx = [i for i, r in enumerate(ds.annotations)
                        if r.get('uid') not in _eval_uids]
            drop = cur - len(keep_idx)
            if drop > 0:
                ds.annotations = [ds.annotations[i] for i in keep_idx]
                if hasattr(ds, 'lengths') and ds.lengths is not None \
                        and len(ds.lengths) == cur:
                    ds.lengths = [ds.lengths[i] for i in keep_idx]
                print(f'[eval-leak-filter] {src_name}: dropped {drop} eval-overlap rows '
                      f'({cur} → {len(keep_idx)})', flush=True)
        print(f'[eval-leak-filter] {len(_eval_uids):,} eval uids checked across '
              f'{len(sources)} sources', flush=True)

    train_dataset = ConcatDataset(list(sources.values())) if len(sources) > 1 \
        else next(iter(sources.values()))

    # Curriculum: expand each phase's `sft_mix_weights` to per-sample weights.
    # Schema: list of {at_step: int, sft_mix_weights: {source: weight}}.
    # If set, takes priority over the static sft_mix_weights below — the
    # sampler picks active weights from the current global_step at each
    # epoch boundary.
    expanded_phases = None
    if curriculum_phases:
        expanded_phases = []
        for phase in curriculum_phases:
            mix = phase.get('sft_mix_weights', {})
            w = _expand_mix_to_sample_weights(mix, sources)
            if sum(w) <= 0:
                raise ValueError(
                    f'curriculum phase at step {phase.get("at_step")} has '
                    f'no positive weights — would freeze sampling.'
                )
            expanded_phases.append((int(phase['at_step']), w))
        print(f'[curriculum_phases] expanded {len(expanded_phases)} phases'
              + ' '.join(f'\n  step={s}: {phase["sft_mix_weights"]}'
                         for s, phase in zip(
                             [p[0] for p in expanded_phases],
                             curriculum_phases)),
              flush=True)

    # Convert sft_mix_weights (source-level ratios) → per-sample weights.
    # weight_per_sample = mix[source] / len(source_dataset), so the total
    # probability mass of each source equals its configured mix weight.
    sample_weights = None
    if sft_mix_weights and not curriculum_phases:
        missing = [s for s in sft_mix_weights if s not in sources]
        if missing:
            print(f'[sft_mix_weights] WARNING: {missing} not loaded; '
                  f'effective mix uses only {list(sources.keys())}')
        sample_weights = []
        active = {}
        for src, ds in sources.items():
            n = len(ds)
            w = float(sft_mix_weights.get(src, 0.0))
            if n == 0:
                if w > 0:
                    print(f'[sft_mix_weights] WARNING: source {src!r} is '
                          f'empty (after filter / no train split); dropping '
                          f'from mix even though weight={w}')
                active[src] = 0.0
                continue
            active[src] = w
            if w > 0:
                sample_weights.extend([w / n] * n)
            else:
                sample_weights.extend([0.0] * n)
        if sum(sample_weights) <= 0:
            print('[sft_mix_weights] WARNING: all active weights are 0; '
                  'falling back to uniform sampler')
            sample_weights = None
        else:
            print(f'[sft_mix_weights] enforced via WeightedRandomSampler: {active}')

    # val set selection: always prefer benchcad val when its pkl exists — it's
    # the metric we report regardless of training mix. Some configs (e.g. the
    # HQ-only baseline) zero the benchcad train weight but still want benchcad
    # val for eval comparability. Only fall back to cad-recode val when no
    # benchcad val.pkl is available, and only if cad-recode-v1.5 carries STL
    # meshes (img-only bundles don't, and CadRecodeDataset.get_img would
    # KeyError on `mesh_path`).
    eval_dataset = None
    benchcad_val_pkl = os.path.join(benchcad_path, 'val.pkl')
    if os.path.exists(benchcad_val_pkl):
        eval_dataset = BenchCadDataset(
            root_dir=benchcad_path,
            split='val',
            n_points=256,
            normalize_std_pc=100,
            noise_scale_pc=None,
            img_size=268,
            normalize_std_img=200,
            noise_scale_img=-1,
            num_imgs=4,
            mode=mode,
            max_code_len=max_code_len)
        print(f'[eval] using benchcad val ({len(eval_dataset)} samples)')
    else:
        val_pkl = os.path.join(cad_recode_path, 'val.pkl')
        if os.path.exists(val_pkl):
            with open(val_pkl, 'rb') as _f:
                _val_probe = pickle.load(_f)
            _val_has_stl = bool(_val_probe) and 'mesh_path' in _val_probe[0]
            if _val_has_stl:
                eval_dataset = CadRecodeDataset(
                    root_dir=cad_recode_path,
                    split='val',
                    n_points=256,
                    normalize_std_pc=100,
                    noise_scale_pc=None,
                    img_size=268,
                    normalize_std_img=200,
                    noise_scale_img=-1,
                    num_imgs=4,
                    mode=mode,
                    max_code_len=max_code_len)
                print(f'[eval] using cad-recode val ({len(eval_dataset)} samples)')
            else:
                print(f'[eval] cad-recode val.pkl is img-only (no mesh_path); '
                      'no eval dataset selected')
    has_val = eval_dataset is not None

    processor = AutoProcessor.from_pretrained(
        base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')
    cadrille_cls = get_cadrille_class(backbone)
    print(f'[model] backbone={backbone!r} → {cadrille_cls.__name__}', flush=True)
    # Try flash_attention_2 first (3-5× speedup), fall back to sdpa if flash-attn
    # isn't installed (it lives outside pyproject because it needs --no-build-
    # isolation; setup.sh installs it). sdpa is stock PyTorch, always available.
    try:
        model = cadrille_cls.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2')
    except (ImportError, ValueError, RuntimeError) as e:
        if 'flash' in str(e).lower():
            print(f'[model] flash_attention_2 unavailable ({type(e).__name__}: '
                  f'{str(e)[:80]}…); falling back to sdpa.', flush=True)
            model = cadrille_cls.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                attn_implementation='sdpa')
        else:
            raise

    report_to = 'wandb' if wandb_project else 'none'
    if wandb_project:
        os.environ.setdefault('WANDB_PROJECT', wandb_project)
        # Disable stdout/stderr streaming — prevents W&B rate-limit errors on long runs.
        os.environ.setdefault('WANDB_CONSOLE', 'off')
        # When resuming a checkpoint, re-attach to the existing W&B run so the
        # loss/eval curves are continuous rather than starting a fresh run.
        if resume_from_checkpoint:
            wandb_run_file = os.path.join(output_dir, 'wandb_run.txt')
            if os.path.exists(wandb_run_file):
                with open(wandb_run_file) as f:
                    for line in f:
                        if line.startswith('run_id:'):
                            existing_id = line.split(':', 1)[1].strip()
                            os.environ['WANDB_RUN_ID'] = existing_id
                            os.environ['WANDB_RESUME'] = 'must'
                            print(f'W&B: resuming existing run {existing_id}')
                            break
            else:
                os.environ.setdefault('WANDB_RESUME', 'allow')
        else:
            os.environ.setdefault('WANDB_RESUME', 'allow')

    trainer = WeightedSamplerTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            # `run_name` ends up as the wandb run name (HF integration
            # forwards it to wandb.init). Without this, wandb falls back to
            # its own random adjective-noun ("ethereal-galaxy-46"), which is
            # useless for distinguishing runs. Use the descriptive output-dir
            # basename so wandb panel shows e.g. "sft-s20k-lr2e-4-…0425-1929".
            run_name=os.path.basename(output_dir),
            per_device_train_batch_size=batch_size,
            dataloader_num_workers=dataloader_workers,
            max_steps=max_steps,
            lr_scheduler_type=lr_scheduler_type,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            gradient_accumulation_steps=accumulation_steps,
            bf16=bf16,
            tf32=tf32 if tf32 else None,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant': False} if gradient_checkpointing else None,
            optim=optim,
            remove_unused_columns=False,
            logging_first_step=True,
            logging_steps=log_steps,
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,
            save_strategy='steps',
            save_steps=save_steps,
            eval_strategy='steps' if has_val else 'no',
            eval_steps=eval_steps if has_val else None,
            eval_on_start=eval_on_start and has_val,
            load_best_model_at_end=has_val,
            seed=seed,
            data_seed=seed,
            report_to=report_to),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate, processor=processor, n_points=256),
        # transformers 4.46+ deprecated `tokenizer=`, removed in 5.x.
        # Use `processing_class=` (multimodal models pass an AutoProcessor here).
        processing_class=processor,
        sample_weights=sample_weights,
        group_by_length=group_by_length,
        curriculum_phases=expanded_phases,
        callbacks=_build_callbacks(
            processor=processor, seed=seed,
            hf_upload_repo=hf_upload_repo,
            hf_upload_private=hf_upload_private,
            online_eval_n_per=online_eval_n_per,
            mix_weights=sft_mix_weights,
            max_iou_k=max_iou_k,
            max_iou_temperature=max_iou_temperature,
            max_iou_every_n_evals=max_iou_every_n_evals,
        ))

    # Wire holdout_families into online_eval so BC val + iso val split into
    # IID/OOD buckets (separate metrics in wandb: eval/img/BenchCAD val IID/*,
    # eval/img/BenchCAD val OOD/*, eval/img/iso val IID/*, eval/img/iso val
    # OOD/*). The split is decoupled from train data filtering — even
    # configs that don't filter the train pkl (baseline/iid) still need the
    # 4-bucket eval so all §7 v2 lines plot on the same axes.
    #
    # Default: read configs/sft/holdout_families.yaml unconditionally so
    # every config in the chain shows the same 4 val buckets. cfg.holdout_families
    # (when set) overrides — kept for backward compat.
    from train.sft import online_eval as _oe
    eval_holdout = holdout_families
    if not eval_holdout:
        try:
            with open('configs/sft/holdout_families.yaml') as f:
                eval_holdout = yaml.safe_load(f).get('holdout_families', [])
        except FileNotFoundError:
            eval_holdout = []
    if eval_holdout:
        _oe.set_holdout_families(eval_holdout)
        print(f'[online-eval] BC + iso val IID/OOD split enabled, holdout_families='
              f'{sorted(eval_holdout)}', flush=True)
    if holdout_families_v2:
        _oe.set_holdout_families_v2(holdout_families_v2)
        print(f'[online-eval] bench-simple OOD bucket enabled, holdout_families_v2='
              f'{sorted(holdout_families_v2)}', flush=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Always save final checkpoint (regardless of save_steps cadence)
    final_dir = os.path.join(output_dir, 'checkpoint-final')
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f'Final checkpoint saved → {final_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description='SFT training for Cadrille. '
                    'All settings can be defined in a YAML config file; '
                    'CLI flags override the config.')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g. configs/sft_smoke.yaml)')

    # Run identity & output
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name → checkpoints saved to {checkpoints-dir}/{run-name}/. '
                             'Overrides run_name from config.')
    parser.add_argument('--checkpoints-dir', type=str, default=None,
                        help='Root directory for all checkpoints (default: ./checkpoints). '
                             'Overrides checkpoints_dir from config.')

    # Data
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None,
                        choices=['pc_img', 'img', 'pc'])
    parser.add_argument('--use-text', action='store_true', default=None)

    # Training
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Per-device batch size (default: 28 for pc_img, 8 with --use-text)')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                        help='Gradient accumulation steps (default: 1 for pc_img, 4 with --use-text). '
                             'Effective batch = batch_size × accumulation_steps. '
                             'Example: --batch-size 2 --accumulation-steps 14 on 16 GB GPU.')
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='Path to checkpoint dir, or "latest" to auto-detect the '
                             'most recent checkpoint in the run output directory.')
    parser.add_argument('--dataloader-workers', type=int, default=None)

    # Logging & checkpoints
    parser.add_argument('--log-steps', type=int, default=None)
    parser.add_argument('--save-steps', type=int, default=None)
    parser.add_argument('--eval-steps', type=int, default=None)
    parser.add_argument('--eval-on-start', action='store_true', default=None,
                        help='Run evaluation at step 0 for a pre-training baseline '
                             '(requires a val split to exist)')

    # W&B
    parser.add_argument('--wandb-project', type=str, default=None)

    args = parser.parse_args()

    # Load YAML config if provided
    cfg = _load_yaml(args.config) if args.config else {}

    # Resolve all training parameters first (CLI > YAML > hardcoded default)
    data_path         = _p(args.data_path,          cfg.get('data_path'),          './data')
    mode              = _p(args.mode,                cfg.get('mode'),               'pc_img')
    use_text          = bool(args.use_text or        cfg.get('use_text',            False))
    max_steps         = _p(args.max_steps,           cfg.get('max_steps'),          120000)
    batch_size        = _p(args.batch_size,          cfg.get('batch_size'),         None)
    accum_steps       = _p(args.accumulation_steps,  cfg.get('accumulation_steps'), None)
    learning_rate         = _p(args.learning_rate,       cfg.get('learning_rate'),         2e-4)
    warmup_steps          = _p(None,                      cfg.get('warmup_steps'),          1000)
    lr_scheduler_type     = _p(None,                      cfg.get('lr_scheduler_type'),     'cosine')
    dataloader_workers    = _p(args.dataloader_workers,  cfg.get('dataloader_workers'),     8)
    resume_from_checkpoint= _p(args.resume_from_checkpoint, cfg.get('resume_from_checkpoint'), None)
    log_steps         = _p(args.log_steps,           cfg.get('log_steps'),          1000)
    save_steps        = _p(args.save_steps,          cfg.get('save_steps'),         10000)
    eval_steps        = _p(args.eval_steps,          cfg.get('eval_steps'),         10000)
    eval_on_start         = bool(args.eval_on_start or cfg.get('eval_on_start',         False))
    wandb_project         = _p(args.wandb_project,    cfg.get('wandb_project'),         None)
    base_model            = cfg.get('base_model', 'Qwen/Qwen2-VL-2B-Instruct')
    bf16                  = bool(cfg.get('bf16',              False))
    tf32                  = bool(cfg.get('tf32',              False))
    gradient_checkpointing= bool(cfg.get('gradient_checkpointing', False))
    optim                 = cfg.get('optim', 'adamw_torch')
    seed                  = int(cfg.get('seed', 42))
    max_code_len          = cfg.get('max_code_len', None)
    sft_mix_weights       = cfg.get('sft_mix_weights', None)  # {source: weight}; enforced via WeightedSamplerTrainer
    hf_upload_repo        = cfg.get('hf_upload_repo', None)   # e.g. 'Hula0401/cadrille-<tag>'; null = disabled
    hf_upload_private     = bool(cfg.get('hf_upload_private', True))
    group_by_length       = bool(cfg.get('group_by_length', False))
    # Checkpoint storage tradeoffs:
    #   save_only_model=True  → drop optimizer/scheduler/RNG state, ~4GB/ckpt
    #   save_only_model=False → full trainer state, ~13GB/ckpt, exact resume
    # If you set resume_from_checkpoint, you almost certainly want save_only_model=False
    # so the resume restores LR schedule + AdamW moments. We warn on mismatch below.
    save_only_model       = bool(cfg.get('save_only_model', True))
    save_total_limit      = int(cfg.get('save_total_limit', 1))
    online_eval_n_per     = int(cfg.get('online_eval_n_per', 20))
    # Backbone selection — string passed to common.model.get_cadrille_class.
    # Recognised: qwen2_vl (default), qwen2_5_vl, qwen3_vl. The selected
    # parent must match the `base_model:` checkpoint architecture.
    backbone              = str(cfg.get('backbone', 'qwen2_vl'))
    # Online max_iou@K (sampling at temperature) on IoU buckets. K=0 → off.
    # Default: K=8 t=1.0 every eval. Bump every_n_evals for cheaper runs.
    max_iou_k             = int(cfg.get('max_iou_k', 8))
    max_iou_temperature   = float(cfg.get('max_iou_temperature', 1.0))
    max_iou_every_n_evals = int(cfg.get('max_iou_every_n_evals', 1))
    # Curriculum learning: list of {at_step: int, sft_mix_weights: {...}}.
    # When set, the static `sft_mix_weights` above is overridden — the
    # sampler picks active weights from the running global_step at each
    # epoch boundary. First phase MUST start at step 0.
    curriculum_phases     = cfg.get('curriculum_phases', None)
    # v4-holdout: alternate train pkl + holdout family list (consumed by online_eval).
    benchcad_train_pkl        = cfg.get('benchcad_train_pkl', None)
    cad_iso_106_train_pkl     = cfg.get('cad_iso_106_train_pkl', None)
    benchcad_simple_train_pkl = cfg.get('benchcad_simple_train_pkl', None)
    holdout_families          = cfg.get('holdout_families', None)
    # §7 v2: bench-simple op-pattern holdout (separate from v1 mech holdout
    # so a config can use both — e.g. ood_v2 holds out 10 mech AND 10 simple_op).
    holdout_families_v2       = cfg.get('holdout_families_v2', None)
    # Optional: cap the total unique training rows across all sources to a
    # fixed number, subsampling each source proportionally to its mix weight.
    # Default None → use full source sizes (back-compat). Set to e.g. 500000
    # to enforce identical pool-size across configs whose source composition
    # differs (controls for "data volume" confound in §7 v2 ablation).
    total_train_dp            = cfg.get('total_train_dp', None)
    # Explicit per-source row counts. Takes priority over total_train_dp —
    # when set, train.py truncates each source to exactly the listed count
    # (using the same per-source rng seed for cross-config row identity).
    # Generated by data_prep/compute_v2_pool_rows.py from the saturate-
    # redistribute math, then committed back into the yaml for explicit
    # provenance ("trained on EXACTLY these rows; here's the recipe").
    sft_pool_rows             = cfg.get('sft_pool_rows', None)

    # Resolve effective batch/accum for name generation (mirrors run() auto logic)
    eff_batch = batch_size or (8 if use_text else 28)
    eff_accum = accum_steps or (4 if use_text else 1)

    # Resolve run identity — auto-generate name from hyperparams if not specified
    checkpoints_dir = _p(args.checkpoints_dir, cfg.get('checkpoints_dir'), './checkpoints')
    run_name_base   = _p(args.run_name,         cfg.get('run_name'),         None)
    if run_name_base:
        run_name = run_name_base  # user-specified: use verbatim
    else:
        run_name = _auto_run_name('sft', max_steps, learning_rate,
                                  eff_batch, eff_accum, mode)
    output_dir = os.path.join(checkpoints_dir, run_name)

    # Resolved config dict for saving alongside the checkpoint
    resolved_cfg = {
        'run_name':               run_name,
        'checkpoints_dir':        checkpoints_dir,
        'output_dir':             output_dir,
        'config_file':            args.config,
        'data_path':              data_path,
        'mode':                   mode,
        'use_text':               use_text,
        'max_steps':              max_steps,
        'batch_size':             batch_size,
        'accumulation_steps':     accum_steps,
        'effective_batch':        eff_batch * eff_accum,
        'learning_rate':          learning_rate,
        'warmup_steps':           warmup_steps,
        'lr_scheduler_type':      lr_scheduler_type,
        'bf16':                   bf16,
        'tf32':                   tf32,
        'gradient_checkpointing': gradient_checkpointing,
        'optim':                  optim,
        'dataloader_workers':     dataloader_workers,
        'log_steps':              log_steps,
        'save_steps':             save_steps,
        'eval_steps':             eval_steps,
        'eval_on_start':          eval_on_start,
        'wandb_project':          wandb_project,
        'base_model':             base_model,
        'seed':                   seed,
        'max_code_len':           max_code_len,
        'sft_mix_weights':        sft_mix_weights,
        'hf_upload_repo':         hf_upload_repo,
        'hf_upload_private':      hf_upload_private,
        'group_by_length':        group_by_length,
        'save_only_model':        save_only_model,
        'save_total_limit':       save_total_limit,
        'online_eval_n_per':      online_eval_n_per,
        'max_iou_k':              max_iou_k,
        'max_iou_temperature':    max_iou_temperature,
        'max_iou_every_n_evals':  max_iou_every_n_evals,
        'curriculum_phases':      curriculum_phases,
        'benchcad_train_pkl':         benchcad_train_pkl,
        'cad_iso_106_train_pkl':      cad_iso_106_train_pkl,
        'benchcad_simple_train_pkl':  benchcad_simple_train_pkl,
        'holdout_families':           holdout_families,
        'holdout_families_v2':        holdout_families_v2,
        'total_train_dp':             total_train_dp,
        'sft_pool_rows':              sft_pool_rows,
    }

    print(f'Run name : {run_name}')
    print(f'Output   : {output_dir}')

    run(data_path, output_dir, mode, use_text, max_steps, batch_size,
        accum_steps, learning_rate, warmup_steps, lr_scheduler_type,
        dataloader_workers, log_steps,
        save_steps, eval_steps, wandb_project, eval_on_start,
        bf16, tf32, gradient_checkpointing, optim,
        seed=seed, max_code_len=max_code_len, sft_mix_weights=sft_mix_weights,
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint,
        cfg_to_save=resolved_cfg,
        hf_upload_repo=hf_upload_repo,
        hf_upload_private=hf_upload_private,
        group_by_length=group_by_length,
        save_only_model=save_only_model,
        save_total_limit=save_total_limit,
        online_eval_n_per=online_eval_n_per,
        backbone=backbone,
        max_iou_k=max_iou_k,
        max_iou_temperature=max_iou_temperature,
        max_iou_every_n_evals=max_iou_every_n_evals,
        curriculum_phases=curriculum_phases,
        benchcad_train_pkl=benchcad_train_pkl,
        cad_iso_106_train_pkl=cad_iso_106_train_pkl,
        benchcad_simple_train_pkl=benchcad_simple_train_pkl,
        holdout_families=holdout_families,
        holdout_families_v2=holdout_families_v2,
        total_train_dp=total_train_dp,
        sft_pool_rows=sft_pool_rows)


if __name__ == '__main__':
    main()
