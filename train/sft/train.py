import os
import math
import yaml
from datetime import datetime
from functools import partial
from argparse import ArgumentParser

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from common.model import Cadrille, collate
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
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


def _build_callbacks(processor, seed, hf_upload_repo, hf_upload_private,
                     online_eval_n_per=20, mix_weights=None):
    from train.sft.online_eval import OnlineIoUEvalCallback
    cbs = [
        PrintToFileCallback(),
        WandbRunSaverCallback(),
        OnlineIoUEvalCallback(processor, n_per_dataset=online_eval_n_per,
                              seed=seed, mix_weights=mix_weights),
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

    def __init__(self, *args, sample_weights=None,
                 curriculum_phases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
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
        if self.args.group_by_length:
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
        curriculum_phases=None):

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
    batch_size = batch_size_override or 28
    accumulation_steps = accum_steps_override or 1

    if use_text:
        sources['text2cad'] = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='train',
            max_code_len=max_code_len)
        batch_size = batch_size_override or 8
        accumulation_steps = accum_steps_override or 4

    benchcad_path = os.path.join(data_path, 'benchcad')
    if os.path.isdir(benchcad_path) and os.path.exists(os.path.join(benchcad_path, 'train.pkl')):
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
            max_code_len=max_code_len)

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

    # val set selection: prefer benchcad when it's the dominant source
    # Prefer benchcad val whenever benchcad is in the mix (weight > 0) — that's
    # the metric we're tracking. Fall back to cad-recode only when benchcad is
    # absent entirely.
    eval_dataset = None
    benchcad_dominant = (
        sft_mix_weights
        and float(sft_mix_weights.get('benchcad', 0)) > 0
    )
    benchcad_val_pkl = os.path.join(benchcad_path, 'val.pkl')
    if benchcad_dominant and os.path.exists(benchcad_val_pkl):
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
    has_val = eval_dataset is not None

    processor = AutoProcessor.from_pretrained(
        base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')
    model = Cadrille.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2')

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
            per_device_train_batch_size=batch_size,
            dataloader_num_workers=dataloader_workers,
            max_steps=max_steps,
            lr_scheduler_type=lr_scheduler_type,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            gradient_accumulation_steps=accumulation_steps,
            group_by_length=group_by_length,
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
        tokenizer=processor,
        sample_weights=sample_weights,
        curriculum_phases=expanded_phases,
        callbacks=_build_callbacks(
            processor=processor, seed=seed,
            hf_upload_repo=hf_upload_repo,
            hf_upload_private=hf_upload_private,
            online_eval_n_per=online_eval_n_per,
            mix_weights=sft_mix_weights,
        ))
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
    # Curriculum learning: list of {at_step: int, sft_mix_weights: {...}}.
    # When set, the static `sft_mix_weights` above is overridden — the
    # sampler picks active weights from the running global_step at each
    # epoch boundary. First phase MUST start at step 0.
    curriculum_phases     = cfg.get('curriculum_phases', None)

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
        'curriculum_phases':      curriculum_phases,
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
        curriculum_phases=curriculum_phases)


if __name__ == '__main__':
    main()
