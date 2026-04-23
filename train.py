import os
import math
import yaml
from datetime import datetime
from functools import partial
from argparse import ArgumentParser

import torch
from torch.utils.data import ConcatDataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from cadrille import Cadrille, collate
from dataset import Text2CADDataset, CadRecodeDataset


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
# Training
# ---------------------------------------------------------------------------

def run(data_path, output_dir, mode, use_text, max_steps, batch_size_override,
        accum_steps_override, learning_rate, warmup_steps, lr_scheduler_type,
        dataloader_workers, log_steps,
        save_steps, eval_steps, wandb_project, eval_on_start,
        bf16, tf32, gradient_checkpointing, optim,
        seed=42, max_code_len=None,
        base_model='Qwen/Qwen2-VL-2B-Instruct',
        resume_from_checkpoint=None, cfg_to_save=None):

    os.makedirs(output_dir, exist_ok=True)

    # Resolve "latest" checkpoint shortcut
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
    train_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split='train',
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=0.01,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode,
        max_code_len=max_code_len)
    batch_size = batch_size_override or 28
    accumulation_steps = accum_steps_override or 1

    if use_text:
        text_dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='train',
            max_code_len=max_code_len)
        train_dataset = ConcatDataset([train_dataset, text_dataset])
        batch_size = batch_size_override or 8
        accumulation_steps = accum_steps_override or 4

    # val.pkl is optional — skip evaluation when it doesn't exist
    val_pkl = os.path.join(cad_recode_path, 'val.pkl')
    has_val = os.path.exists(val_pkl)
    eval_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split='val',
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=None,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode,
        max_code_len=max_code_len) if has_val else None

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

    trainer = Trainer(
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
            bf16=bf16,
            tf32=tf32 if tf32 else None,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant': False} if gradient_checkpointing else None,
            optim=optim,
            remove_unused_columns=False,
            logging_first_step=True,
            logging_steps=log_steps,
            save_total_limit=2,
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
        callbacks=[PrintToFileCallback(), WandbRunSaverCallback()])
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Always save final checkpoint (regardless of save_steps cadence)
    final_dir = os.path.join(output_dir, 'checkpoint-final')
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f'Final checkpoint saved → {final_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
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
    sft_mix_weights       = cfg.get('sft_mix_weights', None)  # metadata only — see README

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
    }

    print(f'Run name : {run_name}')
    print(f'Output   : {output_dir}')

    run(data_path, output_dir, mode, use_text, max_steps, batch_size,
        accum_steps, learning_rate, warmup_steps, lr_scheduler_type,
        dataloader_workers, log_steps,
        save_steps, eval_steps, wandb_project, eval_on_start,
        bf16, tf32, gradient_checkpointing, optim,
        seed=seed, max_code_len=max_code_len,
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint,
        cfg_to_save=resolved_cfg)
