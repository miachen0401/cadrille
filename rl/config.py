"""Config loading and argument resolution for RL training.

Priority: CLI flag > YAML config > hardcoded default.
"""

import os
import math
import yaml
from datetime import datetime


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _p(cli_val, cfg_val, default):
    """CLI > YAML > default. None means 'not set'."""
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


def auto_run_name(mode: str, max_steps: int, lr: float, G: int) -> str:
    """rl-s{steps}-{lr}-G{G}-{mode}-MMDD-HHMM"""
    ts = datetime.now().strftime('%m%d-%H%M')
    return f'rl-s{_fmt_steps(max_steps)}-{_fmt_lr(lr)}-G{G}-{mode}-{ts}'


def resolve_args(args, cfg: dict) -> dict:
    """Merge CLI args + YAML config into args namespace. Returns resolved cfg dict.

    args has at minimum: config, run_name, checkpoint_path, max_steps, wandb_offline.
    All other fields are set from cfg with hardcoded defaults.
    """
    # Core
    args.mode            = _p(args.mode,           cfg.get('mode'),           'cppo')
    args.base_model      = cfg.get('base_model', 'Qwen/Qwen2-VL-2B-Instruct')
    args.checkpoint_path = _p(args.checkpoint_path, cfg.get('checkpoint_path'), 'maksimko123/cadrille')
    args.max_steps       = _p(args.max_steps,       cfg.get('max_steps'),       50000)
    args.lr              = cfg.get('lr', 3e-5)
    args.log_steps       = cfg.get('log_steps', 100)
    args.save_steps      = cfg.get('save_steps', 5000)

    # Validation — multi-dataset, multi-modality
    _legacy_dir     = cfg.get('val_split_dir')
    _legacy_samples = cfg.get('val_samples', 0)
    args.val_deepcad_dir       = cfg.get('val_deepcad_dir', _legacy_dir)
    args.val_fusion360_dir     = cfg.get('val_fusion360_dir')
    args.val_samples_deepcad   = cfg.get('val_samples_deepcad',
                                         _legacy_samples if _legacy_samples else 25)
    args.val_samples_fusion360 = cfg.get('val_samples_fusion360', 25)
    args.val_modalities        = cfg.get('val_modalities', 'pc')
    args.eval_steps            = cfg.get('eval_steps', 500)

    # W&B
    args.wandb_project  = cfg.get('wandb_project')
    args.wandb_entity   = cfg.get('wandb_entity')
    args.wandb_offline  = args.wandb_offline or bool(cfg.get('wandb_offline', False))

    # Data
    args.data_dir          = cfg.get('data_dir')
    args.hard_examples_pkl = cfg.get('hard_examples_pkl')
    args.dpo_dataset       = cfg.get('dpo_dataset')

    # CPPO hyperparameters (official: G=16, top_N=4, batch_updates=3, max_new_tokens=400)
    args.G                = cfg.get('G', 8)
    args.top_N            = cfg.get('top_N', 4)
    args.eps_high         = cfg.get('eps_high', 0.1)
    args.eps_low          = cfg.get('eps_low', 0.1)
    args.batch_updates    = cfg.get('batch_updates', 1)
    args.K_update         = cfg.get('K_update', 10)
    args.max_new_tokens   = cfg.get('max_new_tokens', 256)
    args.reward_workers   = cfg.get('reward_workers', 4)
    # sequential_generation: CLI --sequential-generation overrides YAML
    _seq = getattr(args, 'sequential_generation', None)
    if not _seq:
        _seq = cfg.get('sequential_generation', False)
    args.sequential_generation = bool(_seq)

    # DPO
    args.beta                 = cfg.get('beta', 0.3)
    args.dpo_epochs_per_round = cfg.get('dpo_epochs_per_round', 10)

    # Run identity
    checkpoints_dir = _p(getattr(args, 'checkpoints_dir', None),
                         cfg.get('checkpoints_dir'), './checkpoints')
    run_name_base   = _p(getattr(args, 'run_name', None),
                         cfg.get('run_name'), None)
    run_name = run_name_base or auto_run_name(args.mode, args.max_steps, args.lr, args.G)
    args.output_dir     = os.path.join(checkpoints_dir, run_name)
    args.wandb_run_name = cfg.get('wandb_run_name', run_name)
    args.run_name       = run_name

    # Resume step: auto-detect from checkpoint directory name (e.g. checkpoint-5000 → 5000)
    start_step = 0
    ckpt_path = getattr(args, 'checkpoint_path', None)
    if ckpt_path:
        ckpt_name = os.path.basename(ckpt_path.rstrip('/'))
        if ckpt_name.startswith('checkpoint-') and ckpt_name[len('checkpoint-'):].isdigit():
            start_step = int(ckpt_name[len('checkpoint-'):])
    args.start_step = start_step

    # Return resolved config dict for saving alongside checkpoint
    resolved = {k: v for k, v in vars(args).items()
                if not k.startswith('_') and k != 'config'}
    resolved['config_file'] = getattr(args, 'config', None)
    return resolved
