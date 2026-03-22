"""Unified Cadrille eval runner.

Usage
-----
# Run all checkpoints × datasets × modalities in config:
    python3 -m eval.runner configs/eval/quick.yaml

# Override output dir:
    python3 -m eval.runner configs/eval/compare.yaml --out eval_outputs/my_run

# Re-generate report only (no inference):
    python3 -m eval.runner configs/eval/quick.yaml --report-only

# Dry run — print plan, don't execute:
    python3 -m eval.runner configs/eval/quick.yaml --dry-run

Output layout
-------------
    eval_outputs/{tag}/
        config.yaml                    ← copy of config used
        {ckpt_label}/
            {dataset}_{modality}/
                metadata.jsonl         ← per-case: case_id, iou, cd, error_type, code_len
                {case_id}.py           ← generated CadQuery code
                {case_id}.stl          ← predicted mesh (success cases)
                passk.json             ← pass@k results (if enabled)
        renders/
            gt/{dataset}/{case_id}.png
            pred/{ckpt_label}/{dataset}_{modality}/{case_id}.png
        report.md                      ← auto-generated markdown report
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoProcessor

from cadrille import Cadrille
from eval.config import EvalConfig
from eval.pipeline import run_combo, _load_all_stls
from eval.render import select_cases, copy_gt_renders, render_pred_stls
from eval.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Unified Cadrille eval runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('config', help='Path to YAML eval config')
    parser.add_argument(
        '--out',
        default=None,
        help='Override output base dir (default: eval_outputs/{tag})',
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Skip inference; only regenerate report from existing metadata',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print plan without running anything',
    )
    parser.add_argument(
        '--ckpt',
        nargs='*',
        help='Subset of checkpoint labels to run (default: all)',
    )
    parser.add_argument(
        '--modality',
        nargs='*',
        help='Subset of modalities to run (default: from config)',
    )
    parser.add_argument(
        '--dataset',
        nargs='*',
        help='Subset of datasets to run (default: all in config)',
    )
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)

    if args.out:
        cfg.out_dir = str(Path(args.out).parent)
        cfg.tag = Path(args.out).name

    if args.modality:
        cfg.modalities = args.modality

    if args.ckpt:
        cfg.checkpoints = [c for c in cfg.checkpoints if c.label in args.ckpt]

    if args.dataset:
        cfg.datasets = {k: v for k, v in cfg.datasets.items() if k in args.dataset}

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_yaml = cfg.to_yaml()
    (run_dir / 'config.yaml').write_text(cfg_yaml)

    print(f'\n{"================================================================"}')
    print(f'Eval run: {run_dir}')
    print(f'{"================================================================"}')
    print(f'  Checkpoints : {[c.label for c in cfg.checkpoints]}')
    print(f'  Datasets    : {list(cfg.datasets)}')
    print(f'  Modalities  : {cfg.modalities}')
    print(f'  pass@k      : {"enabled" if cfg.pass_k.enabled else "disabled"}')
    print(f'  Renders     : {"enabled" if cfg.render.enabled else "disabled"}')
    print()

    if args.dry_run:
        print('Dry run — exiting.')
        return

    if args.report_only:
        print('Report-only mode — regenerating report...')
        report = generate_report(run_dir, cfg_yaml)
        print(f'Report written to {run_dir / "report.md"}')
        return

    _check_resources()

    print('Loading processor...', flush=True)
    processor = AutoProcessor.from_pretrained(
        cfg.base_model,
        min_pixels=200704,
        max_pixels=1003520,
        padding_side='left',
    )

    for ckpt_cfg in cfg.checkpoints:
        ckpt_path = ckpt_cfg.resolved_path()
        ckpt_label = ckpt_cfg.label
        ckpt_dir_name = ckpt_label.replace('/', '_').replace(' ', '_')

        print(f'\n{"────────────────────────────────────────────────────────────────"}')
        print(f'Checkpoint: {ckpt_label}  ({ckpt_path})')
        print(f'{"────────────────────────────────────────────────────────────────"}', flush=True)

        if not (ckpt_path / 'model.safetensors').exists():
            print(f'  SKIP: model.safetensors not found at {ckpt_path}')
            continue

        print('  Loading model...', flush=True)
        model = Cadrille.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map='cuda',
        )
        model.eval()

        for ds_name, ds_cfg in cfg.datasets.items():
            ds_path = ds_cfg.resolved_path()
            if not ds_path.exists():
                print(f'  SKIP dataset {ds_name}: path not found ({ds_path})')
                continue

            all_stls = _load_all_stls(ds_path, ds_cfg.n_samples)
            print(f'\n  Dataset: {ds_name}  ({len(all_stls)} STLs)', flush=True)

            for modality in cfg.modalities:
                combo_dir = run_dir / ckpt_dir_name / f'{ds_name}_{modality}'
                print(f'\n  [{modality}] → {combo_dir}', flush=True)

                res = cfg.resources
                batch_size = res.batch_size_img if modality == 'img' else res.batch_size_pc
                prep_threads = res.prep_threads_img if modality == 'img' else res.prep_threads_pc

                summary = run_combo(
                    model=model,
                    processor=processor,
                    stl_paths=all_stls,
                    modality=modality,
                    out_dir=combo_dir,
                    batch_size=batch_size,
                    max_new_tokens=cfg.max_new_tokens,
                    score_workers=res.score_workers,
                    prep_threads=prep_threads,
                    queue_size=res.queue_size,
                    save_code=res.save_code,
                    save_stl=res.save_stl,
                )

                _print_summary(ckpt_label, ds_name, modality, summary)

                if cfg.pass_k.enabled:
                    from eval.passk import run_passk
                    run_passk(
                        model=model,
                        processor=processor,
                        stl_paths=all_stls,
                        modality=modality,
                        out_dir=combo_dir,
                        n_samples=cfg.pass_k.n_samples,
                        k_values=cfg.pass_k.k,
                        iou_threshold=cfg.pass_k.iou_threshold,
                        temperature=cfg.pass_k.temperature,
                        batch_size=batch_size,
                        max_new_tokens=cfg.max_new_tokens,
                        score_workers=res.score_workers,
                    )

                if cfg.render.enabled:
                    meta_path = combo_dir / 'metadata.jsonl'
                    case_ids = select_cases(meta_path, cfg.render.strategy, cfg.render.n)
                    if case_ids:
                        gt_dir = run_dir / 'renders' / 'gt' / ds_name
                        pred_dir = run_dir / 'renders' / 'pred' / ckpt_dir_name / f'{ds_name}_{modality}'
                        n_gt = copy_gt_renders(case_ids, ds_path, gt_dir)
                        n_pred = render_pred_stls(case_ids, combo_dir, pred_dir)
                        print(f'    Renders: {n_gt} GT, {n_pred} pred copied/rendered')

        del model
        torch.cuda.empty_cache()
        print('\n  Model unloaded.', flush=True)

    print(f'\n{"================================================================"}')
    print('Generating report...')
    report = generate_report(run_dir, cfg_yaml)
    print(f'Report → {run_dir / "report.md"}')
    print(f'{"================================================================"}\n')


def _check_resources() -> None:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader'],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        free_mb = int(result.stdout.strip().split()[0])
        if free_mb < 2000:
            print(f'WARNING: Only {free_mb} MiB GPU VRAM free. May OOM.')
            return
        return
    print('nvidia-smi not available; skipping GPU check.')


def _print_summary(ckpt: str, ds: str, mod: str, s: dict) -> None:
    n = s.get('n', 0)
    if n == 0:
        return
    iou = s.get('mean_iou', 0) * 100
    fail = s.get('failure_rate', 0) * 100
    cd = s.get('mean_cd')
    cd_str = f'  CD={cd:.4f}' if cd else ''
    print(f'    → IoU={iou:.2f}%  fail={fail:.1f}%{cd_str}  (n={n})', flush=True)


if __name__ == '__main__':
    main()
