"""Pull predictions/ subdir from each §7 v2 HF model repo to local.

Used by the analyst (any machine) to aggregate per-step JSONLs from all 5
parallel HPC training runs. After sync, plot_main_appendix.py reads from
the local synced dir as if all runs were on the same box.

Each per-config repo is configured in `train/sft/hf_uploader.py` to push
both checkpoint-N/ AND predictions/ on each save_steps. Predictions are
small (~few MB total) so a fresh sync each time the analyst regenerates
figures is cheap.

Usage:
    set -a; source .env; set +a
    uv run python scripts/sync_predictions_from_hf.py \
        [--out eval_outputs/v2_synced] [--repos baseline_v2,ood_v2,...]

Output layout:
  eval_outputs/v2_synced/<config-name>/predictions/step-NNNNNN.jsonl
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Default §7 v2 → HF repo map. Mirrors hf_upload_repo in each *_v2.yaml.
DEFAULT_REPOS = {
    'baseline_v2':       'Hula0401/cadrille-qwen3vl-2b-baseline-v2-50k',
    'iid_enhanced_v2':       'Hula0401/cadrille-qwen3vl-2b-iid-enhanced-v2-50k',
    'ood_v2':            'Hula0401/cadrille-qwen3vl-2b-ood-v2-50k',
    'ood_enhanced_v2':   'Hula0401/cadrille-qwen3vl-2b-ood-enhanced-v2-50k',
    'iid_v2':            'Hula0401/cadrille-qwen3vl-2b-iid-v2-50k',
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',   type=Path, default=REPO / 'eval_outputs/v2_synced')
    ap.add_argument('--repos', default=','.join(DEFAULT_REPOS.keys()),
                    help='Comma-separated config keys to sync (default: all 5)')
    ap.add_argument('--cache-dir', type=Path,
                    default=Path('/ephemeral/_hf_pred_cache'),
                    help='Working cache for HF downloads (deleted after copy).')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    keys = [k.strip() for k in args.repos.split(',') if k.strip()]
    token = os.environ.get('HF_TOKEN')

    for key in keys:
        repo = DEFAULT_REPOS.get(key)
        if not repo:
            print(f'[skip] unknown config key: {key}'); continue
        print(f'\n=== {key} ({repo}) ===', flush=True)
        try:
            local = snapshot_download(
                repo_id=repo, repo_type='model', token=token,
                cache_dir=str(args.cache_dir),
                allow_patterns=['predictions/*.jsonl', 'predictions/*.csv'],
            )
        except Exception as e:
            print(f'  download failed: {e}', flush=True)
            continue
        src = Path(local) / 'predictions'
        if not src.is_dir():
            print(f'  no predictions/ dir on the repo yet — skip')
            continue
        dst = args.out / key / 'predictions'
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n = len(list(dst.glob('step-*.jsonl')))
        print(f'  → {dst} ({n} step JSONLs)', flush=True)
        # Drop the snapshot cache to avoid disk bloat
        shutil.rmtree(local, ignore_errors=True)
        repo_cache = args.cache_dir / f'models--{repo.replace("/", "--")}'
        shutil.rmtree(repo_cache, ignore_errors=True)

    print(f'\nsync done → {args.out}')
    print(f'next:  uv run python -m scripts.analysis.plot_main_appendix')


if __name__ == '__main__':
    main()
