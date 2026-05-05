# Multi-HPC parallel §7 v2 launch — 4-machine assignment

## TL;DR

Each HPC pulls the repo, runs **one** v2 config, training pushes ckpts +
predictions to HF in the background. An analyst (any 5th machine) syncs
all 5 from HF and regenerates the §7 figure on demand.

## HPC assignments (4 machines, 5 configs)

Configs are independent — order doesn't matter. Sample assignment:

| HPC | Config | Approx GPU-h | HF model repo |
|---|---|---|---|
| HPC1 | `configs/sft/baseline_v2.yaml`     | ~24h | `Hula0401/cadrille-qwen3vl-2b-baseline-v2-50k` |
| HPC2 | `configs/sft/iid_enhanced_v2.yaml` | ~24h | `Hula0401/cadrille-qwen3vl-2b-iid-enhanced-v2-50k` |
| HPC3 | `configs/sft/ood_v2.yaml`          | ~24h | `Hula0401/cadrille-qwen3vl-2b-ood-v2-50k` |
| HPC4 | `configs/sft/ood_enhanced_v2.yaml` | ~24h | `Hula0401/cadrille-qwen3vl-2b-ood-enhanced-v2-50k` |
| (queue) | `configs/sft/iid_v2.yaml`        | ~24h | `Hula0401/cadrille-qwen3vl-2b-iid-v2-50k` |

The 5th config (`iid_v2`) waits for the first HPC to finish, then runs
on whichever frees up first.

## On EACH HPC (single command, run from zero)

```bash
git clone https://github.com/miachen0401/cadrille.git
cd cadrille
cp .env.example .env && $EDITOR .env       # HF_TOKEN, WANDB_API_KEY required
bash scripts/setup.sh --data               # ~30-50 min, idempotent
nohup bash scripts/launch_one.sh configs/sft/<your-assignment>.yaml \
    > logs/run.log 2>&1 &
```

`launch_one.sh` does its own pre-flight (generates `essential_ops_simple.yaml`
and `data/benchcad-simple/train_v2_holdout.pkl` if absent), so no other
manual steps. Training auto-pushes ckpts + predictions/ to HF every
`save_steps` (default 2000).

## On the analyst machine (any time, no GPU needed)

```bash
set -a; source .env; set +a   # need HF_TOKEN read access
uv run python scripts/sync_predictions_from_hf.py \
    --out eval_outputs/v2_synced
uv run python -m scripts.analysis.plot_main_appendix
# §7 figures land in paper/figures/, can rerun any time as new ckpts upload
```

`sync_predictions_from_hf.py` pulls the `predictions/` subdir from each
of the 5 HF repos (~few MB total) and lays them out under
`eval_outputs/v2_synced/<config-name>/predictions/`. Idempotent — running
it again only downloads new step JSONLs.

## What HF receives

For each `Hula0401/cadrille-qwen3vl-2b-<config>-v2-50k` repo, training pushes:

```
checkpoint-2000/        # ~12 GB per ckpt (kept by save_total_limit=4)
checkpoint-4000/
…
predictions/            # ~50 KB per step JSONL × 25 steps = ~1.5 MB total
  step-001000.jsonl
  step-002000.jsonl
  step-002000.max@8.jsonl
  …
```

`predictions/` is rsync'd on each save (idempotent — HF dedupes) so the
analyst always sees the latest. ckpts get rotated locally (save_total_limit=4)
but stay on HF permanently, so retro re-eval is always possible.

## Resumability

Each HPC training run resumes from its last local ckpt automatically (HF
Trainer's `resume_from_checkpoint=null` skips, but if a step exists in
`output_dir/checkpoint-N`, train.py picks it up). To force-resume from HF:

```bash
huggingface-cli download Hula0401/cadrille-qwen3vl-2b-baseline-v2-50k \
    --include 'checkpoint-10000/**' --local-dir /ephemeral/checkpoints/<run>
# then edit yaml: resume_from_checkpoint: /ephemeral/checkpoints/<run>/checkpoint-10000
```

## Why HF (not Slack / shared FS / S3)

- **Versioned**: each step push is a commit, so the analyst can
  reproduce a specific revision.
- **No cluster ingress**: the analyst doesn't need access to the HPCs.
- **Free**: HF datasets/models are free up to 1 TB.
- **Already wired**: `train/sft/hf_uploader.py` does this on save.

## Failure modes

- **HF rate limit (1k req/h)**: predictions are pushed once per save, so
  with save_steps=2000 and 50k step ceiling, only 25 pushes per HPC over
  24h — well within limits.
- **Network outage**: uploads happen in a daemon thread, training never
  blocks. On reconnect, the next save retries.
- **HF revoke / token leak**: revoke + reissue, set the new value in
  `.env`, restart the training. Predictions resume from the next save.
