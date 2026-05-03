#!/usr/bin/env bash
# Watch the predictions/ dir for new step-NNNNNN.jsonl files, and fire
# scripts/analysis/eval_to_discord.py for each new step. Idempotent via
# {step}.posted marker. Designed to run alongside training.
#
# Usage:
#   nohup bash scripts/watch_eval_post_discord.sh \
#       /home/ubuntu/cadrille/logs/big_bench_shell_50k_20260427_061353.log \
#       /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0427-0613 \
#       > logs/watch_post_discord.log 2>&1 &

set -euo pipefail
LOG_PATH="${1:?usage: $0 <training-log> <ckpt-output-dir> [poll-secs]}"
OUT_DIR="${2:?usage: $0 <training-log> <ckpt-output-dir> [poll-secs]}"
POLL="${3:-60}"

PRED_DIR="$OUT_DIR/predictions"
WORKERS="${WORKERS:-4}"

# Load Discord webhook (handles non-interactive bashrc early-return)
eval "$(grep '^export DISCORD_WEBHOOK_URL' ~/.bashrc)"
[ -n "${DISCORD_WEBHOOK_URL:-}" ] || { echo "DISCORD_WEBHOOK_URL not set"; exit 2; }

cd /home/ubuntu/cadrille

echo "[watch] log=$LOG_PATH"
echo "[watch] out_dir=$OUT_DIR"
echo "[watch] poll=${POLL}s  workers=$WORKERS"

while true; do
    if [ -d "$PRED_DIR" ]; then
        for f in "$PRED_DIR"/step-*.jsonl; do
            [ -e "$f" ] || continue
            base=$(basename "$f" .jsonl)             # step-001000
            # Skip max@8 sample dump files (step-001000.max@8.jsonl) — only
            # the greedy step-NNNNNN.jsonl drives Discord posts.
            [[ "$base" == *.max@*  ]] && continue
            step_str="${base#step-}"                  # 001000
            step=$((10#$step_str))                    # 1000
            [ "$step" -eq 0 ] && continue             # skip eval_on_start
            marker="$PRED_DIR/${base}.posted"
            [ -e "$marker" ] && continue
            # Check that the eval block is actually present in the log
            if ! grep -q "step=${step} running IoU eval" "$LOG_PATH" 2>/dev/null; then
                continue
            fi
            # Wait until all expected buckets are logged for this step
            # (otherwise IoU for the last bucket might be missing). Heuristic:
            # count "[img/" + "[text/" lines after the marker.
            #
            # IMPORTANT: scope the block to step=N's eval block only —
            # stop at the NEXT "step=M running IoU eval" marker (M ≠ N).
            # Otherwise max_iou@K lines from later eval cycles can satisfy
            # the gate prematurely and trigger a Discord post for step=N
            # before its own max-IoU results have actually landed.
            block=$(awk -v m="step=${step} running IoU eval" '
                $0 ~ m {found=1; print; next}
                found && /running IoU eval/ {exit}
                found {print}
            ' "$LOG_PATH")
            # New online_eval splits BenchCAD val into IID + OOD when
            # holdout_families is set, so the post-refactor world has 5
            # buckets: BC val IID + BC val OOD + recode20k train + DC test +
            # Fu test. Pre-refactor runs emit a single 'BenchCAD val' bucket
            # → still hits 4. text2cad train is skipped at 0 items.
            n_done=$(echo "$block" | grep -cE '\[(img|text)/(BenchCAD val( IID| OOD)?|recode20k train|text2cad train|DeepCAD test|Fusion360 test)\]' || true)
            n_thresh=4
            if echo "$block" | grep -qE '\[(img|text)/BenchCAD val (IID|OOD)\]'; then
                n_thresh=5
            fi
            if [ "$n_done" -lt "$n_thresh" ]; then
                echo "[watch] step=$step only $n_done/$n_thresh buckets logged, waiting"
                continue
            fi
            # max_iou@K cycle: fires at step 1k, 3k, 5k, 7k, ... (odd thousand).
            # When applicable, wait for all 3 max_iou@K bucket lines too so the
            # Discord post includes both greedy + max@K. K may be 8 or 16
            # depending on cfg.max_iou_k — match either via @[0-9]+.
            half_step=$((step / 1000))
            if [ $((half_step % 2)) -eq 1 ]; then
                n_max=$(echo "$block" | grep -cE '\] max_iou@[0-9]+ \(t=[0-9.]+\)=' || true)
                if [ "$n_max" -lt 3 ]; then
                    echo "[watch] step=$step max_iou@K only $n_max/3 buckets logged, waiting"
                    continue
                fi
            fi

            echo "[watch] $(date -u +%H:%M:%S) firing eval_to_discord for step=$step"
            if uv run python -m scripts.analysis.eval_to_discord \
                    --step "$step" \
                    --log "$LOG_PATH" \
                    --output-dir "$OUT_DIR" \
                    --workers "$WORKERS"; then
                touch "$marker"
                echo "[watch] step=$step posted, marker=$marker"
            else
                echo "[watch] step=$step FAILED, will retry next poll"
            fi

            # Every 5000 steps, refresh the §7 main+appendix figure suite
            # and post — keeps trajectory plots up-to-date across all 4 runs
            # without manual intervention.
            if [ $((step % 5000)) -eq 0 ] && [ "$step" -ge 5000 ]; then
                echo "[watch] step=$step refreshing §7 figure suite ..."
                if uv run python -m scripts.analysis.plot_main_appendix > /dev/null; then
                    uv run python -m scripts.analysis.eval_to_discord --send \
                        --message "§7 figure suite refresh @ step ${step}" \
                        --file paper/figures/fig_7_4line_ess_pass.png \
                        --file paper/figures/fig_7_ood_iou_4line.png \
                        --file paper/figures/fig_app_ood_exec.png \
                        --file paper/figures/fig_app_iid_ess_pass.png \
                        --file paper/figures/fig_app_iid_iou.png \
                        --file paper/figures/fig_app_iid_exec.png \
                        --file paper/figures/fig_app_deepcad_iou.png \
                        --file paper/figures/fig_app_fusion360_iou.png \
                        || echo "[watch] §7 figure post failed (non-fatal)"
                fi
            fi
        done
    fi
    sleep "$POLL"
done
