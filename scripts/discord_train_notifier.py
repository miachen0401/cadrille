"""Post current RL training status to Discord webhook.

Reads DISCORD_WEBHOOK_URL from env. Skips POST if URL is unset / looks
like a placeholder.

Pulls:
  * latest step + train scalars from wandb (if WANDB_API_KEY + run id available)
  * latest eval line from <output_dir>/log.txt
  * recent error lines from rl log file
  * process state (PID alive, GPU mem)

Usage:
    set -a; source ~/.bashrc; set +a   # ensure DISCORD_WEBHOOK_URL exported
    uv run python scripts/discord_train_notifier.py \
        --rl-log logs/rl_ess_iou_from_24k.log \
        --output-dir /ephemeral/checkpoints/rl-ess-iou-0.2-G16-from-ood50k-24k \
        --run-tag "ess-iou-v1 (24k init)"
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print('requests not installed; pip install requests')
    sys.exit(0)


def _is_real_webhook(url: str | None) -> bool:
    if not url:
        return False
    if 'webhook_url' in url or 'YOUR_WEBHOOK' in url or '占位' in url or '你的' in url:
        return False
    return url.startswith('https://discord.com/api/webhooks/') or url.startswith('https://discordapp.com/api/webhooks/')


def _last_eval(eval_log: Path) -> str | None:
    if not eval_log.exists():
        return None
    lines = [l.strip() for l in eval_log.read_text().splitlines() if l.strip()]
    return lines[-1] if lines else None


def _last_steps(rl_log: Path, n: int = 3) -> list[str]:
    if not rl_log.exists():
        return []
    txt = rl_log.read_text(errors='replace')
    step_lines = [l for l in txt.splitlines() if re.search(r'\bstep[ =]\d+\b', l)]
    return step_lines[-n:]


def _recent_errors(rl_log: Path, last_n_lines: int = 200) -> list[str]:
    if not rl_log.exists():
        return []
    txt = rl_log.read_text(errors='replace').splitlines()[-last_n_lines:]
    pat = re.compile(r'(Traceback|RuntimeError|Error[: ]|FAILED|OOM|Killed|exit code)')
    return [l.strip() for l in txt if pat.search(l)]


def _gpu_state() -> str | None:
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            used, total, util = [s.strip() for s in r.stdout.strip().split(',')]
            return f'GPU {used}/{total} MiB, util {util}%'
    except Exception:
        pass
    return None


def _wandb_run_url(rl_log: Path) -> str | None:
    if not rl_log.exists():
        return None
    m = re.search(r'(https://wandb\.ai/\S+/runs/\S+)', rl_log.read_text(errors='replace'))
    return m.group(1).rstrip('\x1b[m') if m else None


def _proc_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid(rl_log: Path) -> int | None:
    """Find the python process PID by command-line match."""
    try:
        r = subprocess.run(['pgrep', '-f', 'train.rl.train'],
                           capture_output=True, text=True, timeout=3)
        if r.stdout.strip():
            return int(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rl-log', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True,
                    help='train_cppo output dir (contains log.txt with eval scalars)')
    ap.add_argument('--run-tag', default='cadrille-rl',
                    help='Short label for the Discord message header')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print message instead of posting')
    args = ap.parse_args()

    webhook = os.environ.get('DISCORD_WEBHOOK_URL')
    if not _is_real_webhook(webhook) and not args.dry_run:
        print(f'[skip] DISCORD_WEBHOOK_URL not set or looks like a placeholder: {webhook!r}',
              file=sys.stderr)
        sys.exit(0)

    pid = _read_pid(args.rl_log)
    alive = _proc_alive(pid)
    eval_line = _last_eval(args.output_dir / 'log.txt')
    step_lines = _last_steps(args.rl_log)
    errs = _recent_errors(args.rl_log)
    gpu = _gpu_state()
    wburl = _wandb_run_url(args.rl_log)

    status_emoji = '🟢' if alive else ('❌' if errs else '⚪')

    parts = [f'{status_emoji} **{args.run_tag}**  (PID {pid or "?"}, '
             f'{"alive" if alive else "DEAD"})']
    if gpu:
        parts.append(f'`{gpu}`')
    if eval_line:
        parts.append(f'**Eval**: `{eval_line}`')
    if step_lines:
        parts.append('**Recent steps**:\n```\n' + '\n'.join(step_lines) + '\n```')
    if errs:
        parts.append('**⚠️ Errors**:\n```\n' + '\n'.join(errs[-5:]) + '\n```')
    if wburl:
        parts.append(f'**wandb**: <{wburl}>')

    msg = '\n'.join(parts)
    # Discord 2000-char limit per message
    if len(msg) > 1900:
        msg = msg[:1900] + '\n... (truncated)'

    if args.dry_run:
        print(msg)
        return

    try:
        r = requests.post(webhook,
                          json={'content': msg, 'username': 'cadrille-rl'},
                          timeout=10)
        if r.status_code >= 400:
            print(f'[discord] HTTP {r.status_code}: {r.text[:200]}', file=sys.stderr)
        else:
            print(f'[discord] posted ({len(msg)} chars)', file=sys.stderr)
    except Exception as e:
        print(f'[discord] post failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
