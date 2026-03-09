"""
Cadrille pipeline daemon — runs forever.

State machine:
  MINING_DEEPCAD  → MINING_FUSION360 → MERGING → TRAINING → EVAL → DONE
  (each state restarts its job if it crashes; never exits on its own)

Usage:
  nohup python3 scripts/daemon.py > logs/daemon.log 2>&1 &
"""

import os, sys, time, pickle, subprocess, signal, re, textwrap
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
os.chdir(ROOT)

LOGS = ROOT / "logs"
MINED = ROOT / "data" / "mined"
LOGS.mkdir(exist_ok=True)
MINED.mkdir(exist_ok=True)

# ── tunables ─────────────────────────────────────────────────────────────────
CHECK_INTERVAL   = 60    # seconds between heartbeats
PROGRESS_INTERVAL = 600  # seconds between progress.md updates (10 min)

DEEPCAD_OUTPUT   = MINED / "deepcad_hard.pkl"
FUSION360_OUTPUT = MINED / "fusion360_hard.pkl"
COMBINED_OUTPUT  = MINED / "combined_hard.pkl"

DEEPCAD_MAX_SAMPLES   = 20000
FUSION360_MAX_SAMPLES = 8000

# ── helpers ──────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    # Also return False for zombie processes (defunct — finished but not yet reaped)
    try:
        status = open(f"/proc/{pid}/status").read()
        if "State:\tZ" in status:
            return False
    except FileNotFoundError:
        return False
    return True

def count_processed(pkl_path):
    proc_file = Path(str(pkl_path) + ".processed")
    if not proc_file.exists():
        return 0
    return sum(1 for _ in open(proc_file))

def count_hard(pkl_path):
    if not Path(pkl_path).exists():
        return 0
    try:
        return len(pickle.load(open(pkl_path, "rb")))
    except Exception:
        return 0

def gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5).decode().strip()
        return int(out.split()[0])
    except Exception:
        return -1

def launch(cmd, log_path):
    """Launch a subprocess, return its PID."""
    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            cmd, shell=False,
            stdout=lf, stderr=lf,
            start_new_session=True)
    return proc.pid

def update_progress(state, extra=""):
    """Append a one-line heartbeat to progress.md."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    dc_proc = count_processed(DEEPCAD_OUTPUT)
    dc_hard = count_hard(DEEPCAD_OUTPUT)
    f3_proc = count_processed(FUSION360_OUTPUT)
    f3_hard = count_hard(FUSION360_OUTPUT)
    line = (f"- [{ts}] daemon state={state} | "
            f"dc={dc_proc}/{DEEPCAD_MAX_SAMPLES} hard={dc_hard} | "
            f"f3={f3_proc}/{FUSION360_MAX_SAMPLES} hard={f3_hard}")
    if extra:
        line += f" | {extra}"
    with open(ROOT / "progress.md", "a") as f:
        f.write(line + "\n")
    log(f"progress.md updated: {line}")

# ── state machine ─────────────────────────────────────────────────────────────

class State:
    MINING_DEEPCAD   = "MINING_DEEPCAD"
    MINING_FUSION360 = "MINING_FUSION360"
    MERGING          = "MERGING"
    TEMP_EVAL        = "TEMP_EVAL"     # temperature sweep before training
    TRAINING         = "TRAINING"
    EVAL             = "EVAL"
    DONE             = "DONE"

def make_mine_cmd(data_dir, output, max_samples):
    return [
        "python3", "rl/mine.py",
        "--checkpoint-path", "./checkpoints/cadrille-sft",
        "--data-dir",        data_dir,
        "--output",          str(output),
        "--modality",        "img",
        "--K",               "1",
        "--R-th",            "0.75",
        "--max-samples",     str(max_samples),
        "--max-new-tokens",  "400",
        "--temperature",     "0",
        "--batch-size",      "4",
        "--reward-workers",  "8",
        "--checkpoint-every","500",
        "--resume",
    ]

def make_train_cmd(run_name):
    return [
        "python3", "rl/train.py",
        "--config",   "configs/rl/4080.yaml",
        "--run-name", run_name,
    ]

def patch_train_config_for_mined():
    """Update 4080.yaml to use mined pkl and clear data_dir."""
    cfg_path = ROOT / "configs" / "rl" / "4080.yaml"
    cfg = cfg_path.read_text()
    cfg = re.sub(r'^hard_examples_pkl:.*$',
                 f'hard_examples_pkl: ./data/mined/combined_hard.pkl',
                 cfg, flags=re.M)
    cfg = re.sub(r'^data_dir:.*$', 'data_dir: null', cfg, flags=re.M)
    cfg_path.write_text(cfg)
    log("configs/rl/4080.yaml patched: hard_examples_pkl set, data_dir=null")

def do_merge():
    dc = pickle.load(open(DEEPCAD_OUTPUT, "rb"))
    f3 = pickle.load(open(FUSION360_OUTPUT, "rb"))
    combined = dc + f3
    with open(COMBINED_OUTPUT, "wb") as f:
        pickle.dump(combined, f)
    log(f"Merged: deepcad={len(dc)} + fusion360={len(f3)} = {len(combined)} hard examples → {COMBINED_OUTPUT}")
    return len(combined)

def hf_upload(local_path, remote_name):
    try:
        subprocess.run([
            "huggingface-cli", "upload", "Hula0401/mine_CAD",
            str(local_path), remote_name, "--repo-type=dataset"
        ], timeout=300, check=True)
        log(f"HF upload done: {remote_name}")
    except Exception as e:
        log(f"HF upload failed (non-fatal): {e}")

def run_eval(checkpoint_path):
    ts = datetime.now().strftime("%m%d-%H%M")
    eval_log = LOGS / f"eval-{ts}.log"
    cmd = [
        "python3", "tools/eval_img.py",
        "--checkpoint", checkpoint_path,
        "--splits", "deepcad", "--splits", "fusion360",
        "--n-samples", "500",
        "--out-dir", f"./work_dirs/eval-{ts}",
    ]
    log(f"Starting eval: {' '.join(cmd)}")
    log(f"Eval log: {eval_log}")
    pid = launch(cmd, eval_log)
    log(f"Eval PID: {pid}")
    return pid, str(eval_log)

# ── main loop ─────────────────────────────────────────────────────────────────

STATE_FILE = LOGS / "daemon_state.json"

def save_state(state, job_pid):
    with open(STATE_FILE, "w") as f:
        import json as _j
        _j.dump({"state": state, "job_pid": job_pid}, f)

def load_state():
    """Infer current pipeline state from completed artifacts + running processes."""
    # Check what's already done
    dc_done  = count_processed(DEEPCAD_OUTPUT)   >= DEEPCAD_MAX_SAMPLES
    f3_done  = count_processed(FUSION360_OUTPUT)  >= FUSION360_MAX_SAMPLES
    combined = COMBINED_OUTPUT.exists()

    # Check what's running
    try:
        mine_out = subprocess.check_output(
            ["pgrep", "-f", "rl/mine.py"], timeout=5).decode().strip()
        mine_pids = [int(p) for p in mine_out.split()] if mine_out else []
    except Exception:
        mine_pids = []

    try:
        train_out = subprocess.check_output(
            ["pgrep", "-f", "rl/train.py"], timeout=5).decode().strip()
        train_pid = int(train_out.strip()) if train_out.strip() else None
    except Exception:
        train_pid = None

    # Determine state
    if train_pid:
        return State.TRAINING, train_pid
    if combined:
        return State.TEMP_EVAL, None   # will start temp eval or proceed to training
    if f3_done:
        return State.MERGING, None
    if dc_done:
        # Fusion360 mining should be running or needs starting
        pid = mine_pids[0] if mine_pids else None
        return State.MINING_FUSION360, pid
    # DeepCAD mining
    pid = mine_pids[0] if mine_pids else None
    return State.MINING_DEEPCAD, pid


def main():
    last_progress = 0.0
    train_run_name = None
    eval_pid = None

    state, job_pid = load_state()
    log(f"Inferred state: {state}, job_pid: {job_pid}")

    log("=" * 60)
    log("Cadrille pipeline daemon started")
    log(f"  State: {state}")
    log(f"  Mine PID: {job_pid}")
    log("=" * 60)

    while True:
        now = time.time()

        # ── periodic progress.md update ──────────────────────────────────────
        if now - last_progress > PROGRESS_INTERVAL:
            extra = f"state={state} mine_pid={job_pid}"
            update_progress(state, extra)
            last_progress = now

        # ── state transitions ─────────────────────────────────────────────────

        if state == State.MINING_DEEPCAD:
            dc_proc = count_processed(DEEPCAD_OUTPUT)
            dc_hard = count_hard(DEEPCAD_OUTPUT)
            log(f"[MINING_DEEPCAD] pid={job_pid} alive={pid_alive(job_pid)} "
                f"processed={dc_proc}/{DEEPCAD_MAX_SAMPLES} hard={dc_hard}")

            if dc_proc >= DEEPCAD_MAX_SAMPLES and not pid_alive(job_pid):
                # Done
                log(f"DeepCAD mining complete: {dc_proc} scanned, {dc_hard} hard")
                state = State.MINING_FUSION360
                job_pid = None

            elif not pid_alive(job_pid):
                # Not done yet → restart
                log("DeepCAD mining not running → (re)starting with --resume")
                mine_log = LOGS / f"mine_deepcad_{datetime.now().strftime('%m%d-%H%M')}.log"
                cmd = make_mine_cmd(
                    "./data/cadrille_training/deepcad",
                    DEEPCAD_OUTPUT,
                    DEEPCAD_MAX_SAMPLES)
                job_pid = launch(cmd, mine_log)
                log(f"DeepCAD mining started: PID={job_pid} log={mine_log}")

        elif state == State.MINING_FUSION360:
            f3_proc = count_processed(FUSION360_OUTPUT)
            f3_hard = count_hard(FUSION360_OUTPUT)
            log(f"[MINING_FUSION360] pid={job_pid} alive={pid_alive(job_pid)} "
                f"processed={f3_proc}/{FUSION360_MAX_SAMPLES} hard={f3_hard}")

            if f3_proc >= FUSION360_MAX_SAMPLES and not pid_alive(job_pid):
                log(f"Fusion360 mining complete: {f3_proc} scanned, {f3_hard} hard")
                state = State.MERGING
                job_pid = None

            elif not pid_alive(job_pid):
                log("Fusion360 mining not running → (re)starting with --resume")
                mine_log = LOGS / f"mine_fusion360_{datetime.now().strftime('%m%d-%H%M')}.log"
                cmd = make_mine_cmd(
                    "./data/cadrille_training/fusion360",
                    FUSION360_OUTPUT,
                    FUSION360_MAX_SAMPLES)
                job_pid = launch(cmd, mine_log)
                log(f"Fusion360 mining started: PID={job_pid} log={mine_log}")

        elif state == State.MERGING:
            log("[MERGING] Merging DeepCAD + Fusion360 hard examples...")
            n = do_merge()
            log(f"Merge done: {n} combined hard examples")
            log("Uploading combined_hard.pkl to HF...")
            hf_upload(COMBINED_OUTPUT, "combined_hard.pkl")
            patch_train_config_for_mined()
            state = State.TEMP_EVAL
            job_pid = None

        elif state == State.TEMP_EVAL:
            log(f"[TEMP_EVAL] pid={job_pid} alive={pid_alive(job_pid)}")
            if not pid_alive(job_pid):
                if job_pid is None:
                    # Start temperature sweep
                    ts = datetime.now().strftime("%m%d-%H%M")
                    teval_log = LOGS / f"eval_temperature_{ts}.log"
                    cmd = [
                        "python3", "tools/eval_temperature.py",
                        "--n-samples", "100",
                        "--K", "4",
                        "--temperatures",
                        "0.0", "0.15", "0.3", "0.45", "0.6",
                        "0.75", "0.9", "1.05", "1.2",
                        "--reward-workers", "8",
                    ]
                    job_pid = launch(cmd, teval_log)
                    log(f"Temperature sweep started: PID={job_pid} log={teval_log}")
                else:
                    # Sweep finished
                    log("Temperature sweep complete → starting training")
                    state = State.TRAINING
                    job_pid = None

        elif state == State.TRAINING:
            log(f"[TRAINING] pid={job_pid} alive={pid_alive(job_pid)}")

            if not pid_alive(job_pid):
                ts = datetime.now().strftime("%m%d-%H%M")
                train_run_name = f"cadrille-rl-run8-mined-{ts}"
                train_log = LOGS / f"rl-run8-{ts}.log"
                cmd = make_train_cmd(train_run_name)
                job_pid = launch(cmd, train_log)
                log(f"Training started: PID={job_pid} run={train_run_name} log={train_log}")

            else:
                # Check if training has hit step 1000 for eval
                train_log_glob = list(LOGS.glob("rl-run8-*.log"))
                if train_log_glob:
                    latest_log = max(train_log_glob, key=lambda p: p.stat().st_mtime)
                    try:
                        tail = subprocess.check_output(
                            ["tail", "-20", str(latest_log)]).decode()
                        # Extract step number from log
                        steps = re.findall(r'step[=\s]+(\d+)', tail, re.I)
                        if steps and int(steps[-1]) >= 1000 and eval_pid is None:
                            log("Training reached step 1000 → starting eval")
                            # Find latest checkpoint
                            ckpt_dir = ROOT / "checkpoints" / train_run_name
                            ckpts = sorted(ckpt_dir.glob("checkpoint-*")) if ckpt_dir.exists() else []
                            if ckpts:
                                eval_pid, eval_log = run_eval(str(ckpts[-1]))
                    except Exception as e:
                        log(f"Log check failed (non-fatal): {e}")

        elif state == State.EVAL:
            log(f"[EVAL] pid={eval_pid} alive={pid_alive(eval_pid)}")
            if not pid_alive(eval_pid):
                log("Eval complete. Pipeline DONE.")
                state = State.DONE

        elif state == State.DONE:
            log("[DONE] All pipeline stages complete. Daemon sleeping...")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # Ignore SIGHUP so nohup works
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
