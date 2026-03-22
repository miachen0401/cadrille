"""Download latest RL checkpoints from HF, skipping optimizer/rng states.

Saves to: checkpoints/<run_name>/checkpoint-NNN/
"""
from huggingface_hub import snapshot_download
from pathlib import Path

CKPTS = [
    ("rl-s3600-lr2e-5-G16-cppo-0320-0313", "checkpoint-90"),
    ("rl-s3600-lr1e-5-G16-cppo-0320-0524", "checkpoint-360"),
    ("rl-s3600-lr1e-5-G16-cppo-0320-0531", "checkpoint-360"),
]

for run_name, ckpt in CKPTS:
    prefix = f"{run_name}/{ckpt}/"
    out = Path(f"checkpoints/{run_name}/{ckpt}")

    # Quick check: skip if model.safetensors already downloaded
    if (out / "model.safetensors").exists():
        print(f"SKIP (exists): {run_name}/{ckpt}")
        continue

    print(f"\n=== Downloading {run_name}/{ckpt} ===", flush=True)
    snapshot_download(
        repo_id="Hula0401/cad_ckpt",
        local_dir="checkpoints",
        allow_patterns=[f"{prefix}*.json", f"{prefix}*.txt", f"{prefix}model.safetensors",
                        f"{prefix}*.model", f"{prefix}tokenizer.json", f"{prefix}vocab.json",
                        f"{prefix}merges.txt"],
        ignore_patterns=["optimizer.pt", "rng_state.pt", "scheduler.pt", "*.pt"],
        local_dir_use_symlinks=False,
    )
    print(f"Done: {run_name}/{ckpt}", flush=True)

print("\nAll done.", flush=True)
