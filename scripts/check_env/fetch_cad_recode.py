"""Download cad-recode-v1.5 from HuggingFace with throttling to avoid the 5000 req/5min rate limit.

Resumable: already-downloaded files are skipped by snapshot_download.

Run: set -a; source .env; set +a; uv run python tools/check_env/fetch_cad_recode.py
Optional flags: --train-batches N  (default: all batches if not specified; else N batches)
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cad-recode-v1.5")
    ap.add_argument("--train-batches", type=int, default=None,
                    help="If set, only pull train/batch_00..batch_{N-1}. Default: all.")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="HF rate-limit workaround: 2 is safe.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — anonymous access may hit rate limits")

    from huggingface_hub import snapshot_download

    patterns = ["val/*.py"]
    if args.train_batches is None:
        patterns.append("train/**/*.py")
    else:
        for i in range(args.train_batches):
            patterns.append(f"train/batch_{i:02d}/*.py")

    print(f"Downloading filapro/cad-recode-v1.5 -> {args.out}")
    print(f"  patterns: {patterns}")
    print(f"  max_workers: {args.max_workers}")

    snapshot_download(
        "filapro/cad-recode-v1.5",
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
        max_workers=args.max_workers,
        token=token,
    )

    out = Path(args.out)
    n_train = sum(1 for _ in out.glob("train/**/*.py"))
    n_val = sum(1 for _ in out.glob("val/*.py"))
    print(f"\nDone. train .py: {n_train}, val .py: {n_val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
