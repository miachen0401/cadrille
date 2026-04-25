"""Verify torch + CUDA + pytorch3d + flash-attn + bf16 are all usable.

Run: uv run python tools/check_env/check_torch.py
Exits 0 on success, non-zero on any failure.
"""
from __future__ import annotations
import sys


def main() -> int:
    failed: list[str] = []

    try:
        import torch
        print(f"torch {torch.__version__}")
        assert torch.cuda.is_available(), "CUDA not available"
        n = torch.cuda.device_count()
        assert n > 0, "no CUDA devices"
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB  sm_{p.major}{p.minor}")
        assert torch.cuda.is_bf16_supported(), "bf16 not supported on this GPU"
        print("  bf16 supported: True")
    except Exception as e:
        failed.append(f"torch: {e}")

    try:
        import pytorch3d
        from pytorch3d import _C  # noqa — CUDA ext must load
        print(f"pytorch3d {pytorch3d.__version__}  (_C loaded)")
    except Exception as e:
        failed.append(f"pytorch3d: {e}")

    try:
        import flash_attn
        from flash_attn import flash_attn_func  # noqa
        print(f"flash-attn {flash_attn.__version__}")
    except Exception as e:
        failed.append(f"flash-attn: {e}")

    try:
        import transformers
        import accelerate
        print(f"transformers {transformers.__version__}  accelerate {accelerate.__version__}")
    except Exception as e:
        failed.append(f"transformers/accelerate: {e}")

    if failed:
        print("\nFAIL")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
