"""Pass@k evaluation — thin wrapper around rl/eval_passk.py's core logic.

Reuses the unbiased estimator and batched generation from eval_passk.py,
but plugs into the unified EvalConfig system.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


# ── unbiased estimator (Chen et al. 2021) ─────────────────────────────────────

def _pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def pass_at_k_mean(n_list: list[int], c_list: list[int], k: int) -> float:
    vals = [_pass_at_k(n, c, k) for n, c in zip(n_list, c_list)]
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ── main pass@k runner ────────────────────────────────────────────────────────

def run_passk(
    model,
    processor,
    stl_paths: list[str],
    modality: str,
    out_dir: Path,
    n_samples: int = 8,
    k_values: list[int] = (1, 5),
    iou_threshold: float = 0.95,
    temperature: float = 0.8,
    batch_size: int = 16,
    max_new_tokens: int = 768,
    score_workers: int = 8,
) -> dict:
    """Run pass@k evaluation for one (model, dataset, modality) combo.

    Delegates to rl.eval_passk.eval_passk with the right example format,
    then saves results to out_dir/passk.json.

    Returns pass@k result dict.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from rl.eval_passk import eval_passk, load_val_examples as _load_passk_examples

    # Build example list from stl_paths
    examples = _build_examples(stl_paths, modality)
    if not examples:
        print(f"  [pass@k] No examples for {modality} — skipping")
        return {}

    print(f"  [pass@k] {modality} n={len(examples)} n_samples={n_samples} "
          f"k={list(k_values)} threshold={iou_threshold}", flush=True)

    results = eval_passk(
        model=model,
        processor=processor,
        examples=examples,
        n_samples=n_samples,
        k_values=list(k_values),
        threshold=iou_threshold,
        max_new_tokens=max_new_tokens,
        eval_batch_size=batch_size,
        reward_workers=score_workers,
        temperature=temperature,
        sequential=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "passk.json"
    save_data = {k: v for k, v in results.items() if k != "wandb_metrics"}
    out_file.write_text(json.dumps(save_data, indent=2))
    print(f"  [pass@k] Saved → {out_file}")

    return results


def _build_examples(stl_paths: list[str], modality: str) -> list[dict]:
    """Build example dicts for eval_passk (pc or img mode)."""
    import sys
    import trimesh
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset import mesh_to_point_cloud
    from rl.dataset import render_img

    examples = []
    for gt_path in stl_paths:
        stem = Path(gt_path).stem
        try:
            if modality == "pc":
                mesh = trimesh.load(gt_path)
                pc = mesh_to_point_cloud(mesh, 256)
                pc = (pc - 0.5) * 2
                examples.append({
                    "point_cloud": pc,
                    "description": "Generate cadquery code",
                    "file_name": stem,
                    "gt_mesh_path": gt_path,
                    "_modality": "pc",
                })
            else:  # img
                render_result = render_img(gt_path)
                examples.append({
                    "video": render_result["video"],
                    "description": "Generate cadquery code",
                    "file_name": stem,
                    "gt_mesh_path": gt_path,
                    "_modality": "img",
                })
        except Exception:
            continue

    return examples
