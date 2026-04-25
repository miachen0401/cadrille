"""Hard example mining for RL fine-tuning of Cadrille.

Scans a directory of STL files, generates K completions from the SFT model,
keeps examples where mean(IoU) < R_th (model struggles → informative for RL).

Saves a pkl: list of {gt_mesh_path, file_name} dicts, compatible with
rl/dataset.py:RLDataset and rl/train.py --hard-examples-pkl.

Checkpoints every --checkpoint-every examples so the run can be interrupted
and resumed (--resume).

Paper values (img mode, IoU scale):
  K=3, R_th=0.75, max_new_tokens=400
  Expected yield: ~50k / 160k DeepCAD  →  ~30% hard rate (pc mode)
                  ~12k / 84k  DeepCAD  →  ~14% hard rate (img mode, IoU~86%)
                  ~7k  / 30k  Fusion360 → ~23% hard rate (img mode, IoU~77%)

On a single RTX 4080 with K=1 and max_new_tokens=400:
  ~6-8s per example  →  feasible overnight for 10k-20k samples.

Usage:
    # Mine DeepCAD train (up to 20k samples)
    python rl/mine.py \\
        --data-dir ./data/cadrille_training/deepcad \\
        --output ./data/mined/deepcad_hard.pkl \\
        --max-samples 20000 --K 1

    # Mine Fusion360 train (all ~30k STLs)
    python rl/mine.py \\
        --data-dir ./data/cadrille_training/fusion360 \\
        --output ./data/mined/fusion360_hard.pkl \\
        --K 1

    # Resume an interrupted run
    python rl/mine.py --data-dir ... --output ... --resume
"""

import os
import sys
import random
import pickle
import argparse
from glob import glob
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from common.model import Cadrille, collate
from common.meshio import MeshDataset
from common.metrics import compute_rewards_parallel


def parse_args():
    p = argparse.ArgumentParser(description="Mine hard examples for RL fine-tuning")
    p.add_argument("--checkpoint-path", default="./checkpoints/cadrille-sft")
    p.add_argument("--data-dir",   required=True, help="Directory of .stl files (recursive)")
    p.add_argument("--output",     required=True, help="Output pkl path")
    p.add_argument("--modality",   default="img", choices=["img", "pc"])
    p.add_argument("--K",          type=int,   default=1,    help="Completions per example")
    p.add_argument("--R-th",       type=float, default=0.75, help="Keep if mean(reward) < R_th (raw IoU scale 0-1)")
    p.add_argument("--max-samples",type=int,   default=None, help="Random subset of STLs to scan")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--reward-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature. 0 = greedy (default). Only used when K>1.")
    p.add_argument("--batch-size",  type=int, default=4,
                   help="Examples per generate() call. Larger = better GPU utilization (default 4)")
    p.add_argument("--checkpoint-every", type=int, default=500, help="Save intermediate pkl every N examples")
    p.add_argument("--resume",     action="store_true", help="Skip already-processed STLs (uses output pkl to track)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model from {args.checkpoint_path} ...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left")
    model = Cadrille.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto")
    model.eval()

    # ── Collect STL paths ────────────────────────────────────────────────────
    all_stls = sorted(glob(os.path.join(args.data_dir, "**", "*.stl"), recursive=True))
    if args.max_samples and len(all_stls) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(all_stls)
        all_stls = all_stls[:args.max_samples]
    print(f"STLs to scan: {len(all_stls)}")

    # ── Resume: skip already-processed paths ────────────────────────────────
    hard_examples = []
    processed_paths = set()
    scores_path = args.output.replace(".pkl", "_scores.jsonl")
    if args.resume:
        # Load processed paths regardless of whether the pkl exists yet
        ckpt_path = args.output + ".processed"
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                processed_paths = set(l.strip() for l in f)
        # Load hard examples if checkpoint pkl exists
        if os.path.exists(args.output):
            with open(args.output, "rb") as f:
                hard_examples = pickle.load(f)
        print(f"  Resumed: {len(hard_examples)} hard examples, {len(processed_paths)} already processed")

    todo = [p for p in all_stls if p not in processed_paths]
    print(f"  Remaining: {len(todo)} STLs to process")
    print(f"  Scores table: {scores_path}")

    # ── Mine (batched, serial) ────────────────────────────────────────────────
    # Batch B items per generate() call so GPU is busy more continuously.
    # Rendering is parallelised with ThreadPoolExecutor; rewards run serially
    # after generate to avoid CPU/CUDA contention on consumer GPUs.
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if args.modality == "img":
        from rl.dataset import render_img
    else:
        import trimesh
        from rl.dataset import mesh_to_point_cloud

    B  = args.batch_size   # examples per generate() call
    n_processed = 0
    ckpt_file   = open(args.output + ".processed", "a")
    scores_file = open(scores_path, "a")

    collate_fn = partial(collate, processor=processor, n_points=256, eval=True)

    def prepare_item(stl_path):
        """Render / load one STL → item dict, or None on failure."""
        file_name = os.path.splitext(os.path.basename(stl_path))[0]
        try:
            if args.modality == "img":
                item = render_img(stl_path)
                item.update({"description": "Generate cadquery code",
                              "file_name": file_name})
            else:
                mesh = trimesh.load(stl_path)
                pc   = mesh_to_point_cloud(mesh, 256)
                pc   = (pc - 0.5) * 2
                item = {"point_cloud": pc,
                        "description": "Generate cadquery code",
                        "file_name": file_name}
            return stl_path, item
        except Exception:
            return stl_path, None

    pbar = tqdm(total=len(todo),
                desc=f"Mining ({args.modality}, K={args.K}, B={B}, R_th={args.R_th})")

    with ThreadPoolExecutor(max_workers=min(B, 4)) as pool:
        i = 0
        while i < len(todo):
            chunk_paths = todo[i:i + B]
            i += B

            # Render chunk in parallel
            futures = {pool.submit(prepare_item, p): p for p in chunk_paths}
            items_ok = []   # (stl_path, item)
            for fut in as_completed(futures):
                stl_path, item = fut.result()
                if item is not None:
                    items_ok.append((stl_path, item))

            if not items_ok:
                pbar.update(len(chunk_paths))
                continue

            # Collate batch
            try:
                batch = collate_fn([it for _, it in items_ok])
            except Exception:
                pbar.update(len(chunk_paths))
                continue

            # If K > 1 repeat each item K times along batch dim
            if args.K > 1:
                g_batch = {
                    k: v.repeat_interleave(args.K, dim=0)
                       if isinstance(v, torch.Tensor) else [x for x in v for _ in range(args.K)]
                    for k, v in batch.items() if k != "file_name"
                }
            else:
                g_batch = {k: v for k, v in batch.items() if k != "file_name"}

            # Batched generate
            with torch.no_grad():
                try:
                    generated_ids = model.generate(
                        input_ids=g_batch["input_ids"].to(model.device),
                        attention_mask=g_batch["attention_mask"].to(model.device),
                        point_clouds=g_batch["point_clouds"].to(model.device),
                        is_pc=g_batch["is_pc"].to(model.device),
                        is_img=g_batch["is_img"].to(model.device),
                        pixel_values_videos=(
                            g_batch["pixel_values_videos"].to(model.device)
                            if g_batch.get("pixel_values_videos") is not None else None),
                        video_grid_thw=(
                            g_batch["video_grid_thw"].to(model.device)
                            if g_batch.get("video_grid_thw") is not None else None),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(args.K > 1 and args.temperature > 0),
                        temperature=args.temperature if args.temperature > 0 else 1.0,
                        bad_words_ids=[[model.config.video_token_id]],
                    )
                except Exception:
                    pbar.update(len(chunk_paths))
                    continue

            # Decode — prompt lengths may differ per item due to padding;
            # use each item's original input length to slice completions.
            prompt_lens = batch["input_ids"].shape[1]  # all same after left-pad collate
            all_completions = processor.batch_decode(
                generated_ids[:, prompt_lens:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            # Compute rewards after generate (serial avoids CPU/CUDA contention)
            all_stl_paths = [p for p, _ in items_ok for _ in range(args.K)]
            all_rewards_flat = compute_rewards_parallel(
                all_completions, all_stl_paths,
                workers=args.reward_workers,
                timeout=30.0)

            # Write results per original example
            for idx, (stl_path, _) in enumerate(items_ok):
                file_name = os.path.splitext(os.path.basename(stl_path))[0]
                rewards = all_rewards_flat[idx * args.K:(idx + 1) * args.K]
                mean_reward = float(np.mean(rewards))
                is_hard = mean_reward < args.R_th

                if is_hard:
                    hard_examples.append({"gt_mesh_path": stl_path,
                                          "file_name": file_name,
                                          "is_pc": (args.modality == "pc")})

                record = {
                    "gt_mesh_path": stl_path,
                    "file_name":    file_name,
                    "mean_reward":  round(mean_reward, 4),
                    "rewards":      [round(r, 4) for r in rewards],
                    "is_hard":      is_hard,
                }
                scores_file.write(_json.dumps(record) + "\n")
                ckpt_file.write(stl_path + "\n")
                n_processed += 1

            scores_file.flush()
            ckpt_file.flush()
            pbar.update(len(chunk_paths))

            # Periodic checkpoint
            if n_processed // args.checkpoint_every > (n_processed - len(items_ok)) // args.checkpoint_every:
                with open(args.output, "wb") as f:
                    pickle.dump(hard_examples, f)
                hard_rate = 100 * len(hard_examples) / max(n_processed, 1)
                pbar.write(f"  [{n_processed}/{len(todo)}] hard={len(hard_examples)} "
                           f"({hard_rate:.1f}%) — checkpoint saved")

    pbar.close()
    ckpt_file.close()
    scores_file.close()

    # Final save
    with open(args.output, "wb") as f:
        pickle.dump(hard_examples, f)

    print(f"\nDone. Scanned {n_processed} examples → {len(hard_examples)} hard "
          f"({100*len(hard_examples)/max(n_processed,1):.1f}%)")
    print(f"Hard pkl  → {args.output}")
    print(f"Scores    → {scores_path}")
    print(f"\nTo re-filter by a different threshold:")
    print(f"  python3 rl/filter_scores.py {scores_path} --R-th 0.6 --output custom_hard.pkl")


if __name__ == "__main__":
    main()
