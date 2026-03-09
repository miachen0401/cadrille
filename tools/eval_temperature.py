"""eval_temperature.py — Best-IoU and Pass@k vs. sampling temperature.

For each temperature in a sweep, generate K completions per example from the
SFT (or any) checkpoint, compute IoU, then plot:
  - mean best-IoU@K  (max IoU across K completions)
  - pass@K           (fraction of examples where ≥1 completion has IoU > iou_th)

Useful for choosing the right sampling temperature for RL rollouts.

Usage
-----
    python3 tools/eval_temperature.py                         # quick run, 50 samples
    python3 tools/eval_temperature.py --n-samples 200 --K 4
    python3 tools/eval_temperature.py --checkpoint ./checkpoints/cadrille-rl-run8/checkpoint-1000

Output
------
    work_dirs/eval_temperature/
        results.jsonl          — raw per-example per-temperature data
        temperature_sweep.png  — combined plot
"""

import argparse, json, os, random, sys
from pathlib import Path
from glob import glob
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from rl.dataset import render_img
from rl.reward import compute_rewards_parallel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./checkpoints/cadrille-sft")
    p.add_argument("--data-dir",   default="./data/deepcad_test_mesh")
    p.add_argument("--n-samples",  type=int, default=50)
    p.add_argument("--K",          type=int, default=4,
                   help="Completions per example per temperature")
    p.add_argument("--temperatures", nargs="+", type=float,
                   default=[0.0, 0.3, 0.5, 0.7, 1.0, 1.2],
                   help="Temperature values to sweep")
    p.add_argument("--iou-th",     type=float, default=0.75,
                   help="IoU threshold for pass@K (default 0.75)")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--reward-workers", type=int, default=8)
    p.add_argument("--out-dir",    default="./work_dirs/eval_temperature")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left")
    model = Cadrille.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto")
    model.eval()

    collate_fn = partial(collate, processor=processor, n_points=256, eval=True)

    # Sample STLs
    stls = sorted(glob(os.path.join(args.data_dir, "**", "*.stl"), recursive=True))
    rng  = random.Random(args.seed)
    rng.shuffle(stls)
    stls = stls[:args.n_samples]
    print(f"Evaluating {len(stls)} STLs × {args.K} completions × "
          f"{len(args.temperatures)} temperatures")

    # Pre-render all images once (shared across temperatures)
    print("Pre-rendering images ...")
    items = {}   # stl_path → item dict (or None)
    def _render(p):
        fn = os.path.splitext(os.path.basename(p))[0]
        try:
            it = render_img(p)
            it.update({"description": "Generate cadquery code", "file_name": fn})
            return p, it
        except Exception:
            return p, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for p, it in tqdm(pool.map(_render, stls), total=len(stls)):
            items[p] = it

    valid_stls = [p for p in stls if items[p] is not None]
    print(f"Valid renders: {len(valid_stls)} / {len(stls)}")

    results_path = out / "results.jsonl"
    all_results = {}   # temp → {stl_path → [iou, ...]}

    # Load existing results to allow resume
    if results_path.exists():
        for line in open(results_path):
            r = json.loads(line)
            t = r["temperature"]
            if t not in all_results:
                all_results[t] = {}
            all_results[t][r["stl_path"]] = r["ious"]
        print(f"Resumed: {len(all_results)} temperatures already done")

    rf = open(results_path, "a")

    for temp in args.temperatures:
        if temp in all_results and len(all_results[temp]) >= len(valid_stls):
            print(f"  temp={temp:.1f} — already done, skipping")
            continue

        print(f"\n── temperature={temp:.1f} ──────────────────────────────")
        all_results[temp] = all_results.get(temp, {})
        done = set(all_results[temp].keys())
        todo = [p for p in valid_stls if p not in done]

        # Generate K completions per example in batches of 4
        B = 4
        pbar = tqdm(total=len(todo))
        i = 0
        while i < len(todo):
            chunk = todo[i:i + B]
            i += B

            batch_items = [items[p] for p in chunk]
            try:
                batch = collate_fn(batch_items)
            except Exception:
                pbar.update(len(chunk))
                continue

            # Repeat K times along batch dim for K completions per example
            if args.K > 1:
                g_batch = {
                    k: v.repeat_interleave(args.K, dim=0)
                       if isinstance(v, torch.Tensor)
                       else [x for x in v for _ in range(args.K)]
                    for k, v in batch.items() if k != "file_name"
                }
            else:
                g_batch = {k: v for k, v in batch.items() if k != "file_name"}

            do_sample = (temp > 0 and args.K > 1)
            with torch.no_grad():
                try:
                    gen_ids = model.generate(
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
                        do_sample=do_sample,
                        temperature=temp if do_sample else 1.0,
                        bad_words_ids=[[model.config.video_token_id]],
                    )
                except Exception:
                    pbar.update(len(chunk))
                    continue

            prompt_len = batch["input_ids"].shape[1]
            completions = processor.batch_decode(
                gen_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            stl_paths_flat = [p for p in chunk for _ in range(args.K)]
            ious_flat = compute_rewards_parallel(
                completions, stl_paths_flat,
                workers=args.reward_workers)

            for idx, stl_path in enumerate(chunk):
                ious = [max(0.0, r) for r in ious_flat[idx * args.K:(idx + 1) * args.K]]
                all_results[temp][stl_path] = ious
                rf.write(json.dumps({
                    "temperature": temp,
                    "stl_path": stl_path,
                    "ious": ious,
                }) + "\n")

            rf.flush()
            pbar.update(len(chunk))

        pbar.close()

        # Report per-temperature stats
        all_ious = [all_results[temp][p] for p in valid_stls if p in all_results[temp]]
        best_ious  = [max(x) for x in all_ious]
        mean_ious  = [float(np.mean(x)) for x in all_ious]
        pass_k     = sum(1 for x in all_ious if any(v >= args.iou_th for v in x)) / len(all_ious)
        print(f"  mean best-IoU@{args.K}: {np.mean(best_ious):.4f}  "
              f"pass@{args.K}(th={args.iou_th}): {pass_k:.3f}  "
              f"mean@1: {np.mean(mean_ious):.4f}")

    rf.close()

    # ── Plot ──────────────────────────────────────────────────────────────────
    temps_done = sorted(t for t in args.temperatures if t in all_results)
    mean_best, mean_avg, pass_vals = [], [], []

    for temp in temps_done:
        ious_per = [all_results[temp][p] for p in valid_stls if p in all_results[temp]]
        mean_best.append(np.mean([max(x) for x in ious_per]))
        mean_avg.append(np.mean([np.mean(x) for x in ious_per]))
        pass_vals.append(
            sum(1 for x in ious_per if any(v >= args.iou_th for v in x)) / len(ious_per))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"SFT model: K={args.K} completions per example  (n={len(valid_stls)})",
                 fontsize=13)

    ax1.plot(temps_done, mean_best,  "o-", color="steelblue",  label=f"best-IoU@{args.K}")
    ax1.plot(temps_done, mean_avg,   "s--", color="tomato",    label="mean-IoU@1 (avg completion)")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("IoU")
    ax1.set_title("IoU vs Temperature")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(temps_done, pass_vals, "^-", color="green",
             label=f"pass@{args.K}  (IoU ≥ {args.iou_th})")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(f"Fraction (pass@{args.K})")
    ax2.set_title(f"Pass@{args.K} vs Temperature  (IoU threshold = {args.iou_th})")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = out / "temperature_sweep.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved → {plot_path}")

    # Print summary table
    print(f"\n{'Temp':>6}  {'best-IoU@'+str(args.K):>12}  {'mean-IoU@1':>11}  {'pass@'+str(args.K):>8}")
    print("-" * 46)
    for t, b, a, p in zip(temps_done, mean_best, mean_avg, pass_vals):
        print(f"{t:>6.1f}  {b:>12.4f}  {a:>11.4f}  {p:>8.3f}")


if __name__ == "__main__":
    main()
