"""Quick eval of new H100 RL checkpoints.

Runs pc-mode eval (eval_one_pass) + a 5-sample img mode garbled-output probe
for each checkpoint. Reports IoU, failure rate, and raw output snippets.

Usage:
    python3 tools/eval_new_ckpts.py                   # eval all 3 new ckpts
    python3 tools/eval_new_ckpts.py --n 300            # larger sample
    python3 tools/eval_new_ckpts.py --ckpt checkpoints/rl-s3600-lr1e-5-G16-cppo-0320-0524/checkpoint-360
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from rl.eval import eval_one_pass, load_val_examples

# ── checkpoint list ───────────────────────────────────────────────────────────
CKPTS = [
    ("rl-s3600-lr2e-5-G16-cppo-0320-0313", "checkpoint-90"),
    ("rl-s3600-lr1e-5-G16-cppo-0320-0524", "checkpoint-360"),
    ("rl-s3600-lr1e-5-G16-cppo-0320-0531", "checkpoint-360"),
]

DATASETS = {
    "deepcad": "data/deepcad_test_mesh",
    "fusion360": "data/fusion360_test_mesh",
}


def garbled_probe(model, processor, ds_dir: str, n: int = 5) -> list[str]:
    """Generate code for n img-mode examples; return raw output strings."""
    examples = load_val_examples(ds_dir, n, modalities=("img",))
    if not examples:
        return []

    device = next(model.parameters()).device
    if hasattr(model, "rope_deltas"):
        model.rope_deltas = None

    collate_items = [{k: v for k, v in ex.items() if not k.startswith("_")}
                     for ex in examples]
    batch = collate(collate_items, processor=processor, n_points=256, eval=True)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            point_clouds=batch["point_clouds"].to(device),
            is_pc=batch["is_pc"].to(device),
            is_img=batch["is_img"].to(device),
            pixel_values_videos=(
                batch["pixel_values_videos"].to(device)
                if batch.get("pixel_values_videos") is not None else None),
            video_grid_thw=(
                batch["video_grid_thw"].to(device)
                if batch.get("video_grid_thw") is not None else None),
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            bad_words_ids=[[model.config.video_token_id]],
        )
    prompt_len = batch["input_ids"].shape[1]
    texts = [
        processor.decode(generated_ids[j, prompt_len:], skip_special_tokens=True)
        for j in range(len(examples))
    ]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="*", help="Specific checkpoint dirs")
    parser.add_argument("--n", type=int, default=100, help="PC eval samples per dataset")
    parser.add_argument("--probe-n", type=int, default=5, help="Img garbled probe samples")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--reward-workers", type=int, default=4)
    args = parser.parse_args()

    # resolve checkpoints
    if args.ckpt:
        ckpt_paths = [Path(c) for c in args.ckpt]
    else:
        ckpt_paths = []
        for run_name, ckpt_name in CKPTS:
            p = Path(f"checkpoints/{run_name}/{ckpt_name}")
            if (p / "model.safetensors").exists():
                ckpt_paths.append(p)
            else:
                print(f"SKIP (not downloaded): {p}")

    if not ckpt_paths:
        print("No checkpoints available. Download first.")
        sys.exit(1)

    # processor (always from base model)
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
    )

    # load val examples (pc) once per dataset
    print("Loading val examples (pc mode) ...", flush=True)
    val_pc = {}
    for ds, ds_dir in DATASETS.items():
        val_pc[ds] = load_val_examples(ds_dir, args.n, modalities=("pc",))
        print(f"  {ds}/pc: {len(val_pc[ds])} examples")

    all_results = {}

    for ckpt_path in ckpt_paths:
        run_label = f"{ckpt_path.parent.name}/{ckpt_path.name}"
        print(f"\n{'='*64}")
        print(f"  Checkpoint: {run_label}")
        print(f"{'='*64}", flush=True)

        model = Cadrille.from_pretrained(
            str(ckpt_path), torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()

        run_res = {}

        # ── 1. IMG garbled probe ──────────────────────────────────────────────
        print(f"\n[img probe — {args.probe_n} samples from deepcad]", flush=True)
        texts = garbled_probe(model, processor, DATASETS["deepcad"], n=args.probe_n)
        kw = ("import cadquery", "cq.", ".sketch(", ".extrude(", ".box(", "result")
        for i, t in enumerate(texts):
            has_kw = any(k in t for k in kw)
            snippet = t[:120].replace("\n", "↵")
            print(f"  [{i}] {'OK' if has_kw else 'GARBLED'}: {snippet!r}")

        # ── 2. PC eval (eval_one_pass) ────────────────────────────────────────
        print(f"\n[pc eval — n={args.n} per dataset]", flush=True)

        class _Args:
            max_new_tokens = args.max_new_tokens
            eval_batch_size = args.eval_batch_size
            eval_workers = args.reward_workers
            eval_timeout = 120.0

        all_pc_examples = []
        for ds, examples in val_pc.items():
            all_pc_examples.extend(examples)

        metrics = eval_one_pass(
            model=model,
            examples=all_pc_examples,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            reward_workers=args.reward_workers,
            eval_timeout=120.0,
        )
        run_res["pc_metrics"] = metrics
        run_res["img_garbled_rate"] = sum(
            1 for t in texts if not any(k in t for k in kw)
        ) / max(len(texts), 1)

        all_results[run_label] = run_res

        del model
        torch.cuda.empty_cache()

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("SUMMARY (pc mode IoU, N={})".format(args.n))
    print(f"{'='*64}")
    for run_label, res in all_results.items():
        m = res["pc_metrics"]
        dc_iou  = m.get("eval/pc/DeepCAD test/IoU mean", float("nan"))
        f360_iou = m.get("eval/pc/Fusion360 test/IoU mean", float("nan"))
        dc_fail  = m.get("eval/pc/DeepCAD test/Failures fraction", float("nan"))
        img_gr   = res["img_garbled_rate"]
        print(f"  {run_label}")
        print(f"    pc/DeepCAD IoU={dc_iou:.4f}  pc/Fusion360 IoU={f360_iou:.4f}  "
              f"dc_fail={dc_fail*100:.1f}%  img_garbled={img_gr*100:.0f}%")

    # save
    out = Path("eval_outputs/new_ckpts_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
