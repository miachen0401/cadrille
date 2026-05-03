"""Img-modality eval used by train/rl/eval.py:_run_img_eval_subprocess.

Generates greedy CadQuery code for N samples from each split, computes IoU
(and Chamfer distance) against the ground-truth STL, and writes a per-sample
results.csv that the parent process aggregates.

CLI matches what train/rl/eval.py invokes:
    python tools/eval_img.py \
        --checkpoint <dir>           # Cadrille ckpt (model.safetensors + configs)
        --splits     short:dir [short:dir ...]
        --n-samples  N               # samples per split (seed=42 stratified)
        --out-dir    <dir>           # writes <out-dir>/<short>/results.csv
        --batch-size B
        --max-new-tokens T
        --seed       S
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Load .env so HF_TOKEN/etc. are available
_env = REPO / '.env'
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

from common.model import collate, get_cadrille_class  # noqa: E402
from common.meshio import render_img  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402


def _detect_backbone_from_config(ckpt_dir: str) -> str:
    """Read config.json; map model_type to backbone name expected by get_cadrille_class."""
    import json
    cfg = json.loads((Path(ckpt_dir) / 'config.json').read_text())
    mt = cfg.get('model_type', '')
    if mt == 'qwen3_vl':
        return 'qwen3_vl'
    if mt == 'qwen2_5_vl':
        return 'qwen2_5_vl'
    return 'qwen2_vl'


def _base_model_for(backbone: str) -> str:
    return {
        'qwen3_vl':   'Qwen/Qwen3-VL-2B-Instruct',
        'qwen2_5_vl': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'qwen2_vl':   'Qwen/Qwen2-VL-2B-Instruct',
    }[backbone]


@torch.inference_mode()
def _generate_one_split(model, processor, split_dir: Path, n_samples: int,
                        batch_size: int, max_new_tokens: int, seed: int,
                        out_csv: Path) -> None:
    """Generate greedy code for n_samples STLs in split_dir; write IoU/CD per row."""
    import random as _r
    stls = sorted(f for f in os.listdir(split_dir) if f.endswith('.stl'))
    rng = _r.Random(seed)
    rng.shuffle(stls)
    stls = stls[:n_samples]
    if not stls:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text('file_name,iou,cd\n')
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fout = out_csv.open('w', newline='')
    writer = csv.writer(fout)
    writer.writerow(['file_name', 'iou', 'cd'])

    device = next(model.parameters()).device
    eos_id = model.config.eos_token_id
    img_token_id = getattr(model.config, 'image_token_id', None)
    vid_token_id = getattr(model.config, 'video_token_id', None)
    bad_words = [[t] for t in (img_token_id, vid_token_id) if t is not None]

    for batch_start in range(0, len(stls), batch_size):
        chunk = stls[batch_start: batch_start + batch_size]
        items = []
        for stl_name in chunk:
            stl_path = split_dir / stl_name
            try:
                img_item = render_img(str(stl_path))
            except Exception as e:
                print(f'[eval_img] render failed for {stl_name}: {e}', flush=True)
                writer.writerow([stl_name[:-4], '', ''])
                continue
            items.append({
                'video':       img_item['video'],
                'description': 'Generate cadquery code',
                'file_name':   stl_name[:-4],
                '_stl':        str(stl_path),
            })
        if not items:
            continue
        batch = collate(items, processor, n_points=256, eval=True)
        gen_kwargs = dict(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            point_clouds=batch['point_clouds'].to(device),
            is_pc=batch['is_pc'].to(device),
            is_img=batch['is_img'].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=eos_id,
            bad_words_ids=(bad_words or None),
        )
        for k in ('pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids'):
            v = batch.get(k)
            if v is not None:
                gen_kwargs[k] = v.to(device)
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None
        out_ids = model.generate(**gen_kwargs)
        prompt_len = batch['input_ids'].shape[1]
        for i, item in enumerate(items):
            code = processor.decode(out_ids[i, prompt_len:], skip_special_tokens=True)
            iou, cd = compute_metrics(code, item['_stl'], timeout=60)
            iou_w = '' if iou is None else float(iou)
            cd_w  = '' if cd  is None else float(cd)
            writer.writerow([item['file_name'], iou_w, cd_w])
            fout.flush()
    fout.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',     required=True, help='Cadrille ckpt dir')
    ap.add_argument('--splits',         required=True, nargs='+',
                    help='Each arg: short:dir (e.g. deepcad:/path/to/dir)')
    ap.add_argument('--n-samples',      type=int, default=50)
    ap.add_argument('--out-dir',        required=True)
    ap.add_argument('--batch-size',     type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=400)
    ap.add_argument('--seed',           type=int, default=42)
    ap.add_argument('--backbone',       default=None,
                    help='Auto-detected from ckpt config.json if omitted.')
    args = ap.parse_args()

    backbone = args.backbone or _detect_backbone_from_config(args.checkpoint)
    base_model = _base_model_for(backbone)

    print(f'[eval_img] backbone={backbone} base_model={base_model}', flush=True)
    processor = AutoProcessor.from_pretrained(
        base_model, token=os.environ.get('HF_TOKEN'),
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        padding_side='left',
    )
    cadrille_cls = get_cadrille_class(backbone)
    model = cadrille_cls.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ).eval().to('cuda')

    out_root = Path(args.out_dir)
    for spec in args.splits:
        if ':' not in spec:
            print(f'[eval_img] bad split spec {spec!r}, expected short:dir', flush=True)
            continue
        short, gt_dir = spec.split(':', 1)
        out_csv = out_root / short / 'results.csv'
        print(f'[eval_img] {short} ({args.n_samples} samples) → {out_csv}', flush=True)
        _generate_one_split(
            model, processor, Path(gt_dir), args.n_samples,
            args.batch_size, args.max_new_tokens, args.seed,
            out_csv,
        )

    del model
    torch.cuda.empty_cache()
    print('[eval_img] done', flush=True)


if __name__ == '__main__':
    main()
