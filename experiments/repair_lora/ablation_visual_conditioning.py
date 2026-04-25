"""Visual conditioning ablation for repair LoRA.

Tests whether the fine-tuned model actually uses visual input by comparing
three inference conditions:
  C1: text only (bad code + action, NO images)
  C2: GT views only (no pred render)
  C3: GT views + pred render (full input, same as eval_repair_lora)

If C1 ≈ C3 → visual input is NOT being used.
If C3 >> C1 → visual conditioning is working.

Usage
-----
  python3 tools/ablation_visual_conditioning.py
  python3 tools/ablation_visual_conditioning.py --n 50
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from common.model import Cadrille, collate
from experiments.repair_lora.repair_feasibility import is_wrong_primitive, score_code, _hstack

REPAIR_PROMPT = (
    "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).\n"
    "Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced with a proper "
    "sketch+extrude pattern matching the target geometry.\n"
    "Rewrite the code using sketch+extrude.\n\n"
    "Broken code:\n{code}"
)

PROMPT_NO_RENDER = (
    "The target 3D shape is shown. The code below uses box() instead of sketch+extrude.\n"
    "Repair action: SWITCH_TO_SKETCH_EXTRUDE — replace the box() with a proper "
    "sketch+extrude pattern matching the target geometry.\n"
    "Rewrite the code using sketch+extrude.\n\n"
    "Broken code:\n{code}"
)

PROMPT_TEXT_ONLY = (
    "The code below uses box() instead of sketch+extrude.\n"
    "Repair action: SWITCH_TO_SKETCH_EXTRUDE — replace the box() with a proper "
    "sketch+extrude pattern.\n"
    "Rewrite the code using sketch+extrude.\n\n"
    "Broken code:\n{code}"
)


def build_items(condition, gt_img, pred_img, bad_code):
    """Build item dict for each condition."""
    if condition == 'C1':
        # Text only — no video
        return {
            'description': PROMPT_TEXT_ONLY.format(code=bad_code),
            'file_name': 'repair'
        }
    elif condition == 'C2':
        # GT views only
        return {
            'video': [gt_img],
            'description': PROMPT_NO_RENDER.format(code=bad_code),
            'file_name': 'repair'
        }
    else:  # C3
        # GT + pred render hstacked
        return {
            'video': [_hstack(gt_img, pred_img)],
            'description': REPAIR_PROMPT.format(code=bad_code),
            'file_name': 'repair'
        }


@torch.no_grad()
def infer_batch(model, processor, items, max_new_tokens=512):
    batch = collate(items, processor=processor, n_points=256, eval=True)
    gen_ids = model.generate(
        input_ids=batch['input_ids'].to(model.device),
        attention_mask=batch['attention_mask'].to(model.device),
        point_clouds=batch['point_clouds'].to(model.device),
        is_pc=batch['is_pc'].to(model.device),
        is_img=batch['is_img'].to(model.device),
        pixel_values_videos=(batch['pixel_values_videos'].to(model.device)
                              if batch.get('pixel_values_videos') is not None else None),
        video_grid_thw=(batch['video_grid_thw'].to(model.device)
                        if batch.get('video_grid_thw') is not None else None),
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        bad_words_ids=[[model.config.video_token_id]],
    )
    prompt_len = batch['input_ids'].shape[1]
    return [processor.decode(gen_ids[j, prompt_len:], skip_special_tokens=True)
            for j in range(len(items))]


def fix_trailing_paren(code):
    """Strip trailing extra ) if it causes SyntaxError."""
    import ast
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        stripped = code.rstrip()
        for _ in range(5):
            if stripped.endswith(')'):
                stripped = stripped[:-1]
                try:
                    ast.parse(stripped)
                    return stripped
                except SyntaxError:
                    continue
    return None


def score_codes(codes, selected, n_workers=4):
    """Score a list of codes against GT STLs. Returns list of dicts."""
    fixed = [fix_trailing_paren(c) for c in codes]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [
            pool.submit(score_code, f, str(c['gt_stl'])) if f is not None else None
            for f, c in zip(fixed, selected)
        ]
    results = []
    for fut in futs:
        if fut is None:
            results.append({'iou': None, 'cd': None})
        else:
            results.append(fut.result())
    return results


def report(cond, results, selected, label):
    base_ious = [c['iou'] for c in selected]
    n = len(selected)
    valid = sum(1 for r in results if r['iou'] is not None)
    deltas = [r['iou'] - c['iou'] for r, c in zip(results, selected)
              if r['iou'] is not None]
    meaningful = sum(1 for d in deltas if d > 0.05)
    print(f'\n{label}')
    print(f'  valid rate:      {valid}/{n} = {valid/n*100:.1f}%')
    if deltas:
        arr = np.array(deltas)
        print(f'  mean  ΔIoU:      {arr.mean():+.4f}')
        print(f'  median ΔIoU:     {np.median(arr):+.4f}')
        print(f'  % ΔIoU > 0.05:   {meaningful/n*100:.1f}%  ({meaningful}/{n})')
        print(f'  ΔIoU dist:  <-0.05:{(arr<-0.05).mean()*100:.0f}%  '
              f'[-0.05,0):{((arr>=-0.05)&(arr<0)).mean()*100:.0f}%  '
              f'[0,0.05):{((arr>=0)&(arr<0.05)).mean()*100:.0f}%  '
              f'≥0.05:{(arr>=0.05).mean()*100:.0f}%')
    return {
        'condition': cond,
        'valid_rate': valid / n,
        'mean_delta_iou': float(np.mean(deltas)) if deltas else None,
        'median_delta_iou': float(np.median(deltas)) if deltas else None,
        'pct_meaningful': meaningful / n,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',    default='checkpoints/repair-lora/best')
    parser.add_argument('--base-model',    default='checkpoints/cadrille-rl')
    parser.add_argument('--analysis-dir',  default='data/analysis/deepcad_rl_img')
    parser.add_argument('--gt-dir',        default='data/deepcad_test_mesh')
    parser.add_argument('--n',             type=int, default=100)
    parser.add_argument('--batch-size',    type=int, default=4)
    parser.add_argument('--score-workers', type=int, default=4)
    parser.add_argument('--conditions',    default='C1,C2,C3')
    parser.add_argument('--out',           default='data/repair_ablation')
    args = parser.parse_args()

    conditions = args.conditions.split(',')
    analysis_dir = _REPO / args.analysis_dir
    gt_dir       = _REPO / args.gt_dir
    out_dir      = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select wrong_primitive cases (same as eval_repair_lora)
    print('Selecting wrong_primitive cases...')
    with open(analysis_dir / 'metadata.jsonl') as f:
        meta = [json.loads(l) for l in f]

    candidates = []
    for row in meta:
        if row.get('iou') is None:
            continue
        iou = float(row['iou'])
        if not (0.30 <= iou <= 0.88):
            continue
        stem = row['case_id']
        py_path     = analysis_dir / f'{stem}_pred.py'
        pred_render = analysis_dir / f'{stem}_pred_render.png'
        gt_render   = gt_dir / f'{stem}_render.png'
        gt_stl      = gt_dir / f'{stem}.stl'
        if not all(p.exists() for p in [py_path, pred_render, gt_render, gt_stl]):
            continue
        code = py_path.read_text()
        if not is_wrong_primitive(code):
            continue
        candidates.append({'stem': stem, 'iou': iou, 'code': code,
                            'pred_render': pred_render, 'gt_render': gt_render,
                            'gt_stl': gt_stl})

    candidates.sort(key=lambda x: abs(x['iou'] - 0.6))
    selected = candidates[:args.n]
    print(f'  {len(selected)} cases  IoU [{min(c["iou"] for c in selected):.3f}, '
          f'{max(c["iou"] for c in selected):.3f}]')

    # Load model
    print(f'\nLoading base model {args.base_model}...')
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28, padding_side='left')
    model = Cadrille.from_pretrained(
        str(_REPO / args.base_model), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')

    print(f'Loading LoRA from {args.checkpoint}...')
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(_REPO / args.checkpoint))
    model = model.merge_and_unload()
    model.eval()
    print('  Model ready.')

    # Load images
    print('\nLoading renders...')
    for c in selected:
        c['gt_img']   = Image.open(c['gt_render']).convert('RGB')
        c['pred_img'] = Image.open(c['pred_render']).convert('RGB')

    # Run each condition
    all_results = {}
    summaries = []

    for cond in conditions:
        print(f'\n{"="*50}')
        print(f'Condition {cond}...')
        items = [build_items(cond, c['gt_img'], c['pred_img'], c['code'])
                 for c in selected]

        codes = []
        for i in tqdm(range(0, len(items), args.batch_size), desc=f'{cond} infer'):
            codes.extend(infer_batch(model, processor, items[i:i + args.batch_size]))

        # Save codes
        cond_dir = out_dir / cond
        cond_dir.mkdir(exist_ok=True)
        for c, code in zip(selected, codes):
            (cond_dir / f'{c["stem"]}.py').write_text(code)

        # Score
        print(f'Scoring {cond}...')
        results = score_codes(codes, selected, n_workers=args.score_workers)
        all_results[cond] = results

        label = {
            'C1': 'C1 — text only (no images)',
            'C2': 'C2 — GT views only (no pred render)',
            'C3': 'C3 — GT views + pred render (full input)',
        }.get(cond, cond)
        summaries.append(report(cond, results, selected, label))

    # Final comparison
    print('\n' + '=' * 60)
    print('VISUAL CONDITIONING ABLATION — SUMMARY')
    print('=' * 60)
    print(f'Baseline (bad code): mean IoU={np.mean([c["iou"] for c in selected]):.4f}')
    for s in summaries:
        print(f'  {s["condition"]}: valid={s["valid_rate"]*100:.0f}%  '
              f'ΔIoU={s["mean_delta_iou"]:+.4f}  '
              f'pct>0.05={s["pct_meaningful"]*100:.0f}%')

    if 'C1' in all_results and 'C3' in all_results:
        d1 = [r['iou'] for r in all_results['C1'] if r['iou'] is not None]
        d3 = [r['iou'] for r in all_results['C3'] if r['iou'] is not None]
        if d1 and d3:
            diff = np.mean(d3) - np.mean(d1)
            print(f'\n  C3 vs C1 mean IoU gap: {diff:+.4f}')
            if abs(diff) < 0.02:
                print('  ⚠ VISUAL INPUT NOT BEING USED (C1 ≈ C3)')
            elif diff > 0.05:
                print('  ✓ Visual conditioning IS working (C3 >> C1)')
            else:
                print('  ~ Weak visual conditioning signal')

    (out_dir / 'ablation_results.json').write_text(json.dumps({
        'n': len(selected),
        'baseline_mean_iou': float(np.mean([c['iou'] for c in selected])),
        'summaries': summaries,
    }, indent=2))
    print(f'\nSaved to {out_dir}/ablation_results.json')


if __name__ == '__main__':
    main()
