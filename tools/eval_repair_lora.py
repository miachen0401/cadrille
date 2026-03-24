"""Evaluate repair LoRA on real wrong_primitive failures (Step 2 transfer test).

Loads the LoRA checkpoint, runs repair inference on the 100 wrong_primitive
cases from the feasibility study, and reports ΔIoU / ΔCD vs baseline.

Usage
-----
  python3 tools/eval_repair_lora.py
  python3 tools/eval_repair_lora.py --checkpoint checkpoints/repair-lora/best
  python3 tools/eval_repair_lora.py --n 50
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from cadrille import Cadrille, collate
from tools.repair_feasibility import (
    is_wrong_primitive, score_code, PROMPTS, _hstack
)


EVAL_PROMPTS = {
    'hstack': (
        "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).\n"
        "Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced with a proper "
        "sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
    '2frame': (
        "First image: target 3D shape (4 views). Second image: current broken prediction (4 views).\n"
        "Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced with a proper "
        "sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
    'gt-only': (
        "The target 3D shape is shown.\n"
        "Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced with a proper "
        "sketch+extrude pattern matching the target geometry.\n"
        "Rewrite the code using sketch+extrude.\n\n"
        "Broken code:\n{code}"
    ),
}


def build_item(gt_img, pred_img, bad_code, input_mode='hstack'):
    desc = EVAL_PROMPTS[input_mode].format(code=bad_code)
    if input_mode == 'hstack':
        video = [_hstack(gt_img, pred_img)]
    elif input_mode == '2frame':
        video = [gt_img, pred_img]
    else:  # gt-only
        video = [gt_img]
    return {'video': video, 'description': desc, 'file_name': 'repair'}


@torch.no_grad()
def infer_batch(model, processor, items, max_new_tokens=1024):
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',   default='checkpoints/repair-lora/best')
    parser.add_argument('--base-model',   default='checkpoints/cadrille-rl')
    parser.add_argument('--analysis-dir', default='data/analysis/deepcad_rl_img')
    parser.add_argument('--gt-dir',       default='data/deepcad_test_mesh')
    parser.add_argument('--n',            type=int, default=100)
    parser.add_argument('--batch-size',   type=int, default=4)
    parser.add_argument('--score-workers',type=int, default=4)
    parser.add_argument('--input-mode',  default='hstack',
                        choices=['hstack', '2frame', 'gt-only'])
    parser.add_argument('--out',          default='data/repair_eval')
    args = parser.parse_args()

    analysis_dir = _REPO / args.analysis_dir
    gt_dir       = _REPO / args.gt_dir
    out_dir      = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Select wrong_primitive cases (same logic as feasibility test)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Load LoRA model
    # ------------------------------------------------------------------
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
    model = model.merge_and_unload()  # merge for faster inference
    model.eval()
    print('  Model ready.')

    # ------------------------------------------------------------------
    # 3. Load images + run inference
    # ------------------------------------------------------------------
    print('\nLoading renders...')
    for c in selected:
        c['gt_img']   = Image.open(c['gt_render']).convert('RGB')
        c['pred_img'] = Image.open(c['pred_render']).convert('RGB')

    print('Running inference...')
    items = [build_item(c['gt_img'], c['pred_img'], c['code'], args.input_mode) for c in selected]
    repaired_codes = []
    for i in tqdm(range(0, len(items), args.batch_size), desc='infer'):
        repaired_codes.extend(
            infer_batch(model, processor, items[i:i + args.batch_size]))

    del model
    torch.cuda.empty_cache()

    # Save generated codes
    code_dir = out_dir / 'repaired_codes'
    code_dir.mkdir(exist_ok=True)
    for c, code in zip(selected, repaired_codes):
        (code_dir / f'{c["stem"]}_repaired.py').write_text(code)

    # ------------------------------------------------------------------
    # 4. Score
    # ------------------------------------------------------------------
    # Post-process: strip trailing ) if it causes SyntaxError
    import ast as _ast
    def fix_trailing_paren(code):
        try:
            _ast.parse(code)
            return code
        except SyntaxError:
            s = code.rstrip()
            for _ in range(5):
                if s.endswith(')'):
                    s = s[:-1]
                    try:
                        _ast.parse(s)
                        return s
                    except SyntaxError:
                        continue
        return code

    repaired_codes = [fix_trailing_paren(c) for c in repaired_codes]

    print('\nScoring repaired codes (IoU + CD)...')
    results = []
    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        futs = [pool.submit(score_code, code, str(c['gt_stl']))
                for code, c in zip(repaired_codes, selected)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc='score'):
            results.append(fut.result())

    # Collect results in submission order
    results = [fut.result() for fut in futs]

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    base_ious = [c['iou'] for c in selected]
    deltas_iou, deltas_cd = [], []
    valid = meaningful = 0

    for c, r in zip(selected, results):
        if r['iou'] is not None:
            valid += 1
            di = r['iou'] - c['iou']
            deltas_iou.append(di)
            if di > 0.05:
                meaningful += 1

    n = len(selected)
    print('\n' + '=' * 60)
    print('REPAIR LORA — TRANSFER EVAL ON REAL wrong_primitive CASES')
    print('=' * 60)
    print(f'n={n}  baseline mean IoU={np.mean(base_ious):.4f}')
    print(f'\nLoRA repair results:')
    print(f'  valid rate:        {valid}/{n} = {valid/n*100:.1f}%')
    if deltas_iou:
        arr = np.array(deltas_iou)
        print(f'  mean  ΔIoU:        {arr.mean():+.4f}')
        print(f'  median ΔIoU:       {np.median(arr):+.4f}')
        print(f'  % ΔIoU > 0.05:     {meaningful/n*100:.1f}%  ({meaningful}/{n})')
        print(f'  ΔIoU distribution: '
              f'<-0.05:{(arr<-0.05).mean()*100:.0f}%  '
              f'[-0.05,0):{((arr>=-0.05)&(arr<0)).mean()*100:.0f}%  '
              f'[0,0.05):{((arr>=0)&(arr<0.05)).mean()*100:.0f}%  '
              f'≥0.05:{(arr>=0.05).mean()*100:.0f}%')

    # Verdict
    print('\n' + '=' * 60)
    passed = (
        valid / n >= 0.70 and
        (np.mean(deltas_iou) if deltas_iou else -1) > 0 and
        meaningful / n >= 0.20
    )
    if passed:
        print('✓ TRANSFER CONFIRMED — LoRA repair works on real failures.')
        print('  → Proceed to: more actions, more data, pseudo-target mining.')
    else:
        print('✗ Transfer weak. Check valid rate and ΔIoU distribution.')
    print('=' * 60)

    # Save
    out_json = out_dir / 'results.json'
    out_json.write_text(json.dumps({
        'n': n,
        'checkpoint': args.checkpoint,
        'baseline_mean_iou': float(np.mean(base_ious)),
        'valid_rate': valid / n,
        'mean_delta_iou': float(np.mean(deltas_iou)) if deltas_iou else None,
        'median_delta_iou': float(np.median(deltas_iou)) if deltas_iou else None,
        'pct_meaningful': meaningful / n,
        'per_case': [
            {'stem': c['stem'], 'baseline_iou': c['iou'],
             'repaired_iou': r.get('iou'), 'repaired_cd': r.get('cd'),
             'error_type': r.get('error_type')}
            for c, r in zip(selected, results)
        ]
    }, indent=2))
    print(f'\nResults saved to {out_json}')


if __name__ == '__main__':
    main()
