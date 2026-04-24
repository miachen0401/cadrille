"""1-example / 8-example overfit sanity check for repair LoRA.

Trains a fresh LoRA on 1 or 8 examples for many steps.
Success = model can reproduce correct geometry (not just valid syntax).

Usage
-----
  python3 tools/overfit_single.py --n 1 --steps 200
  python3 tools/overfit_single.py --n 8 --steps 200
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import trange
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from cadrille import Cadrille, collate
from experiments.repair_lora.repair_feasibility import score_code, _hstack

REPAIR_PROMPT = (
    "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).\n"
    "Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced with a proper "
    "sketch+extrude pattern matching the target geometry.\n"
    "Rewrite the code using sketch+extrude.\n\n"
    "Broken code:\n{code}"
)


def fix_trailing_paren(code):
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


@torch.no_grad()
def infer_one(model, processor, item, device, max_new_tokens=256):
    batch = collate([item], processor=processor, n_points=256, eval=True)
    gen_ids = model.generate(
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        point_clouds=batch['point_clouds'].to(device),
        is_pc=batch['is_pc'].to(device),
        is_img=batch['is_img'].to(device),
        pixel_values_videos=(batch['pixel_values_videos'].to(device)
                              if batch.get('pixel_values_videos') is not None else None),
        video_grid_thw=(batch['video_grid_thw'].to(device)
                        if batch.get('video_grid_thw') is not None else None),
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        bad_words_ids=[[model.config.video_token_id]],
    )
    prompt_len = batch['input_ids'].shape[1]
    return processor.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',  default='checkpoints/cadrille-rl')
    parser.add_argument('--train-data',  default='data/repair_sft/train.jsonl')
    parser.add_argument('--n',           type=int, default=1,
                        help='Number of examples to overfit on (1 or 8)')
    parser.add_argument('--steps',       type=int, default=200)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--lora-rank',   type=int, default=16)
    parser.add_argument('--eval-every',  type=int, default=25)
    parser.add_argument('--out',         default='checkpoints/overfit-test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data — pick shortest GT codes (simplest shapes)
    print(f'Loading training data (selecting {args.n} simplest examples)...')
    with open(_REPO / args.train_data) as f:
        records = [json.loads(l) for l in f]

    # Sort by GT code length; take type1 only first (cleaner)
    type1 = [r for r in records if r['corruption_type'] == 'type1']
    type1.sort(key=lambda r: len(r['gt_code']))
    selected_records = type1[:args.n]

    print(f'Selected {len(selected_records)} examples:')
    for r in selected_records:
        print(f'  {r["stem"]}  gt_len={len(r["gt_code"])}  gt={r["gt_code"][:80]}')

    # Build items with images
    train_items = []
    for r in selected_records:
        gt_img   = Image.open(r['gt_render']).convert('RGB')
        corr_img = Image.open(r['corrupt_render']).convert('RGB')
        train_items.append({
            'video':       [_hstack(gt_img, corr_img)],
            'description': REPAIR_PROMPT.format(action=r['action'], code=r['corrupt_code']),
            'answer':      r['gt_code'],
            'file_name':   r['stem'],
        })

    # Build eval items (same as train but without 'answer' key)
    eval_items = []
    for item in train_items:
        eval_items.append({k: v for k, v in item.items() if k != 'answer'})

    # Load model
    print(f'\nLoading {args.checkpoint}...')
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28, padding_side='left')
    model = Cadrille.from_pretrained(
        str(_REPO / args.checkpoint), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')

    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,  # no dropout for overfit test
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0)

    print(f'\nOverfitting {args.n} example(s) for {args.steps} steps (lr={args.lr})...')
    print('=' * 60)

    log = []
    for step in trange(1, args.steps + 1):
        # Cycle through train items
        batch_items = [train_items[step % len(train_items)]]
        batch = collate(batch_items, processor=processor, n_points=256, eval=False)

        out = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device),
            point_clouds=batch['point_clouds'].to(device),
            is_pc=batch['is_pc'].to(device),
            is_img=batch['is_img'].to(device),
            pixel_values_videos=(batch['pixel_values_videos'].to(device)
                                  if batch.get('pixel_values_videos') is not None else None),
            video_grid_thw=(batch['video_grid_thw'].to(device)
                             if batch.get('video_grid_thw') is not None else None),
        )

        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            print(f'\n--- Step {step}  loss={loss.item():.4f} ---')
            for i, (eval_item, r) in enumerate(zip(eval_items, selected_records)):
                code = infer_one(model, processor, eval_item, device)
                fixed = fix_trailing_paren(code)
                result = score_code(fixed, str(Path(r['gt_render']).parent.parent / 'deepcad_test_mesh' / f'{r["stem"]}.stl')) if fixed else {'iou': None}
                iou = result.get('iou')
                match = code.strip() == r['gt_code'].strip()
                iou_str = f'{iou:.3f}' if iou is not None else 'None'
                print(f'  Ex{i}: IoU={iou_str}  exact_match={match}')
                print(f'    out: {code[:100]}')
                print(f'    gt:  {r["gt_code"][:100]}')
                log.append({'step': step, 'loss': loss.item(),
                             'stem': r['stem'], 'iou': iou, 'exact_match': match,
                             'output': code})
            model.train()

    print('\n' + '=' * 60)
    print('OVERFIT SANITY CHECK COMPLETE')
    print(f'Final loss: {loss.item():.6f}')

    # Final eval
    model.eval()
    print('\nFinal outputs:')
    for i, (eval_item, r) in enumerate(zip(eval_items, selected_records)):
        code = infer_one(model, processor, eval_item, device)
        fixed = fix_trailing_paren(code)
        gt_stl = _REPO / 'data/deepcad_test_mesh' / f'{r["stem"]}.stl'
        if not gt_stl.exists():
            # Try corrupt_render dir parent
            result = score_code(fixed, None) if fixed else {'iou': None}
        else:
            result = score_code(fixed, str(gt_stl)) if fixed else {'iou': None}
        iou = result.get('iou')
        print(f'  Example {i} ({r["stem"]}):')
        print(f'    GT code:  {r["gt_code"]}')
        print(f'    Output:   {code}')
        print(f'    IoU: {iou}  exact_match: {code.strip()==r["gt_code"].strip()}')

    (out_dir / f'overfit_n{args.n}_log.json').write_text(json.dumps(log, indent=2))
    print(f'\nLog saved to {out_dir}/overfit_n{args.n}_log.json')


if __name__ == '__main__':
    main()
