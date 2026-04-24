"""eval_img.py — Partial img-mode evaluation against paper benchmarks.

Runs cadrille inference in img mode on a random sample of N STLs from one or
more test splits, saves the generated .py files, then calls evaluate.py to
compute IoU and Chamfer Distance. Results are printed to stdout and saved to
a CSV alongside the .py files.

Usage
-----
    python3 tools/eval_img.py [OPTIONS]

Options
-------
    --checkpoint PATH       Path to cadrille checkpoint (default: ./checkpoints/cadrille-sft)
    --splits   NAME[:DIR]   Test split(s) to evaluate. Can be specified multiple times.
                            NAME is a short label (e.g. deepcad, fusion360).
                            DIR defaults to ./data/{NAME}_test_mesh.
                            (default: deepcad, fusion360)
    --n-samples N           Samples per split (default: 500)
    --out-dir   PATH        Parent directory for output .py files (default: ./work_dirs/eval_img)
    --seed      N           Random seed for sampling (default: 42)
    --batch-size N          Inference batch size (default: 8)
    --max-new-tokens N      Max tokens to generate per sample (default: 768)
    --no-evaluate           Skip evaluate.py; only generate .py files.

Examples
--------
    # Quick 200-sample sanity check on DeepCAD only:
    python3 tools/eval_img.py --splits deepcad --n-samples 200

    # Full partial eval on both benchmarks (paper Table 2 numbers):
    python3 tools/eval_img.py --n-samples 500

    # Use RL checkpoint:
    python3 tools/eval_img.py --checkpoint ./checkpoints/cadrille-rl --out-dir ./work_dirs/eval_rl

Output
------
    work_dirs/eval_img/{label}/
        {stem}+0.py     — generated CadQuery code for each sample
        results.csv     — per-sample IoU and CD (written by evaluate.py)

    Stdout summary:
        [deepcad] mean iou: 0.864  IR=1.6%  median CD=0.181

Notes
-----
    - Skips samples whose .py file already exists (safe to resume after interruption).
    - GT meshes must be pre-normalised to [0, 1]^3 (deepcad_test_mesh and
      fusion360_test_mesh from HuggingFace are already in this format).
    - evaluate.py writes tmp_mesh/ and tmp_brep/ as siblings of out_dir;
      these are cleaned up automatically after each split.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys

import torch
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadrille import Cadrille, collate
from rl.dataset import render_img

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_split(test_dir, out_dir, label, model, processor, n_samples, seed, batch_size, max_new_tokens):
    os.makedirs(out_dir, exist_ok=True)
    existing = {f for f in os.listdir(out_dir) if f.endswith('.py')}

    stl_files = sorted(f for f in os.listdir(test_dir) if f.endswith('.stl'))
    rng = random.Random(seed)
    rng.shuffle(stl_files)
    selected = stl_files[:n_samples]
    print(f'\n[{label}] {len(selected)} samples from {test_dir}')

    examples = []
    for fname in tqdm(selected, desc=f'{label} render'):
        stem = fname[:-4]
        if f'{stem}+0.py' in existing:
            continue
        gt_path = os.path.join(test_dir, fname)
        try:
            img_item = render_img(gt_path)
            img_item.update({'description': 'Generate cadquery code', 'file_name': stem})
            examples.append(img_item)
        except Exception as e:
            print(f'  SKIP {fname}: {e}')

    print(f'[{label}] {len(existing)} cached + {len(examples)} new to generate')

    for i in tqdm(range(0, len(examples), batch_size), desc=f'{label} infer'):
        chunk = examples[i:i + batch_size]
        batch = collate(chunk, processor=processor, n_points=256, eval=True)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                point_clouds=batch['point_clouds'].to(model.device),
                is_pc=batch['is_pc'].to(model.device),
                is_img=batch['is_img'].to(model.device),
                pixel_values_videos=(
                    batch['pixel_values_videos'].to(model.device)
                    if batch.get('pixel_values_videos') is not None else None),
                video_grid_thw=(
                    batch['video_grid_thw'].to(model.device)
                    if batch.get('video_grid_thw') is not None else None),
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                bad_words_ids=[[model.config.video_token_id]])
        prompt_len = batch['input_ids'].shape[1]
        for j, ex in enumerate(chunk):
            code = processor.decode(generated_ids[j, prompt_len:], skip_special_tokens=True)
            with open(os.path.join(out_dir, f"{ex['file_name']}+0.py"), 'w') as f:
                f.write(code)


def _run_evaluate(gt_dir, out_dir, label):
    evaluate_py = os.path.join(_REPO_ROOT, 'evaluate.py')
    results_csv = os.path.join(out_dir, 'results.csv')

    # evaluate.py creates tmp_mesh/ and tmp_brep/ as siblings of out_dir
    work_dir = os.path.dirname(out_dir)
    for d in ['tmp_mesh', 'tmp_brep']:
        p = os.path.join(work_dir, d)
        if os.path.exists(p):
            shutil.rmtree(p)

    print(f'\n[{label}] Running evaluate.py ...')
    result = subprocess.run([
        sys.executable, evaluate_py,
        '--gt-mesh-path', gt_dir,
        '--pred-py-path', out_dir,
        '--n-points', '8192',
        '--results-csv', results_csv,
    ])
    if result.returncode != 0:
        print(f'[{label}] evaluate.py failed (exit {result.returncode})')

    for d in ['tmp_mesh', 'tmp_brep']:
        p = os.path.join(work_dir, d)
        if os.path.exists(p):
            shutil.rmtree(p)


def main():
    parser = argparse.ArgumentParser(
        description='Partial img-mode evaluation of a cadrille checkpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--checkpoint', default='./checkpoints/cadrille-sft',
                        help='Checkpoint directory (default: %(default)s)')
    parser.add_argument('--splits', nargs='+', default=['deepcad', 'fusion360'],
                        metavar='NAME[:DIR]',
                        help='Test splits to evaluate (default: deepcad fusion360)')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Samples per split (default: %(default)s)')
    parser.add_argument('--out-dir', default='./work_dirs/eval_img',
                        help='Output parent directory (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=768)
    parser.add_argument('--no-evaluate', action='store_true',
                        help='Skip evaluate.py; only generate .py files')
    args = parser.parse_args()

    # Parse splits
    splits = []
    for s in args.splits:
        if ':' in s:
            label, test_dir = s.split(':', 1)
        else:
            label = s
            test_dir = os.path.join(_REPO_ROOT, 'data', f'{label}_test_mesh')
        out_dir = os.path.join(args.out_dir, label)
        splits.append((label, test_dir, out_dir))

    print(f'Loading model from {args.checkpoint} ...')
    model = Cadrille.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='auto')
    model.eval()

    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, padding_side='left')

    for label, test_dir, out_dir in splits:
        _run_split(test_dir, out_dir, label, model, processor,
                   args.n_samples, args.seed, args.batch_size, args.max_new_tokens)

    del model
    torch.cuda.empty_cache()

    if not args.no_evaluate:
        for label, test_dir, out_dir in splits:
            _run_evaluate(test_dir, out_dir, label)


if __name__ == '__main__':
    main()
