"""Generate CadQuery code for specific cases and score them.

Given a checkpoint, a list of case IDs, and a modality, this tool:
  1. Finds the STL file for each case ID (searches common data dirs automatically:
     deepcad_train_mesh, deepcad_test_mesh, fusion360_train_mesh, fusion360_test_mesh).
  2. Loads the model once.
  3. Generates N completions per case at the requested temperature.
  4. Computes IoU reward for each completion.
  5. Prints per-sample code + IoU, and a summary table when n > 1 or cases > 1.
  6. Optionally saves all results to a JSON file (--output).

Arguments
---------
  --ckpt            Path to checkpoint directory (required)
  --cases           One or more case IDs, space-separated (required)
  --modality        img or pc (default: img)
  --temp            Sampling temperature; 0 = greedy (default: 0)
  --n               Number of completions per case (default: 1)
  --max-new-tokens  Max tokens to generate (default: 1024)
  --data-dir        Override STL search directory (default: searches all data/ subdirs)
  --no-score        Skip IoU scoring, just print generated code
  --output          Save results to a JSON file

Usage
-----
# Single case, img mode, greedy (temp=0):
python tools/infer_cases.py \\
    --ckpt  checkpoints/cadrille-sft \\
    --cases 00097786 \\
    --modality img --temp 0

# Multiple cases, stochastic (temp=0.5), 16 samples, save to file:
python tools/infer_cases.py \\
    --ckpt  checkpoints/cadrille-sft \\
    --cases 00857683 00773298 00187217 00769659 00808536 00412993 \\
    --modality img --temp 0.5 --n 16 \\
    --output results.json

# Fusion360 case, pc mode:
python tools/infer_cases.py \\
    --ckpt  checkpoints/cadrille-sft \\
    --cases 45359_1768ab3f_0043_0298 \\
    --modality pc --temp 0.3 --n 4

# From a specific directory only:
python tools/infer_cases.py \\
    --ckpt  checkpoints/cadrille-sft \\
    --cases 00097786 \\
    --data-dir data/deepcad_test_mesh
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from common.meshio import render_img
from common.metrics import _get_worker_path

import subprocess
import json as _json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIRS = [
    'data/deepcad_train_mesh',
    'data/deepcad_test_mesh',
    'data/fusion360_train_mesh',
    'data/fusion360_test_mesh',
]


def find_stl(case_id: str, search_dirs: list) -> str | None:
    """Return first STL path whose stem matches case_id."""
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        # Direct match
        candidate = os.path.join(d, f'{case_id}.stl')
        if os.path.exists(candidate):
            return candidate
        # Recursive search (Fusion360 uses subdirs)
        import glob
        hits = glob.glob(os.path.join(d, '**', f'{case_id}.stl'), recursive=True)
        if hits:
            return hits[0]
    return None


def score_code(code: str, gt_mesh_path: str) -> dict:
    """Run code through the reward subprocess, return {'iou', 'error'}."""
    payload = _json.dumps({
        'code_str': code,
        'gt_mesh_path': gt_mesh_path,
        'compute_chamfer': False,
    })
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker_path()],
            input=payload, capture_output=True, text=True, timeout=30)
        if proc.stdout.strip():
            return _json.loads(proc.stdout.strip())
        return {'iou': None, 'error': proc.stderr.strip()[-200:] or 'empty stdout'}
    except subprocess.TimeoutExpired:
        return {'iou': None, 'error': 'timeout'}
    except Exception as e:
        return {'iou': None, 'error': str(e)}


@torch.no_grad()
def generate_for_case(model, processor, stl_path: str, modality: str,
                      temperature: float, n: int, max_new_tokens: int,
                      device) -> list[str]:
    """Generate n completions for one case. Returns list of code strings."""
    gen_model = model.module if hasattr(model, 'module') else model

    # Build input item
    item: dict = {'description': 'Generate cadquery code',
                  'file_name': os.path.splitext(os.path.basename(stl_path))[0],
                  'gt_mesh_path': stl_path}
    if modality == 'img':
        item.update(render_img(stl_path))
    else:
        import trimesh, numpy as np
        from dataset import mesh_to_point_cloud
        mesh = trimesh.load(stl_path)
        pc = mesh_to_point_cloud(mesh, 256)
        pc = (pc - 0.5) * 2
        item['point_cloud'] = pc

    batch = collate([item], processor=processor, n_points=256, eval=True)
    prompt_len = batch['input_ids'].shape[1]

    had_gc = getattr(gen_model, 'is_gradient_checkpointing', False)
    if had_gc:
        gen_model.gradient_checkpointing_disable()
    gen_model.eval()

    # Block vision tokens
    bad_words = None
    cfg = getattr(gen_model, 'config', None)
    if cfg is not None:
        blocked = []
        for attr in ('video_token_id', 'image_token_id'):
            tid = getattr(cfg, attr, None)
            if tid is not None:
                blocked.append([tid])
        if blocked:
            bad_words = blocked

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=1.0 if do_sample else None,
        top_k=50  if do_sample else None,
        bad_words_ids=bad_words,
    )

    codes = []
    for _ in range(n):
        if hasattr(gen_model, 'rope_deltas'):
            gen_model.rope_deltas = None
        gen_input = {
            'input_ids':           batch['input_ids'].to(device),
            'attention_mask':      batch['attention_mask'].to(device),
            'point_clouds':        batch['point_clouds'].to(device),
            'is_pc':               batch['is_pc'].to(device),
            'is_img':              batch['is_img'].to(device),
        }
        if batch.get('pixel_values_videos') is not None:
            gen_input['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
        if batch.get('video_grid_thw') is not None:
            gen_input['video_grid_thw'] = batch['video_grid_thw'].to(device)

        ids = gen_model.generate(**gen_input, **gen_kwargs)
        code = processor.decode(ids[0, prompt_len:],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        codes.append(code)

    if had_gc:
        gen_model.gradient_checkpointing_enable()

    return codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt',      required=True,
                        help='Checkpoint directory')
    parser.add_argument('--cases',     nargs='+', required=True,
                        help='One or more case IDs (stem of the STL file)')
    parser.add_argument('--modality',  choices=['img', 'pc'], default='img',
                        help='Input modality (default: img)')
    parser.add_argument('--temp',      type=float, default=0.0,
                        help='Sampling temperature; 0 = greedy (default: 0)')
    parser.add_argument('--n',         type=int, default=1,
                        help='Number of completions per case (default: 1)')
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                        help='Max generation tokens (default: 1024)')
    parser.add_argument('--data-dir',  type=str, default=None,
                        help='Override STL search dir (default: searches all data/ subdirs)')
    parser.add_argument('--no-score',  action='store_true',
                        help='Skip IoU scoring (just print generated code)')
    parser.add_argument('--output',    type=str, default=None,
                        help='Save results to a JSON file (e.g. results.json)')
    args = parser.parse_args()

    search_dirs = [args.data_dir] if args.data_dir else _DEFAULT_DATA_DIRS

    # ── model ────────────────────────────────────────────────────────────────
    print(f'Loading model from {args.ckpt} ...')
    proc_kwargs = dict(min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')
    processor = AutoProcessor.from_pretrained(args.ckpt, **proc_kwargs)
    model = Cadrille.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto')
    model.gradient_checkpointing_enable()
    device = next(model.parameters()).device
    print(f'Model on {device}\n')

    # ── per-case loop ─────────────────────────────────────────────────────────
    all_results = []
    for case_id in args.cases:
        stl = find_stl(case_id, search_dirs)
        if stl is None:
            print(f'[{case_id}] STL not found in: {search_dirs}')
            all_results.append({'case_id': case_id, 'samples': []})
            continue

        print(f'{"─"*64}')
        print(f'Case : {case_id}  |  modality={args.modality}  '
              f'temp={args.temp}  n={args.n}')
        print(f'STL  : {stl}')

        codes = generate_for_case(model, processor, stl, args.modality,
                                  args.temp, args.n, args.max_new_tokens, device)

        samples = []
        for i, code in enumerate(codes):
            result = {'code': code, 'iou': None, 'error': None}
            if not args.no_score:
                scored = score_code(code, stl)
                result['iou']   = scored.get('iou')
                result['error'] = scored.get('error')

            iou_str = (f'{result["iou"]:.4f}' if result['iou'] is not None
                       else f'FAIL ({(result["error"] or "")[:60]})')
            label = f'[{i}]' if args.n > 1 else ''
            print(f'\n{label} IoU = {iou_str}')
            print(f'Code ({len(code)} chars):')
            for line in code.splitlines():
                print(f'  {line}')
            samples.append(result)

        if not args.no_score:
            ious = [s['iou'] for s in samples if s['iou'] is not None]
            pos = [s for s in samples if s.get('iou') is not None and s['iou'] > 0]
            if ious and args.n > 1:
                print(f'\n  → best={max(ious):.4f}  mean={sum(ious)/len(ious):.4f}  '
                      f'valid={len(ious)}/{args.n}')
            if pos:
                max_len_pos = max(len(s['code']) for s in pos)
                avg_len_iou = sum(len(s['code']) * s['iou'] for s in pos) / len(pos)
                print(f'     max_len(IoU>0)={max_len_pos}  '
                      f'avg(len×IoU)={avg_len_iou:.2f}')

        all_results.append({'case_id': case_id, 'stl': stl, 'samples': samples})

    # ── final summary ─────────────────────────────────────────────────────────
    if len(args.cases) > 1:
        print(f'\n{"="*64}')
        print('SUMMARY')
        for r in all_results:
            ious = [s['iou'] for s in r['samples'] if s.get('iou') is not None]
            if not ious:
                print(f'  {r["case_id"]:40s}  no valid results')
            else:
                pos = [s for s in r['samples'] if s.get('iou') is not None and s['iou'] > 0]
                extra = ''
                if pos:
                    max_len_pos = max(len(s['code']) for s in pos)
                    avg_len_iou = sum(len(s['code']) * s['iou'] for s in pos) / len(pos)
                    extra = f'  max_len(IoU>0)={max_len_pos}  avg(len×IoU)={avg_len_iou:.2f}'
                print(f'  {r["case_id"]:40s}  '
                      f'best={max(ious):.4f}  mean={sum(ious)/len(ious):.4f}  '
                      f'valid={len(ious)}/{len(r["samples"])}{extra}')
        print('=' * 64)

    # ── save to file ──────────────────────────────────────────────────────────
    if args.output:
        import json
        payload = {
            'ckpt':     args.ckpt,
            'modality': args.modality,
            'temp':     args.temp,
            'n':        args.n,
            'results':  all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nSaved → {args.output}')


if __name__ == '__main__':
    main()
