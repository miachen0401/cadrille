"""Evaluate Cadrille / CADEvolve / zero-shot VLM on a directory of STL files.

Same metric pipeline as eval/bench.py (greedy generation, IoU + CD via
compute_metrics) but the input distribution is a local STL directory rather
than an HF dataset row. Used for DeepCAD / Fusion360 evaluation where the
ground truth is a STL file (no GT cadquery code, no upstream composite_png).

Per item:
  1. Sample N STLs from --stl-dir (deterministic with --seed).
  2. Render that STL → 4-view 268×268 PIL via common.meshio.render_img
     (uses the {stem}_render.png pre-rendered cache when available).
  3. Run the model (cadrille / cadevolve / qwen25vl_zs) on that image.
  4. Score gen_code against the GT STL directly via compute_metrics —
     no need to exec a GT-code template because we already have the STL.

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a

    # Cadrille on DeepCAD (300 sampled)
    uv run python -m eval.bench_stl \\
        --ckpt checkpoints/cadrille-rl --model-type cadrille \\
        --stl-dir data/deepcad_test_mesh --n-samples 300 --seed 42 \\
        --out eval_outputs/deepcad_n300/cadrille_rl

    # Q3VL-v3 cadrille on Fusion360
    uv run python -m eval.bench_stl \\
        --ckpt checkpoints/cadrille-qwen3vl-v3-clean-50k/checkpoint-34000 \\
        --model-type cadrille --backbone qwen3_vl \\
        --stl-dir data/fusion360_test_mesh --n-samples 300 --seed 42 \\
        --out eval_outputs/fusion360_n300/cadrille_qwen3vl_v3

    # CADEvolve (8-view) on DeepCAD
    uv run python -m eval.bench_stl \\
        --ckpt checkpoints/cadevolve-rl1 --model-type cadevolve \\
        --stl-dir data/deepcad_test_mesh --n-samples 300 --seed 42 \\
        --out eval_outputs/deepcad_n300/cadevolve_rl1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoProcessor

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from common.model import Cadrille, get_cadrille_class, collate  # noqa: E402
from common.meshio import render_img                            # noqa: E402
from common.metrics import compute_metrics                       # noqa: E402

_N_POINTS = 256
_DESCRIPTION = 'Generate cadquery code'
_CADEVOLVE_PROMPT = 'Generate CadQuery Python code for this 3D CAD model shown in multiple views.'
_ZS_PROMPT = (
    "Look at the 3D CAD model rendered in this image and write a complete "
    "Python script using the cadquery library that reproduces this exact "
    "geometry.\n\n"
    "Strict output rules:\n"
    "- Start the script with `import cadquery as cq`\n"
    "- Bind the final shape to a variable named exactly `result`\n"
    "- Output ONLY runnable Python — no prose, no markdown code fences.\n"
)


def _strip_code_fences(text: str) -> str:
    import re
    m = re.search(r'```(?:python|cadquery)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r'^\s*```(?:python|cadquery)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


def _score_against_stl(gen_code: str, gt_stl_path: str,
                       timeout: float = 32.0) -> dict:
    """Score directly against a normalised GT STL — simpler than bench.py's
    `_score` because we have the STL on disk; no GT exec template needed."""
    iou_reward, cd = compute_metrics(gen_code, gt_stl_path,
                                     timeout=timeout, use_pool=False)
    if iou_reward == -1.0:
        return {'error_type': 'runtime_error', 'iou': None, 'cd': None}
    return {
        'error_type': 'success' if iou_reward > 0 else 'zero_iou',
        'iou': round(iou_reward, 4),
        'cd':  round(cd, 6) if cd is not None else None,
    }


def sample_stls(stl_dir: Path, n: int, seed: int) -> list[dict]:
    paths = sorted(p for p in stl_dir.iterdir() if p.suffix.lower() == '.stl')
    rng = random.Random(seed); rng.shuffle(paths)
    paths = paths[:n] if n else paths
    print(f'sampled {len(paths)} STLs from {stl_dir}', flush=True)
    return [{'stem': p.stem, 'gt_mesh_path': str(p)} for p in paths]


def _render_to_pil(stl_path: str):
    out = render_img(stl_path)
    return out['video'][0].convert('RGB')


# ---------------------------------------------------------------------------
# Inference loops — adapted from eval/bench.py but with STL-based scoring
# ---------------------------------------------------------------------------

def run_cadrille(rows, model, processor, out_dir: Path,
                 batch_size=4, max_new_tokens=768, score_workers=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'
    done_stems = set()
    if meta_path.exists():
        for line in open(meta_path):
            try: done_stems.add(json.loads(line)['stem'])
            except: pass
    todo = [r for r in rows if r['stem'] not in done_stems]
    if not todo:
        print(f'  all {len(done_stems)} done.', flush=True); return _summarize(meta_path)
    print(f'  {len(todo)} to run, {len(done_stems)} cached', flush=True)

    device = next(model.parameters()).device
    meta_file = open(meta_path, 'a')
    pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []

    def _flush(wait=False):
        keep = []
        for fut, base in list(pending):
            if not wait and not fut.done(): keep.append((fut, base)); continue
            score = fut.result()
            meta_file.write(json.dumps({**base, **score}) + '\n')
            meta_file.flush()
        pending.clear(); pending.extend(keep)

    def _drain(batch):
        if not batch: return
        if hasattr(model, 'rope_deltas'): model.rope_deltas = None
        items = []
        for row in batch:
            img = _render_to_pil(row['gt_mesh_path'])
            items.append({'video': [img], 'description': _DESCRIPTION,
                          'file_name': row['stem']})
        b = collate(items, processor, _N_POINTS, eval=True)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=b['input_ids'].to(device),
                attention_mask=b['attention_mask'].to(device),
                point_clouds=b['point_clouds'].to(device),
                is_pc=b['is_pc'].to(device),
                is_img=b['is_img'].to(device),
                pixel_values_videos=(b['pixel_values_videos'].to(device)
                    if b.get('pixel_values_videos') is not None else None),
                video_grid_thw=(b['video_grid_thw'].to(device)
                    if b.get('video_grid_thw') is not None else None),
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=getattr(model.config, 'eos_token_id', None)
                              or model.config.text_config.eos_token_id)
        prompt_len = b['input_ids'].shape[1]
        for i, row in enumerate(batch):
            gen = processor.decode(out_ids[i, prompt_len:], skip_special_tokens=True)
            (out_dir / f'{row["stem"]}.py').write_text(gen)
            base = {'stem': row['stem'], 'gt_mesh_path': row['gt_mesh_path'],
                    'code_len': len(gen)}
            pending.append((pool.submit(_score_against_stl, gen,
                                        row['gt_mesh_path']), base))
        _flush(wait=False)

    batch = []
    for i, r in enumerate(todo):
        batch.append(r)
        if len(batch) >= batch_size:
            print(f'  [{i+1}/{len(todo)}] batch …', end=' ', flush=True)
            _drain(batch); batch.clear(); print('done', flush=True)
    if batch:
        print(f'  [{len(todo)}/{len(todo)}] final …', end=' ', flush=True)
        _drain(batch); print('done', flush=True)
    _flush(wait=True)
    meta_file.close(); pool.shutdown(wait=True)
    return _summarize(meta_path)


def run_image_only(rows, model, processor, out_dir: Path, prompt: str,
                   batch_size=2, max_new_tokens=1024, score_workers=4,
                   strip_fences=False):
    """For cadevolve and zero-shot VLMs: single-image input via standard chat."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'
    done_stems = set()
    if meta_path.exists():
        for line in open(meta_path):
            try: done_stems.add(json.loads(line)['stem'])
            except: pass
    todo = [r for r in rows if r['stem'] not in done_stems]
    if not todo:
        print(f'  all {len(done_stems)} done.', flush=True); return _summarize(meta_path)
    print(f'  {len(todo)} to run, {len(done_stems)} cached', flush=True)

    device = next(model.parameters()).device
    meta_file = open(meta_path, 'a')
    pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []

    def _flush(wait=False):
        keep = []
        for fut, base in list(pending):
            if not wait and not fut.done(): keep.append((fut, base)); continue
            score = fut.result()
            meta_file.write(json.dumps({**base, **score}) + '\n')
            meta_file.flush()
        pending.clear(); pending.extend(keep)

    def _drain(batch):
        if not batch: return
        msgs, imgs = [], []
        for row in batch:
            img = _render_to_pil(row['gt_mesh_path'])
            imgs.append(img)
            msgs.append([{'role':'user', 'content':[
                {'type':'image', 'image': img},
                {'type':'text',  'text':  prompt}]}])
        texts = [processor.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True)
                 for m in msgs]
        from qwen_vl_utils import process_vision_info
        vis, _ = process_vision_info(msgs)
        inp = processor(text=texts, images=vis, return_tensors='pt',
                        padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=getattr(model.config, 'eos_token_id', None)
                              or model.config.text_config.eos_token_id)
        prompt_len = inp['input_ids'].shape[1]
        for i, row in enumerate(batch):
            gen = processor.decode(out[i, prompt_len:], skip_special_tokens=True)
            if strip_fences: gen = _strip_code_fences(gen)
            (out_dir / f'{row["stem"]}.py').write_text(gen)
            base = {'stem': row['stem'], 'gt_mesh_path': row['gt_mesh_path'],
                    'code_len': len(gen)}
            pending.append((pool.submit(_score_against_stl, gen,
                                        row['gt_mesh_path']), base))
        _flush(wait=False)

    batch = []
    for i, r in enumerate(todo):
        batch.append(r)
        if len(batch) >= batch_size:
            print(f'  [{i+1}/{len(todo)}] batch …', end=' ', flush=True)
            _drain(batch); batch.clear(); print('done', flush=True)
    if batch:
        print(f'  [{len(todo)}/{len(todo)}] final …', end=' ', flush=True)
        _drain(batch); print('done', flush=True)
    _flush(wait=True)
    meta_file.close(); pool.shutdown(wait=True)
    return _summarize(meta_path)


def _summarize(meta_path: Path) -> dict:
    rs = []
    for line in open(meta_path):
        try: rs.append(json.loads(line))
        except: pass
    if not rs: return {}
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    return {
        'n':         len(rs),
        'exec_rate': round(len(ok)/len(rs), 4) if rs else 0,
        'mean_iou':  round(sum(ious)/len(ious), 4) if ious else None,
        'mean_cd':   round(sum(cds)/len(cds),   6) if cds  else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',         required=False, default=None)
    ap.add_argument('--model-type',   default='cadrille',
                    choices=['cadrille', 'cadevolve', 'qwen25vl_zs'])
    ap.add_argument('--backbone',     default='qwen2_vl',
                    choices=['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl'])
    ap.add_argument('--base-model',   default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--stl-dir',      required=True)
    ap.add_argument('--n-samples',    type=int, default=300)
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--batch-size',   type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers',  type=int, default=4)
    ap.add_argument('--attn-impl',    default='sdpa',
                    choices=['sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--out',          required=True)
    ap.add_argument('--label',        default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    rows = sample_stls(Path(args.stl_dir), args.n_samples, args.seed)

    # ── Load model + processor ──────────────────────────────────────────────
    proc_src = args.base_model
    print(f'Loading processor from {proc_src} ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        proc_src, min_pixels=200704, max_pixels=1003520, padding_side='left')

    print(f'Loading model ({args.model_type}, backbone={args.backbone}) '
          f'from {args.ckpt or args.base_model} ...', flush=True)
    if args.model_type == 'qwen25vl_zs':
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl, device_map='cuda')
    elif args.model_type == 'cadevolve':
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(args.ckpt), torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl, device_map='cuda')
        _embed = model.get_input_embeddings()
        if model.lm_head.weight.data_ptr() != _embed.weight.data_ptr():
            model.lm_head.weight = _embed.weight
    else:
        cadrille_cls = get_cadrille_class(args.backbone)
        if args.backbone == 'qwen3_vl':
            processor = AutoProcessor.from_pretrained(
                str(args.ckpt), min_pixels=200704, max_pixels=1003520,
                padding_side='left')
        model = cadrille_cls.from_pretrained(
            str(args.ckpt), torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl, device_map='cuda')
    model.eval()
    print('Model loaded.', flush=True)

    # ── Run ─────────────────────────────────────────────────────────────────
    if args.model_type == 'cadrille':
        summary = run_cadrille(rows, model, processor, out_dir,
                               batch_size=args.batch_size,
                               max_new_tokens=args.max_new_tokens,
                               score_workers=args.score_workers)
    else:
        prompt = _ZS_PROMPT if args.model_type == 'qwen25vl_zs' else _CADEVOLVE_PROMPT
        summary = run_image_only(rows, model, processor, out_dir,
                                 prompt=prompt,
                                 batch_size=args.batch_size,
                                 max_new_tokens=args.max_new_tokens,
                                 score_workers=args.score_workers,
                                 strip_fences=(args.model_type == 'qwen25vl_zs'))
    print(f'\nSummary: {json.dumps(summary, indent=2)}', flush=True)


if __name__ == '__main__':
    main()
