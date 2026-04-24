"""Evaluate CadEvolve on HF bench track (Hula0401/test_bench).

For each sample:
  1. Execute GT CadQuery code → temp STL
  2. Render STL with CadEvolve's 8-view pipeline (experiments/cadevolve/render.py)
  3. Pass rendered image to CadEvolve model (Qwen2VL, image input, no system prompt)
  4. Score generated code vs GT STL (IoU via rl/reward.py)

Usage:
    python3 experiments/cadevolve/eval.py \\
        --ckpt checkpoints/cadevolve-rl1 \\
        --split all --limit 100 --seed 42 \\
        --out eval_outputs/bench/cadevolve_n300 \\
        --score-workers 2

    # smoke test
    python3 experiments/cadevolve/eval.py \\
        --ckpt checkpoints/cadevolve-rl1 \\
        --split test_iid --limit 3 \\
        --out eval_outputs/bench/cadevolve_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from qwen_vl_utils import process_vision_info

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from common.metrics import compute_metrics   # noqa: E402
from experiments.cadevolve.render import render_stl  # noqa: E402

_LD = os.environ.get('LD_LIBRARY_PATH', '/workspace/.local/lib')

# ---------------------------------------------------------------------------
# GT execution → normalized STL
# ---------------------------------------------------------------------------

_GT_TMPL = textwrap.dedent('''\
    import sys, io
    import cadquery as cq
    import trimesh, numpy as np
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get('result') or locals().get('r')
    if _r is None:
        raise ValueError('no result/r variable')
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    buf = trimesh.exchange.stl.export_stl(mesh)
    mesh2 = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')
    mesh2.apply_translation(-(mesh2.bounds[0]+mesh2.bounds[1])/2.0)
    ext = float(np.max(mesh2.extents))
    if ext > 1e-7:
        mesh2.apply_scale(2.0/ext)
    mesh2.export(sys.argv[1])
''')


def _exec_gt(gt_code: str, timeout: float = 60.0) -> str | None:
    """Execute GT code → normalized STL path, or None on failure."""
    script = _GT_TMPL.format(code=gt_code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    env = {**os.environ, 'LD_LIBRARY_PATH': _LD}
    try:
        r = subprocess.run([sys.executable, '-c', script, stl],
                           capture_output=True, timeout=timeout, env=env)
        if r.returncode == 0 and Path(stl).stat().st_size > 100:
            return stl
        return None
    except Exception:
        return None


# For render_stl we need a separate raw STL (not normalized) — render normalizes itself
_GT_RAW_TMPL = textwrap.dedent('''\
    import sys
    import cadquery as cq
    import trimesh
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get('result') or locals().get('r')
    if _r is None:
        raise ValueError('no result/r variable')
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    open(sys.argv[1],'wb').write(__import__('trimesh').exchange.stl.export_stl(mesh))
''')


def _exec_gt_raw(gt_code: str, timeout: float = 60.0) -> str | None:
    """Execute GT code → raw STL (unnormalized) for rendering."""
    script = _GT_RAW_TMPL.format(code=gt_code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    env = {**os.environ, 'LD_LIBRARY_PATH': _LD}
    try:
        r = subprocess.run([sys.executable, '-c', script, stl],
                           capture_output=True, timeout=timeout, env=env)
        if r.returncode == 0 and Path(stl).stat().st_size > 100:
            return stl
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(gen_code: str, gt_code: str, timeout: float = 32.0) -> dict:
    gt_stl = _exec_gt(gt_code, timeout=60.0)
    if gt_stl is None:
        return {'error_type': 'gt_exec_fail', 'iou': None, 'cd': None}
    try:
        iou, cd = compute_metrics(gen_code, gt_stl, timeout=timeout, use_pool=False)
    finally:
        Path(gt_stl).unlink(missing_ok=True)
    if iou is None:
        return {'error_type': 'runtime_error', 'iou': None, 'cd': None}
    return {
        'error_type': 'success' if iou > 0 else 'zero_iou',
        'iou': round(iou, 4),
        'cd':  round(cd, 6) if cd is not None else None,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(rows, model, processor, out_dir: Path,
             batch_size: int = 2, max_new_tokens: int = 768,
             score_workers: int = 2) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'

    done = set()
    if meta_path.exists():
        for line in open(meta_path):
            try:
                done.add(json.loads(line)['stem'])
            except Exception:
                pass

    todo = [r for r in rows if r['stem'] not in done]
    if not todo:
        print(f'  All {len(done)} done.', flush=True)
        return _summarize(meta_path)

    print(f'  {len(todo)} to run, {len(done)} already done', flush=True)
    device = next(model.parameters()).device
    meta_file = open(meta_path, 'a')
    pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []

    def _flush(wait_all=False):
        remaining = []
        for fut, stem, base in list(pending):
            if not wait_all and not fut.done():
                remaining.append((fut, stem, base)); continue
            score = fut.result()
            meta_file.write(json.dumps({**base, **score}) + '\n')
            meta_file.flush()
        pending.clear(); pending.extend(remaining)

    def _drain(batch):
        if not batch: return
        imgs, msgs = [], []
        for row in batch:
            # GT → raw STL → CadEvolve render image
            raw_stl = _exec_gt_raw(row['gt_code'])
            if raw_stl:
                try:
                    img = render_stl(raw_stl)
                except Exception as e:
                    print(f'  render error {row["stem"]}: {e}', flush=True)
                    img = None
                finally:
                    Path(raw_stl).unlink(missing_ok=True)
            else:
                img = None

            if img is None:
                # fallback: composite_png (bench GT render)
                img = row['composite_png'].convert('RGB')

            imgs.append(img)
            msgs.append([{'role': 'user', 'content': [{'type': 'image', 'image': img}]}])

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        vis, _ = process_vision_info(msgs)
        inp = processor(text=texts, images=vis, return_tensors='pt', padding=True).to(device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=model.config.eos_token_id,
            )

        prompt_len = inp['input_ids'].shape[1]
        for i, row in enumerate(batch):
            gen = processor.decode(out[i, prompt_len:], skip_special_tokens=True)
            (out_dir / f'{row["stem"]}.py').write_text(gen)
            base = {
                'stem':          row['stem'],
                'family':        row['family'],
                'difficulty':    row['difficulty'],
                'base_plane':    row['base_plane'],
                'split':         row['split'],
                'feature_tags':  row['feature_tags'],
                'feature_count': row['feature_count'],
                'code_len':      len(gen),
            }
            pending.append((pool.submit(_score, gen, row['gt_code']), row['stem'], base))
        _flush()

    batch = []
    for i, row in enumerate(todo):
        batch.append(row)
        if len(batch) >= batch_size:
            print(f'  [{i+1}/{len(todo)}] ...', end=' ', flush=True)
            _drain(batch); batch.clear()
            print('done', flush=True)
    if batch:
        print(f'  [{len(todo)}/{len(todo)}] final ...', end=' ', flush=True)
        _drain(batch); print('done', flush=True)

    _flush(wait_all=True)
    meta_file.close(); pool.shutdown(wait=True)
    return _summarize(meta_path)


def _summarize(meta_path: Path) -> dict:
    recs = [json.loads(l) for l in open(meta_path)]
    if not recs: return {}
    ok = [r for r in recs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    return {
        'n':         len(recs),
        'exec_rate': round(len(ok)/len(recs), 4) if recs else 0,
        'mean_iou':  round(sum(ious)/len(ious), 4) if ious else 0,
        'mean_cd':   round(sum(cds)/len(cds),   6) if cds  else None,
    }


def _report(meta_path: Path, label: str) -> None:
    recs = [json.loads(l) for l in open(meta_path)]
    if not recs: print('No results.'); return
    ok   = [r for r in recs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    print(f'\n{"="*58}')
    print(f'Model      : {label}')
    print(f'N          : {len(recs)}')
    print(f'Exec rate  : {len(ok)/len(recs)*100:.1f}%  ({len(ok)}/{len(recs)})')
    print(f'Mean IoU   : {sum(ious)/len(ious):.4f}  (n={len(ious)})' if ious else 'Mean IoU   : —')
    print(f'Mean CD    : {sum(cds)/len(cds):.6f}  (n={len(cds)})' if cds else 'Mean CD    : —')
    by_split = defaultdict(list)
    for r in recs: by_split[r.get('split','?')].append(r)
    print(f'\n{"Split":<22} {"N":>5} {"Exec%":>7} {"IoU":>7}')
    print('-' * 44)
    for sp in sorted(by_split):
        rs = by_split[sp]
        ok_sp = [x for x in rs if x.get('error_type') == 'success']
        iou_sp = [x['iou'] for x in ok_sp if x.get('iou') is not None]
        iou_s = f'{sum(iou_sp)/len(iou_sp):>7.4f}' if iou_sp else '      —'
        print(f'{sp:<22} {len(rs):>5} {len(ok_sp)/len(rs)*100:>6.1f}%{iou_s}')
    print('=' * 58)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Eval CadEvolve on Hula0401/test_bench',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--ckpt',          required=True)
    ap.add_argument('--split',         default='test_iid',
                    choices=['test_iid','test_ood_family','test_ood_plane','all'])
    ap.add_argument('--limit',         type=int, default=0)
    ap.add_argument('--seed',          type=int, default=42)
    ap.add_argument('--batch-size',    type=int, default=2)
    ap.add_argument('--max-new-tokens',type=int, default=768)
    ap.add_argument('--score-workers', type=int, default=2)
    ap.add_argument('--out',           required=True)
    ap.add_argument('--hf-repo',       default='Hula0401/test_bench')
    ap.add_argument('--label',         default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir   = Path(args.out)
    label     = args.label or ckpt_path.name

    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    print(f'Loading {args.hf_repo} ...', flush=True)
    ds = load_dataset(args.hf_repo, token=token)

    splits = ['test_iid','test_ood_family','test_ood_plane'] if args.split=='all' else [args.split]
    rows = []
    for sp in splits:
        sp_rows = list(ds[sp])
        if args.limit:
            rng = random.Random(args.seed); rng.shuffle(sp_rows)
            sp_rows = sp_rows[:args.limit]
        rows.extend(sp_rows)
    print(f'Total: {len(rows)} samples', flush=True)

    # Load model
    print(f'Loading model from {ckpt_path} ...', flush=True)
    # Workaround: get_text_config(decoder=True) returns dict in transformers 4.50.3
    Qwen2VLConfig.get_text_config = lambda self, **kw: self
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(ckpt_path), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda')
    # Tie lm_head (not saved in checkpoint — weight-tied to embed_tokens)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = model.model.embed_tokens.weight
        print('  lm_head tied to embed_tokens', flush=True)
    model.eval()
    print('Model loaded.', flush=True)

    processor = AutoProcessor.from_pretrained(str(ckpt_path),
        min_pixels=200704, max_pixels=1003520*4, padding_side='left')

    print(f'\nRunning eval → {out_dir}', flush=True)
    summary = run_eval(rows, model, processor, out_dir,
                       batch_size=args.batch_size,
                       max_new_tokens=args.max_new_tokens,
                       score_workers=args.score_workers)
    print(f'\nSummary: {json.dumps(summary, indent=2)}')
    _report(out_dir / 'metadata.jsonl', label)


if __name__ == '__main__':
    main()
