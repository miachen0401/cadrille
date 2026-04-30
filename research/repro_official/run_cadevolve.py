"""Run CADEvolve with PROPER 8-view input under transformers 4.50.3.

CADEvolve (kulibinai/cadevolve, arXiv:2602.16317) was trained on 8-view
476×952 axis-coloured collages produced by `experiments/cadevolve/render.py`.
Our earlier `eval/bench_stl.py` mistakenly fed it 4-view 268×268 collages
from `common.meshio.render_img` — the wrong distribution → IoU collapsed.

This script:
  - Loads CADEvolve via vanilla Qwen2VLForConditionalGeneration (no Cadrille
    mixin — CADEvolve is plain Qwen2VL fine-tune).
  - For DeepCAD / Fusion360: renders each STL → 8-view via render_stl().
  - For cad_bench_722: exec gt_code → STL → render_stl() (matches what the
    existing experiments/cadevolve/eval.py does).
  - Feeds image to CADEvolve, decodes, scores against GT STL via
    common.metrics.compute_metrics.

Usage (run via .venv/bin/python — NOT `uv run`, which auto-syncs back to
transformers 5.x):
    .venv/bin/python research/repro_official/run_cadevolve.py \\
        --dataset deepcad --n-samples 300

Datasets supported:
    --dataset deepcad   → data/deepcad_test_mesh/<n_samples random STLs>
    --dataset fusion360 → data/fusion360_test_mesh/<n_samples random STLs>
    --dataset cad_bench → BenchCAD/cad_bench_722 (full 720 rows)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


_GT_TMPL_NORM = textwrap.dedent('''\
    import sys, io
    import cadquery as cq
    import trimesh, numpy as np
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get("result") or locals().get("r")
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    buf = trimesh.exchange.stl.export_stl(mesh)
    mesh2 = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')
    mesh2.apply_translation(-(mesh2.bounds[0]+mesh2.bounds[1])/2.0)
    ext = float(np.max(mesh2.extents))
    if ext > 1e-7: mesh2.apply_scale(2.0/ext)
    mesh2.export(sys.argv[1])
''')

_GT_TMPL_RAW = textwrap.dedent('''\
    import sys
    import cadquery as cq
    import trimesh
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get("result") or locals().get("r")
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    open(sys.argv[1],'wb').write(__import__('trimesh').exchange.stl.export_stl(mesh))
''')


def _exec_to_stl(code: str, normalize: bool, timeout: float = 60.0):
    tmpl = _GT_TMPL_NORM if normalize else _GT_TMPL_RAW
    script = tmpl.format(code=code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    try:
        r = subprocess.run([sys.executable, '-c', script, stl],
                           capture_output=True, timeout=timeout)
        if r.returncode != 0 or Path(stl).stat().st_size < 100:
            Path(stl).unlink(missing_ok=True); return None
        return stl
    except Exception:
        Path(stl).unlink(missing_ok=True); return None


def _render_8view(stl_path: str):
    """Render an STL file as CADEvolve's 8-view 476×952 collage (PIL Image)."""
    from experiments.cadevolve.render import render_stl
    return render_stl(stl_path)


def _score(gen_code: str, gt_stl_path: str, timeout: float = 32.0) -> dict:
    from common.metrics import compute_metrics
    iou_r, cd = compute_metrics(gen_code, gt_stl_path, timeout=timeout, use_pool=False)
    if iou_r == -1.0:
        return {'error_type': 'runtime_error', 'iou': None, 'cd': None}
    return {'error_type': 'success' if iou_r > 0 else 'zero_iou',
            'iou': round(iou_r, 4),
            'cd':  round(cd, 6) if cd is not None else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',  required=True,
                    choices=['deepcad', 'fusion360', 'cad_bench'])
    ap.add_argument('--ckpt',     default='checkpoints/cadevolve-rl1')
    ap.add_argument('--base-model', default='Qwen/Qwen2-VL-2B-Instruct')
    ap.add_argument('--n-samples', type=int, default=300)
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers',  type=int, default=4)
    ap.add_argument('--out',      default=None,
                    help='Default: eval_outputs/repro_official/<dataset>_n<N>/cadevolve')
    args = ap.parse_args()

    # ── Build the (stem, render_fn → image, gt_stl_path) iterator ───────────
    if args.dataset in ('deepcad', 'fusion360'):
        stl_dir = Path('data/deepcad_test_mesh' if args.dataset == 'deepcad'
                       else 'data/fusion360_test_mesh')
        stls = sorted(p for p in stl_dir.iterdir() if p.suffix.lower() == '.stl')
        rng = random.Random(args.seed); rng.shuffle(stls)
        stls = stls[:args.n_samples]
        rows = [{'stem': p.stem, 'gt_stl': str(p), 'gt_code': None} for p in stls]
        out_dir = Path(args.out or f'eval_outputs/repro_official/{args.dataset}_n{args.n_samples}/cadevolve')
    elif args.dataset == 'cad_bench':
        from datasets import load_dataset
        ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                          token=os.environ['HF_TOKEN'])
        rows = [{'stem': r['stem'], 'gt_stl': None, 'gt_code': r['gt_code']}
                for r in ds]
        out_dir = Path(args.out or 'eval_outputs/repro_official/cad_bench_722_full/cadevolve')

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'

    print(f'=== CADEvolve on {args.dataset} (n={len(rows)}, transformers 4.50.3) ===',
          flush=True)

    # ── Load model + processor (vanilla Qwen2VL, no Cadrille mixin) ─────────
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    # transformers 4.50.3 bug: Qwen2VLConfig.get_text_config(decoder=True) returns
    # a plain dict instead of a config object, breaking GenerationConfig
    # construction. Patch it to return self.
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    _orig = Qwen2VLConfig.get_text_config
    Qwen2VLConfig.get_text_config = lambda self, **kw: self

    print(f'loading model from {args.ckpt} …', flush=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map='auto')
    finally:
        Qwen2VLConfig.get_text_config = _orig
    # Tie lm_head to embed_tokens (CADEvolve's training trick — lm_head not saved)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = model.model.embed_tokens.weight
        print('  lm_head tied to embed_tokens', flush=True)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=200704, max_pixels=1003520 * 4, padding_side='left')

    PROMPT = 'Generate CadQuery Python code for this 3D CAD model shown in multiple views.'

    # ── Inference loop ──────────────────────────────────────────────────────
    pool = ThreadPoolExecutor(max_workers=args.score_workers)
    pending = []
    mf = open(meta_path, 'a')

    def _flush(wait_all=False):
        keep = []
        for fut, base in list(pending):
            if not wait_all and not fut.done():
                keep.append((fut, base)); continue
            score = fut.result()
            mf.write(json.dumps({**base, **score}) + '\n'); mf.flush()
        pending.clear(); pending.extend(keep)

    t0 = time.time()
    n_done = 0
    batch = []

    def _drain(batch):
        if not batch: return
        # Build 8-view images per row (slow path: each STL → render_stl ≈ 1s)
        msgs = []
        for row in batch:
            if row.get('gt_stl'):
                stl = row['gt_stl']
                tmp = None
            else:
                # cad_bench_722: exec gt_code → raw STL (no normalization for render)
                stl = _exec_to_stl(row['gt_code'], normalize=False)
                tmp = stl
            try:
                img = _render_8view(stl) if stl else None
            except Exception as e:
                print(f'  render error {row["stem"]}: {e}', flush=True)
                img = None
            if tmp:
                Path(tmp).unlink(missing_ok=True)
            if img is None:
                # fallback: skip — score as gt_render_fail
                row['_render_fail'] = True
                continue
            row['_img'] = img
            msgs.append(row)
        if not msgs:
            return

        chat = [[{'role':'user','content':[
            {'type':'image','image': r['_img']},
            {'type':'text','text': PROMPT}]}] for r in msgs]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in chat]
        vis, _ = process_vision_info(chat)
        inp = processor(text=texts, images=vis, return_tensors='pt',
                        padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=model.config.eos_token_id)
        plen = inp['input_ids'].shape[1]
        for i, row in enumerate(msgs):
            gen = processor.decode(out[i, plen:], skip_special_tokens=True)
            (out_dir / f'{row["stem"]}.py').write_text(gen)
            base = {'stem': row['stem'], 'code_len': len(gen)}
            # determine gt_stl path for scoring (need normalized STL)
            if row.get('gt_stl'):
                gt_stl = row['gt_stl']  # already normalized for deepcad/fusion360
            else:
                gt_stl = _exec_to_stl(row['gt_code'], normalize=True)
            if gt_stl is None:
                mf.write(json.dumps({**base, 'error_type': 'gt_exec_fail',
                                     'iou': None, 'cd': None}) + '\n')
                mf.flush(); continue
            def _do_score(gen_code, gt_stl, base):
                s = _score(gen_code, gt_stl)
                # cleanup tmp gt_stl (only for cad_bench)
                if base.get('_tmp_gt'):
                    Path(gt_stl).unlink(missing_ok=True)
                return s
            base['_tmp_gt'] = (row.get('gt_code') is not None)
            pending.append((pool.submit(_do_score, gen, gt_stl, base), base))

    for i, row in enumerate(rows):
        batch.append(row)
        if len(batch) >= args.batch_size:
            _drain(batch); batch.clear()
            _flush(wait_all=False)
            n_done += args.batch_size
            if n_done % 50 == 0 or n_done == len(rows):
                print(f'  [{n_done}/{len(rows)}] {time.time()-t0:.0f}s elapsed',
                      flush=True)
    if batch:
        _drain(batch)
    _flush(wait_all=True)
    mf.close(); pool.shutdown(wait=True)

    # Aggregate
    rs = [json.loads(l) for l in open(meta_path)]
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    print(f'\n=== CADEvolve on {args.dataset} ===')
    print(f'  n: {len(rs)}')
    print(f'  exec_rate: {len(ok)}/{len(rs)} = {len(ok)/len(rs)*100:.1f}%')
    if ious:
        print(f'  mean_iou: {sum(ious)/len(ious):.4f} (over {len(ious)})')
    if cds:
        print(f'  mean_cd:  {sum(cds)/len(cds):.4f}')


if __name__ == '__main__':
    main()
