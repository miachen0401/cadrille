"""Evaluate Cadrille / CadEvolve on the HF bench track (Hula0401/test_bench).

Metrics per sample:
  iou          - volumetric IoU (trimesh boolean, normalised [-1,1]³)
  cd           - Chamfer Distance (8192 surface pts, normalised [-1,1]³)
  error_type   - success | runtime_error | syntax_error | timeout | gt_exec_fail

Usage:
    # Cadrille SFT
    python3 -m tools.eval_bench \\
        --ckpt checkpoints/cadrille-sft \\
        --split all --limit 100 --seed 42 \\
        --out eval_outputs/bench/sft_n300

    # Cadrille RL
    python3 -m tools.eval_bench \\
        --ckpt checkpoints/cadrille-rl \\
        --split all --limit 100 --seed 42 --batch-size 2 \\
        --out eval_outputs/bench/rl_n300

    # CadEvolve (Qwen2-VL, image input, no point cloud)
    python3 -m tools.eval_bench \\
        --ckpt checkpoints/cadevolve-rl1 \\
        --model-type cadevolve \\
        --split all --limit 100 --seed 42 --batch-size 2 \\
        --out eval_outputs/bench/cadevolve_n300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoProcessor

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from cadrille import Cadrille, collate  # noqa: E402
from common.metrics import compute_metrics   # noqa: E402

_N_POINTS    = 256
_DESCRIPTION = 'Generate cadquery code'

# CadEvolve prompt (standard Qwen2-VL, image input)
_CADEVOLVE_PROMPT = 'Generate CadQuery Python code for this 3D CAD model shown in multiple views.'

# ---------------------------------------------------------------------------
# GT code execution  →  temp STL
# ---------------------------------------------------------------------------
# GT code uses `result` as variable name and may include show_object(); we
# normalise it here before passing to the reward worker (which expects `r`).

_GT_EXEC_TEMPLATE = textwrap.dedent('''\
    import sys, io
    import cadquery as cq
    import trimesh
    import numpy as np
    show_object = lambda *a, **kw: None

    {code}

    # bench GT code uses `result`; reward worker convention uses `r`
    _res = locals().get('result') or locals().get('r')
    if _res is None:
        raise ValueError('no result variable found in GT code')

    compound = _res.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
    buf = trimesh.exchange.stl.export_stl(mesh)
    mesh2 = trimesh.load(io.BytesIO(buf), file_type="stl", force="mesh")
    mesh2.apply_translation(-(mesh2.bounds[0] + mesh2.bounds[1]) / 2.0)
    ext = float(np.max(mesh2.extents))
    if ext > 1e-7:
        mesh2.apply_scale(2.0 / ext)
    mesh2.export(sys.argv[1])
''')

# Env with LD_LIBRARY_PATH so cadquery/OCP can find libGL etc.
_LD = os.environ.get('LD_LIBRARY_PATH', '/workspace/.local/lib')


def _exec_gt_code(gt_code: str, timeout: float = 60.0) -> str | None:
    """Execute GT CadQuery code → normalised STL. Returns path or None on error."""
    script = _GT_EXEC_TEMPLATE.format(code=gt_code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl_path = f.name
    env = {**os.environ, 'LD_LIBRARY_PATH': _LD}
    try:
        r = subprocess.run(
            [sys.executable, '-c', script, stl_path],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if r.returncode == 0 and Path(stl_path).stat().st_size > 100:
            return stl_path
        return None
    except Exception:
        return None
    finally:
        pass  # caller is responsible for cleanup



# ---------------------------------------------------------------------------
# Scoring  (gen code + gt_code → iou, cd)
# ---------------------------------------------------------------------------

def _score(gen_code: str, gt_code: str, timeout: float = 32.0) -> dict:
    """Execute GT code → temp STL, score gen_code against it, delete STL immediately.

    GT STL is created and deleted within this call so at most score_workers
    STL files exist at any one time (no /tmp accumulation over the full run).
    """
    gt_stl = _exec_gt_code(gt_code, timeout=60.0)
    if gt_stl is None:
        return {'error_type': 'gt_exec_fail', 'iou': None, 'cd': None}
    try:
        iou_reward, cd = compute_metrics(gen_code, gt_stl, timeout=timeout, use_pool=False)
    finally:
        Path(gt_stl).unlink(missing_ok=True)
    if iou_reward == -1.0:
        return {'error_type': 'runtime_error', 'iou': None, 'cd': None}
    return {
        'error_type': 'success' if iou_reward > 0 else 'zero_iou',
        'iou': round(iou_reward, 4),
        'cd':  round(cd, 6) if cd is not None else None,
    }


# ---------------------------------------------------------------------------
# CadEvolve inference (standard Qwen2VL, image input)
# ---------------------------------------------------------------------------

def run_bench_cadevolve(
    rows: list[dict],
    model,
    processor,
    out_dir: Path,
    batch_size: int = 4,
    max_new_tokens: int = 768,
    score_workers: int = 4,
    save_code: bool = True,
) -> dict:
    """Eval loop for CadEvolve: uses composite_png as single image, no point cloud."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'

    done_stems: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    done_stems.add(json.loads(line)['stem'])
                except Exception:
                    pass

    todo = [r for r in rows if r['stem'] not in done_stems]
    if not todo:
        print(f'  All {len(done_stems)} cases already done.', flush=True)
        return _summarize(meta_path)

    print(f'  {len(todo)} to run, {len(done_stems)} already done', flush=True)

    device = next(model.parameters()).device
    meta_file = open(meta_path, 'a')
    score_pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []

    def _flush_pending(wait_all: bool = False) -> None:
        remaining = []
        for fut, stem, base_rec in list(pending):
            if not wait_all and not fut.done():
                remaining.append((fut, stem, base_rec))
                continue
            score = fut.result()
            rec = {**base_rec, **score}
            meta_file.write(json.dumps(rec) + '\n')
            meta_file.flush()
        pending.clear()
        pending.extend(remaining)

    def _drain_batch(batch: list[dict]) -> None:
        if not batch:
            return

        messages = []
        for row in batch:
            img = row['composite_png']  # PIL Image
            messages.append([{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': _CADEVOLVE_PROMPT},
                ],
            }])

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=model.config.eos_token_id,
            )

        prompt_len = inputs['input_ids'].shape[1]
        for i, row in enumerate(batch):
            stem = row['stem']
            gen_code = processor.decode(out_ids[i, prompt_len:], skip_special_tokens=True)
            if save_code:
                (out_dir / f'{stem}.py').write_text(gen_code)
            base_rec = {
                'stem':          stem,
                'family':        row['family'],
                'difficulty':    row['difficulty'],
                'base_plane':    row['base_plane'],
                'split':         row['split'],
                'feature_tags':  row['feature_tags'],
                'feature_count': row['feature_count'],
                'code_len':      len(gen_code),
            }
            fut = score_pool.submit(_score, gen_code, row['gt_code'])
            pending.append((fut, stem, base_rec))

        _flush_pending(wait_all=False)

    batch: list[dict] = []
    for i, row in enumerate(todo):
        batch.append(row)
        if len(batch) >= batch_size:
            print(f'  [{i+1}/{len(todo)}] batch of {len(batch)} ...', end=' ', flush=True)
            _drain_batch(batch)
            batch.clear()
            print('done', flush=True)

    if batch:
        print(f'  [{len(todo)}/{len(todo)}] final batch of {len(batch)} ...', end=' ', flush=True)
        _drain_batch(batch)
        print('done', flush=True)

    _flush_pending(wait_all=True)
    meta_file.close()
    score_pool.shutdown(wait=True)
    return _summarize(meta_path)


# ---------------------------------------------------------------------------
# Inference batch loop
# ---------------------------------------------------------------------------

def run_bench(
    rows: list[dict],
    model,
    processor,
    out_dir: Path,
    batch_size: int = 4,
    max_new_tokens: int = 768,
    score_workers: int = 4,
    save_code: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'

    done_stems: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    done_stems.add(json.loads(line)['stem'])
                except Exception:
                    pass

    todo = [r for r in rows if r['stem'] not in done_stems]
    if not todo:
        print(f'  All {len(done_stems)} cases already done.', flush=True)
        return _summarize(meta_path)

    print(f'  {len(todo)} to run, {len(done_stems)} already done', flush=True)

    device = next(model.parameters()).device

    meta_file = open(meta_path, 'a')
    score_pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []  # list of (future, stem, meta_row)

    def _flush_pending(wait_all: bool = False) -> None:
        remaining = []
        for fut, stem, base_rec in list(pending):
            if not wait_all and not fut.done():
                remaining.append((fut, stem, base_rec))
                continue
            score = fut.result()
            rec = {**base_rec, **score}
            meta_file.write(json.dumps(rec) + '\n')
            meta_file.flush()
        pending.clear()
        pending.extend(remaining)

    def _drain_batch(batch: list[dict]) -> None:
        if not batch:
            return

        # Reset rope_deltas to avoid KV cache contamination across batches
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None

        # Build collate items: video + description (same format as rl/dataset.py)
        collate_items = [
            {
                'video': [row['composite_png']],
                'description': _DESCRIPTION,
                'file_name': row['stem'],
            }
            for row in batch
        ]

        b = collate(collate_items, processor, _N_POINTS, eval=True)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=b['input_ids'].to(device),
                attention_mask=b['attention_mask'].to(device),
                point_clouds=b['point_clouds'].to(device),
                is_pc=b['is_pc'].to(device),
                is_img=b['is_img'].to(device),
                pixel_values_videos=(
                    b['pixel_values_videos'].to(device)
                    if b.get('pixel_values_videos') is not None else None
                ),
                video_grid_thw=(
                    b['video_grid_thw'].to(device)
                    if b.get('video_grid_thw') is not None else None
                ),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=model.config.eos_token_id,
            )

        prompt_len = b['input_ids'].shape[1]
        for i, row in enumerate(batch):
            stem = row['stem']
            gen_code = processor.decode(out_ids[i, prompt_len:], skip_special_tokens=True)

            if save_code:
                (out_dir / f'{stem}.py').write_text(gen_code)

            base_rec = {
                'stem':          stem,
                'family':        row['family'],
                'difficulty':    row['difficulty'],
                'base_plane':    row['base_plane'],
                'split':         row['split'],
                'feature_tags':  row['feature_tags'],
                'feature_count': row['feature_count'],
                'code_len':      len(gen_code),
            }

            fut = score_pool.submit(_score, gen_code, row['gt_code'])
            pending.append((fut, stem, base_rec))

        _flush_pending(wait_all=False)

    # ── Main loop ──────────────────────────────────────────────────────────
    batch: list[dict] = []
    for i, row in enumerate(todo):
        batch.append(row)
        if len(batch) >= batch_size:
            print(
                f'  [{i+1}/{len(todo)}] batch of {len(batch)} ...', end=' ', flush=True
            )
            _drain_batch(batch)
            batch.clear()
            print('done', flush=True)

    if batch:
        print(f'  [{len(todo)}/{len(todo)}] final batch of {len(batch)} ...', end=' ', flush=True)
        _drain_batch(batch)
        print('done', flush=True)

    _flush_pending(wait_all=True)
    meta_file.close()
    score_pool.shutdown(wait=True)

    return _summarize(meta_path)


# ---------------------------------------------------------------------------
# Summary / report
# ---------------------------------------------------------------------------

def _summarize(meta_path: Path) -> dict:
    records = []
    with open(meta_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return {}

    total = len(records)
    success = [r for r in records if r.get('error_type') == 'success']
    ious  = [r['iou'] for r in success if r.get('iou') is not None]
    cds   = [r['cd']  for r in success if r.get('cd')  is not None]

    return {
        'n':           total,
        'exec_rate':   round(len(success) / total, 4) if total else 0.0,
        'mean_iou':    round(sum(ious) / len(ious), 4) if ious else 0.0,
        'mean_cd':     round(sum(cds)  / len(cds),  6) if cds  else None,
    }


def _report(meta_path: Path, ckpt_label: str) -> None:
    records = []
    with open(meta_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        print('No results.'); return

    total   = len(records)
    success = [r for r in records if r.get('error_type') == 'success']
    ious    = [r['iou'] for r in success if r.get('iou') is not None]
    cds     = [r['cd']  for r in success if r.get('cd')  is not None]

    print(f'\n{"="*62}')
    print(f'Checkpoint : {ckpt_label}')
    print(f'N          : {total}')
    print(f'Exec rate  : {len(success)/total*100:.1f}%  ({len(success)}/{total})')
    print(f'Mean IoU   : {sum(ious)/len(ious):.4f}  (n={len(ious)})' if ious else 'Mean IoU   : —')
    print(f'Mean CD    : {sum(cds)/len(cds):.6f}  (n={len(cds)})' if cds else 'Mean CD    : —')

    # By split
    by_split: dict[str, list] = defaultdict(list)
    for r in records:
        by_split[r.get('split', '?')].append(r)

    print(f'\n{"Split":<22} {"N":>5} {"Exec%":>7} {"IoU":>7} {"CD":>10}')
    print('-' * 56)
    for sp in sorted(by_split):
        rs  = by_split[sp]
        ok  = [x for x in rs if x.get('error_type') == 'success']
        iou = [x['iou'] for x in ok if x.get('iou') is not None]
        cd  = [x['cd']  for x in ok if x.get('cd')  is not None]
        print(
            f'{sp:<22} {len(rs):>5} {len(ok)/len(rs)*100:>6.1f}%'
            f' {sum(iou)/len(iou):>7.4f}' if iou else f'{sp:<22} {len(rs):>5} {len(ok)/len(rs)*100:>6.1f}%      —',
            f'{sum(cd)/len(cd):>10.6f}' if cd else '',
        )

    # By difficulty
    by_diff: dict[str, list] = defaultdict(list)
    for r in records:
        by_diff[r.get('difficulty', '?')].append(r)

    print(f'\n{"Difficulty":<12} {"N":>5} {"Exec%":>7} {"IoU":>7}')
    print('-' * 35)
    for diff in ['easy', 'medium', 'hard']:
        rs = by_diff.get(diff, [])
        if not rs: continue
        ok  = [x for x in rs if x.get('error_type') == 'success']
        iou = [x['iou'] for x in ok if x.get('iou') is not None]
        print(
            f'{diff:<12} {len(rs):>5} {len(ok)/len(rs)*100:>6.1f}%'
            f' {sum(iou)/len(iou):>7.4f}' if iou else f'{diff:<12} {len(rs):>5} {len(ok)/len(rs)*100:>6.1f}%      —'
        )

    print('=' * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Eval Cadrille on HF bench track (Hula0401/test_bench)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--ckpt',         required=True,  help='Checkpoint path (local dir)')
    ap.add_argument('--model-type',   default='cadrille', choices=['cadrille', 'cadevolve'],
                    help='Model architecture: cadrille (default) or cadevolve (Qwen2VL, image input)')
    ap.add_argument('--base-model',   default='Qwen/Qwen2-VL-2B-Instruct',
                    help='Base model id for processor (default: Qwen/Qwen2-VL-2B-Instruct)')
    ap.add_argument('--split',        default='test_iid',
                    choices=['test_iid', 'test_ood_family', 'test_ood_plane', 'all'])
    ap.add_argument('--limit',        type=int, default=0,  help='Max samples per split (0=all)')
    ap.add_argument('--seed',         type=int, default=42,
                    help='Random seed for --limit shuffle (default: 42). Same seed = same samples.')
    ap.add_argument('--batch-size',   type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers',  type=int, default=4)
    ap.add_argument('--out',          required=True, help='Output directory')
    ap.add_argument('--hf-repo',      default='Hula0401/test_bench')
    ap.add_argument('--label',        default=None,
                    help='Human-readable label for report (default: ckpt basename)')
    args = ap.parse_args()

    ckpt_path  = Path(args.ckpt)
    out_dir    = Path(args.out)
    ckpt_label = args.label or ckpt_path.name

    # ── Load dataset ────────────────────────────────────────────────────────
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    print(f'Loading {args.hf_repo} ...', flush=True)
    ds = load_dataset(args.hf_repo, token=token)

    splits = ['test_iid', 'test_ood_family', 'test_ood_plane'] if args.split == 'all' else [args.split]
    rows: list[dict] = []
    for sp in splits:
        sp_rows = list(ds[sp])
        if args.limit:
            import random as _random
            rng = _random.Random(args.seed)
            rng.shuffle(sp_rows)
            sp_rows = sp_rows[:args.limit]
        rows.extend(sp_rows)

    print(f'Total samples: {len(rows)} across splits: {splits}', flush=True)

    # ── Load model ──────────────────────────────────────────────────────────
    if not (ckpt_path / 'model.safetensors').exists():
        # Try sharded variant
        shards = list(ckpt_path.glob('model-*-of-*.safetensors'))
        if not shards:
            print(f'ERROR: no model weights found at {ckpt_path}', file=sys.stderr)
            sys.exit(1)

    print(f'Loading processor from {args.base_model} ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=200704,
        max_pixels=1003520,
        padding_side='left',
    )

    print(f'Loading model ({args.model_type}) from {ckpt_path} ...', flush=True)
    if args.model_type == 'cadevolve':
        from transformers import Qwen2VLForConditionalGeneration
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
        # Workaround: in transformers 4.50.3, get_text_config(decoder=True) returns a plain
        # dict for Qwen2VL, causing AttributeError in GenerationConfig.from_model_config.
        # Patch it to return self so the conditional branch is skipped.
        _orig_get_text_config = Qwen2VLConfig.get_text_config
        Qwen2VLConfig.get_text_config = lambda self, **kw: self
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(ckpt_path),
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2',
                device_map='cuda',
            )
        finally:
            Qwen2VLConfig.get_text_config = _orig_get_text_config
        # lm_head.weight is not saved in checkpoint — tie to embed_tokens
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            model.lm_head.weight = model.model.embed_tokens.weight
            print('  lm_head tied to embed_tokens.', flush=True)
    else:
        model = Cadrille.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map='cuda',
        )
    model.eval()
    print('Model loaded.', flush=True)

    # ── Run eval ────────────────────────────────────────────────────────────
    print(f'\nRunning eval → {out_dir}', flush=True)
    if args.model_type == 'cadevolve':
        summary = run_bench_cadevolve(
            rows=rows,
            model=model,
            processor=processor,
            out_dir=out_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            score_workers=args.score_workers,
        )
    else:
        summary = run_bench(
            rows=rows,
            model=model,
            processor=processor,
            out_dir=out_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            score_workers=args.score_workers,
        )
    print(f'\nSummary: {json.dumps(summary, indent=2)}')

    # ── Report ──────────────────────────────────────────────────────────────
    _report(out_dir / 'metadata.jsonl', ckpt_label)



if __name__ == '__main__':
    main()
