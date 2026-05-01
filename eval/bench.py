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

from common.model import Cadrille, get_cadrille_class, collate  # noqa: E402
from common.metrics import compute_metrics   # noqa: E402

_N_POINTS    = 256
_DESCRIPTION = 'Generate cadquery code'

# CadEvolve prompt (standard Qwen2-VL, image input)
_CADEVOLVE_PROMPT = 'Generate CadQuery Python code for this 3D CAD model shown in multiple views.'

# Zero-shot prompt for off-the-shelf VLMs (e.g. Qwen2.5-VL-3B). The model has
# not seen cadquery during training. We deliberately omit a code example here
# because Qwen2.5-VL-3B has been observed to copy the example verbatim instead
# of looking at the image.
_ZS_PROMPT = (
    "Look at the 3D CAD model rendered in this image and write a complete "
    "Python script using the cadquery library that reproduces this exact "
    "geometry. Match the visible shape, dimensions, holes, fillets, and "
    "features as closely as possible.\n\n"
    "Strict output rules:\n"
    "- Start the script with `import cadquery as cq`\n"
    "- Bind the final shape to a variable named exactly `result`\n"
    "- Output ONLY runnable Python — no prose, no markdown code fences, "
    "no explanation before or after the code\n"
    "- Use real numeric dimensions you can read or estimate from the image\n"
    "- Use cadquery operations like Workplane, box, cylinder, sphere, "
    "extrude, revolve, hole, fillet, chamfer, cut, union as appropriate"
)


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences and surrounding prose
    that VLMs sometimes wrap around generated code."""
    import re
    # Strip leading prose up to first ```python or ``` block
    m = re.search(r'```(?:python|cadquery)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Single fence at start with no closing → strip the opening fence
    text = re.sub(r'^\s*```(?:python|cadquery)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()

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
    prompt: str = _CADEVOLVE_PROMPT,
) -> dict:
    """Eval loop for single-image VLMs: uses composite_png, no point cloud.

    Used by both `cadevolve` (trained on cadquery) and `qwen25vl_zs` (zero-shot
    off-the-shelf VLM with strong cadquery prompt). The only thing that differs
    is the user prompt → take it as a kwarg.
    """
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
                    {'type': 'text', 'text': prompt},
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
                eos_token_id=getattr(model.config, 'eos_token_id', None) or model.config.text_config.eos_token_id,
            )

        prompt_len = inputs['input_ids'].shape[1]
        for i, row in enumerate(batch):
            stem = row['stem']
            gen_code = processor.decode(out_ids[i, prompt_len:], skip_special_tokens=True)
            # Off-the-shelf VLMs almost always wrap output in markdown fences;
            # strip them so the code is directly executable.
            gen_code = _strip_code_fences(gen_code)
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
                eos_token_id=getattr(model.config, 'eos_token_id', None) or model.config.text_config.eos_token_id,
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
    ap.add_argument('--ckpt',         required=False, default=None,
                    help='Checkpoint path (local dir). Optional for --model-type qwen25vl_zs '
                         '— in that case --base-model is loaded directly from HF.')
    ap.add_argument('--model-type',   default='cadrille',
                    choices=['cadrille', 'cadevolve', 'qwen25vl_zs'],
                    help='Model: cadrille (point cloud + image), cadevolve (Qwen2VL image), '
                         'or qwen25vl_zs (Qwen2.5-VL zero-shot, off-the-shelf, no ckpt needed)')
    ap.add_argument('--base-model',   default='Qwen/Qwen2-VL-2B-Instruct',
                    help='Base model id for processor (default: Qwen/Qwen2-VL-2B-Instruct). '
                         'For qwen25vl_zs, set this to e.g. Qwen/Qwen2.5-VL-3B-Instruct. '
                         'For backbone=qwen3_vl, the processor lives in the ckpt itself, '
                         'so this is ignored.')
    ap.add_argument('--backbone',     default='qwen2_vl',
                    choices=['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl'],
                    help='Cadrille backbone family. Only used when --model-type cadrille. '
                         'qwen3_vl matches the Cadrille_Qwen3VLForConditionalGeneration '
                         'architecture from `--config` v3 SFT runs.')
    ap.add_argument('--split',        default='test_iid',
                    help='Split name. Use "all" for the standard '
                         'test_iid+test_ood_family+test_ood_plane triple. '
                         'Any other string is treated as a single literal split name '
                         '(e.g. "train" for BenchCAD/cad_bench_722).')
    ap.add_argument('--limit',        type=int, default=0,  help='Max samples per split (0=all)')
    ap.add_argument('--seed',         type=int, default=42,
                    help='Random seed for --limit shuffle (default: 42). Same seed = same samples.')
    ap.add_argument('--batch-size',   type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--score-workers',  type=int, default=4)
    ap.add_argument('--attn-impl',    default='sdpa',
                    choices=['sdpa', 'flash_attention_2', 'eager'],
                    help='Attention implementation (default: sdpa — works without flash-attn).')
    ap.add_argument('--out',          required=True, help='Output directory')
    ap.add_argument('--hf-repo',      default='Hula0401/test_bench')
    ap.add_argument('--label',        default=None,
                    help='Human-readable label for report (default: ckpt basename)')
    args = ap.parse_args()

    out_dir    = Path(args.out)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        ckpt_label = args.label or ckpt_path.name
    else:
        if args.model_type != 'qwen25vl_zs':
            print('ERROR: --ckpt is required unless --model-type qwen25vl_zs', file=sys.stderr)
            sys.exit(1)
        ckpt_path = None
        ckpt_label = args.label or args.base_model.split('/')[-1] + '-zs'

    # ── Load dataset ────────────────────────────────────────────────────────
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    print(f'Loading {args.hf_repo} ...', flush=True)
    ds = load_dataset(args.hf_repo, token=token)

    splits = ['test_iid', 'test_ood_family', 'test_ood_plane'] if args.split == 'all' else [args.split]
    rows: list[dict] = []
    for sp in splits:
        if sp not in ds:
            print(f'ERROR: split "{sp}" not in dataset {args.hf_repo}. '
                  f'Available: {list(ds.keys())}', file=sys.stderr)
            sys.exit(1)
        sp_rows = list(ds[sp])
        # Inject 'split' field for datasets that don't carry one (e.g. cad_bench_722
        # only has a single 'train' split). Downstream report grouping needs it.
        for r in sp_rows:
            r.setdefault('split', sp)
            r.setdefault('feature_tags', None)
            r.setdefault('feature_count', None)
        if args.limit:
            import random as _random
            rng = _random.Random(args.seed)
            rng.shuffle(sp_rows)
            sp_rows = sp_rows[:args.limit]
        rows.extend(sp_rows)

    print(f'Total samples: {len(rows)} across splits: {splits}', flush=True)

    # ── Load model ──────────────────────────────────────────────────────────
    if ckpt_path is not None and not (ckpt_path / 'model.safetensors').exists():
        # Try sharded variant
        shards = list(ckpt_path.glob('model-*-of-*.safetensors'))
        if not shards:
            print(f'ERROR: no model weights found at {ckpt_path}', file=sys.stderr)
            sys.exit(1)

    proc_src = args.base_model if args.model_type != 'qwen25vl_zs' else args.base_model
    print(f'Loading processor from {proc_src} ...', flush=True)
    processor = AutoProcessor.from_pretrained(
        proc_src,
        min_pixels=200704,
        max_pixels=1003520,
        padding_side='left',
    )

    print(f'Loading model ({args.model_type}) from '
          f'{ckpt_path or args.base_model} ...', flush=True)
    if args.model_type == 'qwen25vl_zs':
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl,
            device_map='cuda',
        )
    elif args.model_type == 'cadevolve':
        from transformers import Qwen2VLForConditionalGeneration
        # Note: the old transformers 4.50.3 monkey-patch on get_text_config
        # broke under transformers 5.x — removed.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl,
            device_map='cuda',
        )
        # lm_head.weight is not saved in checkpoint — tie to embed_tokens.
        # transformers 5.x removed `model.model.embed_tokens`; use get_input_embeddings.
        _embed = model.get_input_embeddings()
        if model.lm_head.weight.data_ptr() != _embed.weight.data_ptr():
            model.lm_head.weight = _embed.weight
            print('  lm_head tied to embed_tokens.', flush=True)
    else:
        # Cadrille — pick the right backbone-flavoured class.
        # For Qwen3-VL the processor lives inside the ckpt (saved by trainer),
        # not in the public --base-model id, so re-load it from ckpt_path.
        cadrille_cls = get_cadrille_class(args.backbone)
        if args.backbone == 'qwen3_vl':
            print(f'  re-loading processor from {ckpt_path} (Qwen3-VL ships in-ckpt) …',
                  flush=True)
            processor = AutoProcessor.from_pretrained(
                str(ckpt_path),
                min_pixels=200704,
                max_pixels=1003520,
                padding_side='left',
            )
        model = cadrille_cls.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl,
            device_map='cuda',
        )
    model.eval()
    print('Model loaded.', flush=True)

    # ── Run eval ────────────────────────────────────────────────────────────
    print(f'\nRunning eval → {out_dir}', flush=True)
    if args.model_type in ('cadevolve', 'qwen25vl_zs'):
        prompt = _ZS_PROMPT if args.model_type == 'qwen25vl_zs' else _CADEVOLVE_PROMPT
        summary = run_bench_cadevolve(
            rows=rows,
            model=model,
            processor=processor,
            out_dir=out_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            score_workers=args.score_workers,
            prompt=prompt,
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
