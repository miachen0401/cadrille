"""100-case trajectory sweep — generate code per ckpt × anchor at scale.

Picks 100 anchors (33 BenchCAD val + 33 DeepCAD test + 34 Fusion360 test) by
fixed seed. For each of 7 ckpts (1k, 4k, 10k, 11k, 15k, 18k, 20k) generates
greedy code, renders mesh, computes IoU. Produces:

  - results.jsonl       — full per-(case, ckpt) data
  - collage_<dset>.png  — per-dataset grid: rows=cases, cols=GT|each ckpt
  - trajectory_iou.png  — line plot of all 100 cases + mean overlay
  - boxplot_iou.png     — IoU distribution per ckpt × dataset
  - forget_matrix.png   — heatmap: per-case Δ-IoU between consecutive ckpts
  - report.md           — narrative + tables + image embeds

Usage:
  uv run python -m scripts.analysis.trajectory_100case \
    --ckpt-dirs /ephemeral/checkpoints/curriculum_best_from_hf \
                /ephemeral/checkpoints/sft-s20k-lr2e-4-b8a4-img-0425-1929 \
    --steps 1000,4000,10000,11000,15000,18000,20000 \
    --out eval_outputs/trajectory_100case_curriculum
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from common.model import collate, get_cadrille_class
from common.metrics import compute_metrics
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Anchor loading
# ---------------------------------------------------------------------------
def load_anchors(seed: int = 42) -> list[dict]:
    """100 anchors: 33 BenchCAD val + 33 DeepCAD test + 34 Fusion360 test."""
    rng = np.random.default_rng(seed)
    out = []

    # BenchCAD val
    bc_root = Path('data/benchcad')
    with (bc_root / 'val.pkl').open('rb') as f:
        rows = pickle.load(f)
    rows_shuffled = rng.permutation(rows).tolist()
    n_added = 0
    for r in rows_shuffled:
        py = bc_root / r['py_path']
        png = bc_root / r['png_path']
        stl = bc_root / r['mesh_path']
        if not (py.exists() and png.exists() and stl.exists()):
            continue
        out.append({
            'dataset': 'benchcad_val',
            'file_name': r['uid'],
            'gt_mesh_path': str(stl),
            'gt_code': py.read_text(),
            'image_path': str(png),
            'description': 'Generate cadquery code',
        })
        n_added += 1
        if n_added >= 33:
            break

    # DeepCAD test (33)
    for label, root_str, target_n in [
        ('deepcad_test', 'data/deepcad_test_mesh', 33),
        ('fusion360_test', 'data/fusion360_test_mesh', 34),
    ]:
        rootp = Path(root_str)
        stls = sorted(rootp.glob('*.stl'))
        idx = rng.choice(len(stls), size=target_n, replace=False).tolist()
        for i in idx:
            stl = stls[i]
            png = stl.with_name(stl.stem + '_render.png')
            if not png.exists():
                continue
            out.append({
                'dataset': label,
                'file_name': stl.stem,
                'gt_mesh_path': str(stl),
                'gt_code': None,
                'image_path': str(png),
                'description': 'Generate cadquery code',
            })

    print(f'Loaded {len(out)} anchors:')
    for ds in ('benchcad_val', 'deepcad_test', 'fusion360_test'):
        print(f'  {ds}: {sum(1 for a in out if a["dataset"] == ds)}')
    return out


# ---------------------------------------------------------------------------
# Generation (batched)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_batch(model, processor, anchors_batch, max_new_tokens=768, device='cuda'):
    """Greedy generation for a batch of anchors."""
    items = []
    for a in anchors_batch:
        items.append({
            'video': [Image.open(a['image_path']).convert('RGB')],
            'description': a['description'],
            'file_name': a['file_name'],
        })
    batch = collate(items, processor=processor, n_points=256, eval=True)
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None

    gen_kw = dict(
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        point_clouds=batch['point_clouds'].to(device),
        is_pc=batch['is_pc'].to(device),
        is_img=batch['is_img'].to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
        bad_words_ids=[[model.config.video_token_id]],
    )
    if batch.get('pixel_values_videos') is not None:
        gen_kw['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
        gen_kw['video_grid_thw'] = batch['video_grid_thw'].to(device)

    out_ids = model.generate(**gen_kw)
    prompt_len = batch['input_ids'].shape[1]
    return [processor.decode(out_ids[j, prompt_len:], skip_special_tokens=True)
            for j in range(len(anchors_batch))]


# ---------------------------------------------------------------------------
# Rendering worker
# ---------------------------------------------------------------------------
def _render_one(args: tuple) -> tuple[int, int, str]:
    """Worker: render code → PNG. Returns (case_idx, step, status)."""
    case_idx, step, code, png_out, img_size = args
    if os.path.exists(png_out):
        return case_idx, step, 'skip'

    import io
    import trimesh
    import open3d
    import cadquery as cq  # noqa: F401
    from common.datasets import mesh_to_image

    def _on_alarm(signum, frame):
        raise TimeoutError('cadquery exec timeout')

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(15)

    try:
        try:
            code_obj = compile(code, '<string>', 'exec')
        except SyntaxError:
            return case_idx, step, 'syntax'
        captured = {}
        g = {'show_object': lambda obj, *a, **kw: captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        res = g.get('r') or g.get('result') or captured.get('r')
        if res is None:
            return case_idx, step, 'no_r'
        compound = res.val()
        verts, faces = compound.tessellate(0.001, 0.1)
        if len(faces) < 3:
            return case_idx, step, 'empty'
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        # Center+normalize like CadRecodeDataset
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        ext = float(np.max(mesh.extents))
        if ext > 1e-7:
            mesh.apply_scale(1.0 / ext)
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        o3d.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d.compute_vertex_normals()
        img = mesh_to_image(o3d, camera_distance=-1.6, front=[1, 1, 1], img_size=img_size)
        img.save(png_out)
        return case_idx, step, 'ok'
    except TimeoutError:
        return case_idx, step, 'timeout'
    except Exception as e:
        return case_idx, step, f'err: {type(e).__name__}: {str(e)[:60]}'
    finally:
        signal.alarm(0)


def _render_gt(args: tuple) -> tuple[int, str]:
    """Render GT mesh from STL."""
    case_idx, gt_mesh_path, png_out, img_size = args
    if os.path.exists(png_out):
        return case_idx, 'skip'
    try:
        import trimesh, open3d
        from common.datasets import mesh_to_image
        mesh = trimesh.load(gt_mesh_path)
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        ext = float(np.max(mesh.extents))
        if ext > 1e-7:
            mesh.apply_scale(1.0 / ext)
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        o3d.paint_uniform_color(np.array([136, 200, 255]) / 255.0)
        o3d.compute_vertex_normals()
        img = mesh_to_image(o3d, camera_distance=-1.6, front=[1, 1, 1], img_size=img_size)
        img.save(png_out)
        return case_idx, 'ok'
    except Exception as e:
        return case_idx, f'err: {type(e).__name__}: {str(e)[:60]}'


def _iou_one(args: tuple) -> tuple[int, int, float | None]:
    """Compute IoU for (case_idx, step, code, gt_mesh_path)."""
    case_idx, step, code, gt_mesh_path = args
    try:
        iou_r, _ = compute_metrics(code, gt_mesh_path, timeout=15, use_pool=False)
        return case_idx, step, (float(iou_r) if iou_r is not None and iou_r >= 0 else None)
    except Exception:
        return case_idx, step, None


# ---------------------------------------------------------------------------
# Collage builder
# ---------------------------------------------------------------------------
def build_collage(out_root: Path, dataset: str, anchors: list[dict],
                  steps: list[int], cell: int = 140, label_h: int = 22):
    """Build an N-row × (1 GT + 1 input + len(steps)) col collage for one dataset."""
    cases = [a for a in anchors if a['dataset'] == dataset]
    if not cases:
        return None

    n_rows = len(cases)
    n_cols = 2 + len(steps)  # GT mesh | input image | each ckpt
    W = n_cols * cell
    H = n_rows * cell + label_h * 2

    img = Image.new('RGB', (W, H), color=(20, 20, 20))
    drw = ImageDraw.Draw(img)

    # Header row
    headers = ['GT mesh', 'input img'] + [f'step {s}' for s in steps]
    for c, h in enumerate(headers):
        drw.text((c * cell + 4, 2), h, fill=(220, 220, 220))

    for r, anchor in enumerate(cases):
        case_idx = anchor['_case_idx']
        y = label_h + r * cell
        # Row label (file_name short)
        drw.text((4, y + 4), f'{case_idx} {anchor["file_name"][:18]}', fill=(180, 220, 255))

        # Col 0: GT mesh
        gt_path = out_root / 'gt_renders' / f'gt_{case_idx:03d}.png'
        if gt_path.exists():
            try:
                gi = Image.open(gt_path).convert('RGB').resize((cell - 4, cell - 4), Image.LANCZOS)
                img.paste(gi, (0 * cell + 2, y + 2))
            except Exception:
                pass

        # Col 1: input image (what the model sees)
        try:
            input_img = Image.open(anchor['image_path']).convert('RGB').resize(
                (cell - 4, cell - 4), Image.LANCZOS)
            img.paste(input_img, (1 * cell + 2, y + 2))
        except Exception:
            pass

        # Col 2..: pred renders per ckpt
        for ci, s in enumerate(steps):
            pred_path = out_root / 'pred_renders' / f'case{case_idx:03d}_step{s:05d}.png'
            cell_x = (2 + ci) * cell
            if pred_path.exists():
                try:
                    pi = Image.open(pred_path).convert('RGB').resize(
                        (cell - 4, cell - 4), Image.LANCZOS)
                    img.paste(pi, (cell_x + 2, y + 2))
                except Exception:
                    drw.rectangle([cell_x + 2, y + 2, cell_x + cell - 2, y + cell - 2],
                                  fill=(60, 30, 30))
            else:
                drw.rectangle([cell_x + 2, y + 2, cell_x + cell - 2, y + cell - 2],
                              fill=(60, 30, 30))

    out_path = out_root / f'collage_{dataset}.png'
    img.save(out_path)
    print(f'  → {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Trajectory + boxplot via matplotlib
# ---------------------------------------------------------------------------
def build_trajectory_plots(out_root: Path, results: list[dict],
                            anchors: list[dict], steps: list[int]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DSETS = ['benchcad_val', 'deepcad_test', 'fusion360_test']
    DCOLOR = {'benchcad_val': '#ff7f0e', 'deepcad_test': '#1f77b4',
              'fusion360_test': '#2ca02c'}

    by_case = {}  # case_idx -> {step: iou}
    for r in results:
        by_case.setdefault(r['case_idx'], {})[r['step']] = r['iou']

    # 1. Trajectory line plot (all cases, transparent)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for ds in DSETS:
        case_indices = [a['_case_idx'] for a in anchors if a['dataset'] == ds]
        for ci in case_indices:
            ys = [by_case.get(ci, {}).get(s) for s in steps]
            ys = [(y if y is not None else np.nan) for y in ys]
            ax.plot(steps, ys, color=DCOLOR[ds], alpha=0.10, linewidth=1)
    # Mean per dataset
    for ds in DSETS:
        case_indices = [a['_case_idx'] for a in anchors if a['dataset'] == ds]
        means = []
        for s in steps:
            vals = [by_case.get(ci, {}).get(s) for ci in case_indices]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(steps, means, color=DCOLOR[ds], linewidth=2.5,
                label=f'{ds} (n={len(case_indices)}) mean')
    ax.set_xlabel('curriculum step')
    ax.set_ylabel('IoU')
    ax.set_title('Per-case IoU trajectory across curriculum (alpha=0.10 individual; bold=mean)')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_root / 'trajectory_iou.png', dpi=110)
    plt.close(fig)
    print(f'  → trajectory_iou.png')

    # 2. Boxplot per ckpt × dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axi, ds in enumerate(DSETS):
        ax = axes[axi]
        case_indices = [a['_case_idx'] for a in anchors if a['dataset'] == ds]
        data = []
        for s in steps:
            vals = [by_case.get(ci, {}).get(s) for ci in case_indices]
            vals = [v for v in vals if v is not None]
            data.append(vals)
        bp = ax.boxplot(data, tick_labels=[f'{s}' for s in steps],
                         patch_artist=True, showmeans=True)
        for box in bp['boxes']:
            box.set_facecolor(DCOLOR[ds])
            box.set_alpha(0.6)
        ax.set_title(f'{ds}  (n_resolved per ckpt: {[len(d) for d in data]})')
        ax.set_xlabel('step')
        ax.set_ylabel('IoU')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('IoU distribution per ckpt × dataset (boxplot, mean=triangle)')
    fig.tight_layout()
    fig.savefig(out_root / 'boxplot_iou.png', dpi=110)
    plt.close(fig)
    print(f'  → boxplot_iou.png')

    # 3. Forgetting matrix — for each consecutive ckpt pair, fraction of cases
    # that REGRESSED (IoU dropped > 0.05) per dataset
    fig, ax = plt.subplots(figsize=(11, 4))
    pair_labels = [f'{steps[i]}→{steps[i+1]}' for i in range(len(steps) - 1)]
    rows = []
    for ds in DSETS:
        case_indices = [a['_case_idx'] for a in anchors if a['dataset'] == ds]
        row = []
        for i in range(len(steps) - 1):
            n_total = 0; n_regress = 0
            for ci in case_indices:
                a = by_case.get(ci, {}).get(steps[i])
                b = by_case.get(ci, {}).get(steps[i + 1])
                if a is None or b is None:
                    continue
                n_total += 1
                if (a - b) > 0.05:
                    n_regress += 1
            row.append(n_regress / n_total if n_total else 0)
        rows.append(row)
    im = ax.imshow(rows, cmap='Reds', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=30)
    ax.set_yticks(range(len(DSETS)))
    ax.set_yticklabels(DSETS)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                     color='white' if v > 0.25 else 'black', fontsize=10)
    ax.set_title('Forgetting rate: fraction of cases with IoU drop > 0.05 between consecutive ckpts')
    fig.colorbar(im, ax=ax, label='fraction')
    fig.tight_layout()
    fig.savefig(out_root / 'forget_matrix.png', dpi=110)
    plt.close(fig)
    print(f'  → forget_matrix.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dirs', nargs='+', required=True)
    ap.add_argument('--steps', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-model', default='Qwen/Qwen3-VL-2B-Instruct')
    ap.add_argument('--backbone', default='qwen3_vl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--gen-batch-size', type=int, default=4,
                     help='examples per generate() call')
    ap.add_argument('--render-workers', type=int, default=12)
    ap.add_argument('--phase', default='all',
                     choices=['gen', 'render', 'iou', 'figures', 'all'],
                     help='Which phase to run. gen/render/iou/figures are resumable.')
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    (out_root / 'pred_renders').mkdir(parents=True, exist_ok=True)
    (out_root / 'gt_renders').mkdir(parents=True, exist_ok=True)

    # 1. Anchors (resolve to disk JSON for resumability)
    anchors_jsonl = out_root / 'anchors.jsonl'
    if anchors_jsonl.exists():
        anchors = [json.loads(l) for l in anchors_jsonl.read_text().splitlines()]
        print(f'Loaded {len(anchors)} anchors from {anchors_jsonl}')
    else:
        anchors = load_anchors(seed=args.seed)
        for i, a in enumerate(anchors):
            a['_case_idx'] = i
        # Persist (drop image objects — only keep paths)
        with anchors_jsonl.open('w') as f:
            for a in anchors:
                f.write(json.dumps({k: v for k, v in a.items()
                                     if not isinstance(v, Image.Image)}) + '\n')

    # 2. Resolve ckpts
    steps = [int(s.strip()) for s in args.steps.split(',') if s.strip()]
    ckpt_paths = []
    for s in steps:
        for parent in args.ckpt_dirs:
            p = Path(parent) / f'checkpoint-{s}'
            if p.is_dir():
                ckpt_paths.append((s, p)); break
        else:
            print(f'WARN: no ckpt for step {s}')

    codes_jsonl = out_root / 'codes.jsonl'

    # ---- PHASE: gen ----
    if args.phase in ('gen', 'all'):
        print(f'PHASE=gen — {len(ckpt_paths)} ckpts × {len(anchors)} anchors', flush=True)
        processor = AutoProcessor.from_pretrained(
            args.base_model, min_pixels=256*28*28, max_pixels=1280*28*28,
            padding_side='left')
        CadrilleCls = get_cadrille_class(args.backbone)
        all_results = []
        t0 = time.time()
        for step, ckpt_path in ckpt_paths:
            ts = time.time()
            print(f'\n=== ckpt-{step} @ {ckpt_path} ===', flush=True)
            try:
                model = CadrilleCls.from_pretrained(
                    str(ckpt_path), torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', device_map='cuda')
            except Exception as e:
                print(f'  load failed: {e}'); continue
            model.eval()

            codes_for_step = []
            for i in range(0, len(anchors), args.gen_batch_size):
                chunk = anchors[i:i + args.gen_batch_size]
                try:
                    gens = generate_batch(model, processor, chunk,
                                           max_new_tokens=args.max_new_tokens, device='cuda')
                except Exception as e:
                    print(f'  batch {i} gen failed: {type(e).__name__}: {e}')
                    gens = [''] * len(chunk)
                codes_for_step.extend(gens)
                if (i // args.gen_batch_size) % 5 == 0:
                    print(f'  generated {len(codes_for_step)}/{len(anchors)}', flush=True)
            for a, code in zip(anchors, codes_for_step):
                all_results.append({
                    'case_idx': a['_case_idx'],
                    'step': step,
                    'dataset': a['dataset'],
                    'file_name': a['file_name'],
                    'code': code,
                    'render_status': None, 'iou': None,
                })
            del model; torch.cuda.empty_cache()
            print(f'  ckpt-{step} gen done ({time.time() - ts:.1f}s)', flush=True)

            # Persist after EACH ckpt — resumable on crash
            with codes_jsonl.open('w') as f:
                for r in all_results:
                    f.write(json.dumps(r) + '\n')

        print(f'\nGen total {time.time() - t0:.0f}s — saved {codes_jsonl}', flush=True)

        if args.phase == 'gen':
            return

    # ---- PHASE: render (load codes from jsonl, no CUDA needed) ----
    if args.phase in ('render', 'all'):
        if not codes_jsonl.exists():
            print(f'No {codes_jsonl}; run --phase gen first.'); return
        all_results = [json.loads(l) for l in codes_jsonl.read_text().splitlines()]
        print(f'PHASE=render — {len(all_results)} codes to render', flush=True)

        # GT renders
        print('Rendering GTs ...', flush=True)
        gt_tasks = [(a['_case_idx'], a['gt_mesh_path'],
                     str(out_root / 'gt_renders' / f'gt_{a["_case_idx"]:03d}.png'), 140)
                    for a in anchors]
        # Use mp.Pool with spawn to avoid CUDA-fork issues
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.render_workers) as pool:
            for ci, status in pool.imap_unordered(_render_gt, gt_tasks, chunksize=4):
                pass

        # Pred renders
        print('Rendering predictions ...', flush=True)
        render_tasks = []
        for r in all_results:
            png = str(out_root / 'pred_renders' / f'case{r["case_idx"]:03d}_step{r["step"]:05d}.png')
            render_tasks.append((r['case_idx'], r['step'], r['code'], png, 140))
        status_map = {}
        t0 = time.time(); n = 0
        with ctx.Pool(args.render_workers) as pool:
            for ci, st, status in pool.imap_unordered(_render_one, render_tasks, chunksize=4):
                status_map[(ci, st)] = status
                n += 1
                if n % 100 == 0:
                    print(f'  rendered {n}/{len(render_tasks)}  ({time.time() - t0:.0f}s)', flush=True)
        for r in all_results:
            r['render_status'] = status_map.get((r['case_idx'], r['step']), 'missing')
        print(f'Render total {time.time() - t0:.0f}s', flush=True)

        # Persist render status
        with codes_jsonl.open('w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')
        print(f'  → updated {codes_jsonl}')
        if args.phase == 'render':
            return

    # ---- PHASE: iou (load codes+render_status, compute IoU) ----
    if args.phase in ('iou', 'all'):
        if not codes_jsonl.exists():
            print(f'No {codes_jsonl}'); return
        all_results = [json.loads(l) for l in codes_jsonl.read_text().splitlines()]
        print(f'PHASE=iou — {sum(1 for r in all_results if r.get("render_status") == "ok")} cases to score', flush=True)
        case_to_gt = {a['_case_idx']: a['gt_mesh_path'] for a in anchors}
        iou_tasks = [(r['case_idx'], r['step'], r['code'], case_to_gt[r['case_idx']])
                      for r in all_results if r.get('render_status') == 'ok']
        iou_map = {}
        t0 = time.time(); n = 0
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.render_workers) as pool:
            for ci, st, iou in pool.imap_unordered(_iou_one, iou_tasks, chunksize=4):
                iou_map[(ci, st)] = iou
                n += 1
                if n % 100 == 0:
                    print(f'  IoU computed {n}/{len(iou_tasks)}  ({time.time() - t0:.0f}s)', flush=True)
        for r in all_results:
            r['iou'] = iou_map.get((r['case_idx'], r['step']))
        print(f'IoU total {time.time() - t0:.0f}s', flush=True)

        # Save final results.jsonl
        with (out_root / 'results.jsonl').open('w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')
        print(f'  → results.jsonl')
        if args.phase == 'iou':
            return

    # ---- PHASE: figures ----
    if args.phase in ('figures', 'all'):
        results_jsonl = out_root / 'results.jsonl'
        if not results_jsonl.exists():
            print(f'No {results_jsonl}'); return
        all_results = [json.loads(l) for l in results_jsonl.read_text().splitlines()]
        print(f'PHASE=figures — building from {len(all_results)} rows', flush=True)
        for ds in ('benchcad_val', 'deepcad_test', 'fusion360_test'):
            build_collage(out_root, ds, anchors, [s for s, _ in ckpt_paths])
        build_trajectory_plots(out_root, all_results, anchors, [s for s, _ in ckpt_paths])

    # ---- Quick markdown summary (only in figures/all phase) ----
    if args.phase in ('figures', 'all'):
        all_results = [json.loads(l) for l in (out_root / 'results.jsonl').read_text().splitlines()]
        by_ds_step = {}
        for r in all_results:
            if r['iou'] is not None:
                key = (r['dataset'], r['step'])
                by_ds_step.setdefault(key, []).append(r['iou'])
        md = ['# 100-case trajectory analysis\n',
              f'\n7 ckpts: {[s for s, _ in ckpt_paths]}\n',
              f'\n## IoU mean per dataset × ckpt (resolved cases only)\n\n']
        md.append('| dataset | ' + ' | '.join(f'step {s}' for s, _ in ckpt_paths) + ' |\n')
        md.append('|---|' + '---:|' * len(ckpt_paths) + '\n')
        for ds in ('benchcad_val', 'deepcad_test', 'fusion360_test'):
            cells = []
            for s, _ in ckpt_paths:
                vals = by_ds_step.get((ds, s), [])
                cells.append(f'{np.mean(vals):.3f} (n={len(vals)})' if vals else '—')
            md.append(f'| {ds} | ' + ' | '.join(cells) + ' |\n')
        md.append('\n## Figures\n\n')
        md.append('### IoU trajectory (line plot)\n![trajectory](trajectory_iou.png)\n\n')
        md.append('### IoU distribution per ckpt × dataset\n![boxplot](boxplot_iou.png)\n\n')
        md.append('### Forgetting matrix\n![forget](forget_matrix.png)\n\n')
        md.append('### Per-case collages\n')
        md.append('- ![BenchCAD val](collage_benchcad_val.png)\n')
        md.append('- ![DeepCAD test](collage_deepcad_test.png)\n')
        md.append('- ![Fusion360 test](collage_fusion360_test.png)\n')
        with (out_root / 'report.md').open('w') as f:
            f.write(''.join(md))
        print(f'\nDONE — report.md + figures @ {out_root}', flush=True)


if __name__ == '__main__':
    main()
