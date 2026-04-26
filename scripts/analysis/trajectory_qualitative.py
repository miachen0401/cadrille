"""Qualitative trajectory analysis — generate code + render mesh per ckpt × anchor.

Picks 6 anchor items (2 from each of BenchCAD val, DeepCAD test, Fusion360 test)
spanning diverse op families. For each curriculum ckpt, generates greedy code,
exec→trimesh→render PNG, computes IoU. Builds markdown report side-by-side.

Usage:
  uv run python -m scripts.analysis.trajectory_qualitative \
    --ckpt-dirs /ephemeral/checkpoints/curriculum_best_from_hf \
                /ephemeral/checkpoints/sft-s20k-lr2e-4-b8a4-img-0425-1929 \
    --steps 1000,5000,10000,11000,15000,18000,19000,20000 \
    --out /home/ubuntu/cadrille/eval_outputs/qualitative_trajectory_<run-tag>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from common.model import collate, get_cadrille_class
from common.metrics import compute_metrics
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Anchor selection — fixed seed for reproducibility, op-family diversity
# ---------------------------------------------------------------------------
def load_anchors(seed: int = 4242, n_per_dataset: int = 2) -> list[dict]:
    """Pick 2 from each: BenchCAD val, DeepCAD test, Fusion360 test.
    Each anchor has: dataset_label, file_name, gt_mesh_path, gt_code (or None),
    image, description.
    """
    out = []

    # 1. BenchCAD val (has GT code + STL)
    bc_root = Path('data/benchcad')
    with (bc_root / 'val.pkl').open('rb') as f:
        rows = pickle.load(f)
    rng = np.random.default_rng(seed)
    rows_shuffled = rng.permutation(rows).tolist()
    picked = 0
    seen_ops = set()
    for r in rows_shuffled:
        py = bc_root / r['py_path']
        png = bc_root / r['png_path']
        stl = bc_root / r['mesh_path']
        if not (py.exists() and png.exists() and stl.exists()):
            continue
        code = py.read_text()
        # Try to pick diverse ops
        op_sig = ''.join(sorted({
            op for op in ['fillet', 'chamfer', 'revolve', 'sphere', 'sweep',
                          'shell', 'mirror', 'box', 'cylinder', 'extrude']
            if f'.{op}(' in code
        }))[:30]
        if op_sig in seen_ops and picked < n_per_dataset:
            continue
        out.append({
            'dataset': 'benchcad_val',
            'family_hint': op_sig or 'simple',
            'file_name': r['uid'],
            'gt_mesh_path': str(stl),
            'gt_code': code,
            'image': Image.open(png).convert('RGB'),
            'description': 'Generate cadquery code',
        })
        seen_ops.add(op_sig)
        picked += 1
        if picked == n_per_dataset:
            break

    # 2. DeepCAD test (no GT code, img + STL)
    for label, root_str in [('deepcad_test', 'data/deepcad_test_mesh'),
                              ('fusion360_test', 'data/fusion360_test_mesh')]:
        rootp = Path(root_str)
        stls = sorted(rootp.glob('*.stl'))
        # Deterministic sampled subset
        idx = rng.integers(0, len(stls), n_per_dataset).tolist()
        for i in idx:
            stl = stls[i]
            png = stl.with_name(stl.stem + '_render.png')
            if not png.exists():
                continue
            out.append({
                'dataset': label,
                'family_hint': 'unknown',  # no GT code to inspect
                'file_name': stl.stem,
                'gt_mesh_path': str(stl),
                'gt_code': None,
                'image': Image.open(png).convert('RGB'),
                'description': 'Generate cadquery code',
            })
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_one(model, processor, anchor, max_new_tokens=768, device='cuda'):
    """Greedy generation for a single anchor. Returns generated code str."""
    ex = {
        'video': [anchor['image']],
        'description': anchor['description'],
        'file_name': anchor['file_name'],
    }
    batch = collate([ex], processor=processor, n_points=256, eval=True)
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
    return processor.decode(out_ids[0, prompt_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Render generated code
# ---------------------------------------------------------------------------
def render_code(code: str, png_out: str, img_size: int = 268, timeout: int = 30) -> str:
    """Exec cadquery code → trimesh → 1-view PNG. Returns 'ok' or err message.

    Renders one front view (not 4-view) for fast inspection.
    """
    import io
    import trimesh
    import open3d
    import cadquery as cq  # noqa: F401
    from common.datasets import mesh_to_image

    def _on_alarm(signum, frame):
        raise TimeoutError('cadquery exec timeout')

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout)

    try:
        try:
            code_obj = compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return f'syntax: {e.msg[:60]}'
        captured = {}
        g = {'show_object': lambda obj, *a, **kw: captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        res = g.get('r') or g.get('result') or captured.get('r')
        if res is None:
            return 'no_r_or_result'
        compound = res.val()
        verts, faces = compound.tessellate(0.001, 0.1)
        if len(faces) < 3:
            return 'empty_mesh'
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 200))
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        o3d.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d.compute_vertex_normals()
        img = mesh_to_image(o3d, camera_distance=-0.9, front=[1, 1, 1], img_size=img_size)
        img = ImageOps.expand(img, border=3, fill='black')
        img.save(png_out)
        return 'ok'
    except TimeoutError:
        return 'timeout'
    except Exception as e:
        return f'{type(e).__name__}: {str(e)[:60]}'
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt-dirs', nargs='+', required=True,
                    help='Parent dirs of checkpoint-{step} subdirs')
    ap.add_argument('--steps', required=True,
                    help='comma-sep list of steps to eval, e.g. 1000,5000,11000')
    ap.add_argument('--out', required=True, help='output dir')
    ap.add_argument('--base-model', default='Qwen/Qwen3-VL-2B-Instruct')
    ap.add_argument('--backbone', default='qwen3_vl')
    ap.add_argument('--anchor-seed', type=int, default=4242)
    ap.add_argument('--n-per-dataset', type=int, default=2)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    args = ap.parse_args()

    out_root = Path(args.out)
    (out_root / 'renders').mkdir(parents=True, exist_ok=True)

    # 1. Anchors
    print('Loading anchors ...', flush=True)
    anchors = load_anchors(seed=args.anchor_seed, n_per_dataset=args.n_per_dataset)
    print(f'Got {len(anchors)} anchors:')
    for i, a in enumerate(anchors):
        print(f'  [{i}] {a["dataset"]} / {a["file_name"]} / family={a["family_hint"]}')
        # Save GT image + GT mesh render
        a['image'].save(out_root / 'renders' / f'anchor_{i:02d}_input.png')
        # Render GT mesh
        gt_render_path = out_root / 'renders' / f'anchor_{i:02d}_gt.png'
        try:
            import trimesh, open3d
            from common.datasets import mesh_to_image
            mesh = trimesh.load(a['gt_mesh_path'])
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
            img = mesh_to_image(o3d, camera_distance=-1.6, front=[1, 1, 1], img_size=268)
            img.save(str(gt_render_path))
        except Exception as e:
            print(f'  GT render failed for anchor {i}: {e}')

    # 2. Resolve ckpt paths
    steps = [int(s.strip()) for s in args.steps.split(',') if s.strip()]
    ckpt_paths = []
    for s in steps:
        for parent in args.ckpt_dirs:
            p = Path(parent) / f'checkpoint-{s}'
            if p.is_dir():
                ckpt_paths.append((s, p))
                break
        else:
            print(f'WARN: no ckpt found for step {s}')
    print(f'Will eval {len(ckpt_paths)} ckpts: {[s for s, _ in ckpt_paths]}', flush=True)

    # 3. Load processor once (shared across ckpts)
    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=256*28*28, max_pixels=1280*28*28,
        padding_side='left')

    CadrilleCls = get_cadrille_class(args.backbone)
    results = []  # rows: {step, anchor_idx, code, render_status, iou}

    for step, ckpt_path in ckpt_paths:
        print(f'\n=== ckpt-{step} @ {ckpt_path} ===', flush=True)
        try:
            model = CadrilleCls.from_pretrained(
                str(ckpt_path), torch_dtype=torch.bfloat16,
                attn_implementation='sdpa', device_map='cuda')
        except Exception as e:
            print(f'  load failed: {e}')
            continue
        model.eval()

        for ai, anchor in enumerate(anchors):
            try:
                code = generate_one(
                    model, processor, anchor,
                    max_new_tokens=args.max_new_tokens, device='cuda')
            except Exception as e:
                print(f'  anchor {ai} gen failed: {type(e).__name__}: {e}')
                code = ''
            png_out = str(out_root / 'renders' / f'anchor_{ai:02d}_step{step:05d}.png')
            status = render_code(code, png_out)
            iou = None
            if anchor['gt_mesh_path'] and status == 'ok':
                try:
                    iou_r, _ = compute_metrics(code, anchor['gt_mesh_path'],
                                               timeout=20, use_pool=False)
                    iou = float(iou_r) if iou_r is not None and iou_r >= 0 else None
                except Exception:
                    iou = None
            print(f'  [{ai}] {anchor["dataset"]:>15s}/{anchor["file_name"]:>10s}  '
                  f'render={status:>20s}  iou={iou}', flush=True)
            results.append({
                'step': step, 'anchor_idx': ai,
                'dataset': anchor['dataset'],
                'file_name': anchor['file_name'],
                'family_hint': anchor['family_hint'],
                'code': code, 'render_status': status, 'iou': iou,
            })

        # Free GPU mem
        del model
        torch.cuda.empty_cache()

    # 4. Save raw + markdown report
    with (out_root / 'results.jsonl').open('w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'\nSaved {len(results)} rows → {out_root / "results.jsonl"}', flush=True)

    # 5. Build markdown grid: one section per anchor, columns = ckpts
    md_lines = []
    md_lines.append(f'# Qualitative trajectory — curriculum_qwen3vl_2b\n')
    md_lines.append(f'Anchors: {len(anchors)} (seed={args.anchor_seed}); '
                     f'ckpts: {[s for s, _ in ckpt_paths]}\n')
    by_anchor = {}
    for r in results:
        by_anchor.setdefault(r['anchor_idx'], []).append(r)

    for ai, anchor in enumerate(anchors):
        md_lines.append(f'\n## Anchor {ai}: {anchor["dataset"]} / `{anchor["file_name"]}` (family={anchor["family_hint"]})\n')
        md_lines.append(f'**Input image** (model sees this):\n')
        md_lines.append(f'![input](renders/anchor_{ai:02d}_input.png)\n')
        gt_render = out_root / 'renders' / f'anchor_{ai:02d}_gt.png'
        if gt_render.exists():
            md_lines.append(f'\n**GT mesh render**:\n\n![GT](renders/anchor_{ai:02d}_gt.png)\n')
        if anchor.get('gt_code'):
            md_lines.append(f'\n**GT code**:\n\n```python\n{anchor["gt_code"]}\n```\n')
        md_lines.append('\n### Trajectory:\n')
        md_lines.append('| step | render | IoU | status |\n|---:|---|---:|---|\n')
        for r in by_anchor.get(ai, []):
            iou_str = f'{r["iou"]:.3f}' if r["iou"] is not None else '—'
            png = f'renders/anchor_{ai:02d}_step{r["step"]:05d}.png'
            png_full = out_root / png
            cell = f'![{r["step"]}]({png})' if png_full.exists() else '_(render failed)_'
            md_lines.append(f'| {r["step"]} | {cell} | {iou_str} | {r["render_status"]} |\n')
        md_lines.append('\n### Code (per step):\n')
        for r in by_anchor.get(ai, []):
            md_lines.append(f'\n**step {r["step"]}** (iou={r["iou"]}, {r["render_status"]}):\n')
            md_lines.append(f'```python\n{r["code"]}\n```\n')

    with (out_root / 'report.md').open('w') as f:
        f.write('\n'.join(md_lines))
    print(f'Markdown report → {out_root / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
