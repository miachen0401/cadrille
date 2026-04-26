"""Materialise filapro/cad-recode-v1.5 train (~1M .py) into data/cad-recode-v1.5/.

Source: HF dataset `filapro/cad-recode-v1.5` (CAD-Recode paper, ~1M cadquery
programs). Repo only ships .py source — no PNG / no STL. We need to:
  1. snapshot_download the train/ + val/ tree  (~500 MB on disk)
  2. exec each .py via cadquery → trimesh mesh
  3. render 4-view composite PNG (268×268 each, stitched 2×2 → 536×536) to
     {stem}_render.png alongside the .py
  4. write train.pkl / val.pkl manifests

The output layout matches `CadRecode20kDataset` so the existing loader works
unchanged once you swap `cad-recode-20k` → `cad-recode-v1.5` in the config or
in train.py's path resolution.

Resumable: per-sample skip-if-exists check on the PNG. Failures (cadquery
exec error / no mesh) are logged but don't abort the sweep — they just don't
appear in the pkl.

Wall-clock estimate (12 CPU cores):
  - Phase 1 (download .py):  ~5-15 min
  - Phase 2 (exec+render):   ~25-50 h for full 1M, scales linearly
                             (~30k items per hour with 12 workers)

Subset flag `--max-samples N` materialises only the first N items by
deterministic sort-order, useful for quickly extending coverage from 20k → 100k
without committing to the full 1M sweep.

Usage:
  set -a; source .env; set +a
  uv run python -m data_prep.fetch_cadrecode_full --phase download
  uv run python -m data_prep.fetch_cadrecode_full --phase render --workers 12
  # Or do both phases:
  uv run python -m data_prep.fetch_cadrecode_full --phase all --workers 12
  # Smaller test:
  uv run python -m data_prep.fetch_cadrecode_full --phase render --max-samples 5000
"""
from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def download_source(out_root: Path) -> None:
    """Phase 1 — pull all .py files from filapro/cad-recode-v1.5."""
    from huggingface_hub import snapshot_download

    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[download] snapshot_download → {out_root}', flush=True)
    snapshot_download(
        repo_id='filapro/cad-recode-v1.5',
        repo_type='dataset',
        allow_patterns=['train/**/*.py', 'val/**/*.py', 'README.md'],
        local_dir=str(out_root),
        max_workers=8,
    )
    n_train = sum(1 for _ in (out_root / 'train').rglob('*.py'))
    n_val = sum(1 for _ in (out_root / 'val').rglob('*.py'))
    print(f'[download] done: train={n_train}, val={n_val}', flush=True)


def _flatten_stem(rel_py: str) -> str:
    """`train/batch_94/949381.py` → `batch_94_949381`. Matches our 20k naming."""
    parts = rel_py.replace('.py', '').split('/')
    return '_'.join(parts[1:])  # drop the 'train' or 'val' prefix


def _render_one(args: tuple) -> tuple[str, str]:
    """Worker: read .py, exec via cadquery, render 4-view PNG, save.

    Returns (stem, status) — status is 'ok', 'skip', or an error message.
    """
    py_path, png_path, img_size = args
    stem = Path(py_path).stem

    if os.path.exists(png_path):
        return stem, 'skip'

    # Inline imports keep worker-cold-start cheap when many tasks skip.
    import io
    import numpy as np
    import trimesh
    from PIL import Image, ImageOps
    import cadquery as cq  # noqa: F401
    import open3d
    from common.datasets import mesh_to_image  # uses Visualizer(visible=False)

    def _on_alarm(signum, frame):
        raise TimeoutError('cadquery exec >30s')

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(30)

    try:
        code = open(py_path).read()
        try:
            code_obj = compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return stem, f'syntax: {e}'
        _captured = {}
        g = {'show_object': lambda obj, *a, **kw: _captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        res = g.get('r') or g.get('result') or _captured.get('r')
        if res is None:
            return stem, 'no_r_result'
        compound = res.val()
        vertices, faces = compound.tessellate(0.001, 0.1)
        if len(faces) < 3:
            return stem, 'empty_mesh'
        mesh = trimesh.Trimesh(
            [(v.x, v.y, v.z) for v in vertices], faces)

        # Match CadRecodeDataset.get_img normalization (train/val branch)
        mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 200))
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        verts = np.asarray(mesh.vertices)
        tri_faces = np.asarray(mesh.faces)
        o3d_mesh = open3d.geometry.TriangleMesh()
        o3d_mesh.vertices = open3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = open3d.utility.Vector3iVector(tri_faces)
        o3d_mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d_mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            img = mesh_to_image(o3d_mesh, camera_distance=-0.9, front=front,
                                img_size=img_size)
            images.append(img)
        images = [ImageOps.expand(im, border=3, fill='black') for im in images]
        # 2×2 grid, matches num_imgs=4 in CadRecodeDataset
        composite = Image.fromarray(np.vstack((
            np.hstack((np.array(images[0]), np.array(images[1]))),
            np.hstack((np.array(images[2]), np.array(images[3])))
        )))
        composite.save(png_path)
        return stem, 'ok'
    except TimeoutError:
        return stem, 'timeout'
    except Exception as e:
        return stem, f'err: {type(e).__name__}: {str(e)[:80]}'
    finally:
        signal.alarm(0)


def render_split(out_root: Path, split: str, workers: int,
                 max_samples: int | None, img_size: int = 268) -> None:
    """Phase 2 — exec + render every .py in {out_root}/{split}/."""
    split_dir = out_root / split
    if not split_dir.exists():
        print(f'[{split}] no dir {split_dir} — run --phase download first', flush=True)
        return

    py_files = sorted(split_dir.rglob('*.py'))
    if max_samples is not None:
        py_files = py_files[:max_samples]
    print(f'[{split}] {len(py_files)} .py files to consider', flush=True)

    tasks = []
    for py in py_files:
        png = py.parent / (py.stem + '_render.png')
        tasks.append((str(py), str(png), img_size))

    n_skip = sum(1 for _, png, _ in tasks if os.path.exists(png))
    print(f'[{split}] {n_skip} already rendered, {len(tasks) - n_skip} to do',
          flush=True)

    t0 = time.time()
    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    fail_log = out_root / f'{split}_render_failures.txt'
    fail_log.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as pool, \
         open(fail_log, 'a') as flog:
        futures = [pool.submit(_render_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            stem, status = fut.result()
            if status == 'ok':
                counts['ok'] += 1
            elif status == 'skip':
                counts['skip'] += 1
            else:
                counts['fail'] += 1
                flog.write(f'{stem}\t{status}\n')
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(tasks) - i - 1) / rate / 3600
                print(f'  {i + 1}/{len(tasks)}  '
                      f'ok={counts["ok"]} skip={counts["skip"]} fail={counts["fail"]}  '
                      f'rate={rate:.1f}/s  eta={eta:.1f}h', flush=True)

    print(f'[{split}] done: ok={counts["ok"]} skip={counts["skip"]} '
          f'fail={counts["fail"]}  failures logged to {fail_log}', flush=True)


def write_pkl(out_root: Path) -> None:
    """Build {train,val}.pkl in CadRecode20kDataset schema:
        [{uid, py_path, png_path}, ...]
    """
    for split in ('train', 'val'):
        rows = []
        split_dir = out_root / split
        if not split_dir.exists():
            continue
        for py in sorted(split_dir.rglob('*.py')):
            png = py.parent / (py.stem + '_render.png')
            if not png.exists():
                continue
            rows.append({
                'uid': _flatten_stem(str(py.relative_to(out_root))),
                'py_path': str(py.relative_to(out_root)),
                'png_path': str(png.relative_to(out_root)),
            })
        pkl = out_root / f'{split}.pkl'
        with pkl.open('wb') as fp:
            pickle.dump(rows, fp)
        print(f'  {split}.pkl: {len(rows)} rows → {pkl}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['download', 'render', 'pkl', 'all'],
                    default='all',
                    help='download = pull .py from HF; render = exec+png; '
                         'pkl = build train.pkl/val.pkl; all = all three')
    ap.add_argument('--out', default='data/cad-recode-v1.5',
                    help='Output root (default: data/cad-recode-v1.5)')
    ap.add_argument('--workers', type=int, default=12,
                    help='Parallel workers for render phase')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='Render only the first N .py files in each split '
                         '(deterministic by sort order). Useful for staged '
                         'corpus expansion: 20k → 100k → 1M')
    ap.add_argument('--img-size', type=int, default=268,
                    help='Per-view PNG size (matches CadRecodeDataset default)')
    args = ap.parse_args()

    out_root = Path(args.out).resolve()

    if args.phase in ('download', 'all'):
        download_source(out_root)
    if args.phase in ('render', 'all'):
        render_split(out_root, 'val', args.workers, args.max_samples,
                     args.img_size)
        render_split(out_root, 'train', args.workers, args.max_samples,
                     args.img_size)
    if args.phase in ('pkl', 'all'):
        write_pkl(out_root)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
