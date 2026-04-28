"""Render all text2cad codes (legacy + bench) to PNG, write next to data.

Two sources, both currently text-only (no input image during training):

  text2cad-bench/{train,test,val}.pkl rows:
    {uid, description, code}
    → render code → save data/text2cad-bench/{split}/{uid}_render.png

  text2cad/{train,test,val}.pkl rows:
    {uid, description}     # NO code in pkl
    → read code from data/text2cad/cadquery/{uid}.py
    → render → save data/text2cad/{split}/{uid}_render.png

Skip already-rendered (cache hit). Idempotent — safe to re-run.

Usage:
    uv run python -m data_prep.render_text2cad
        # default workers=4, all splits, both sources
    uv run python -m data_prep.render_text2cad --workers 8 --splits train --sources bench
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import pickle
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image as _PIL  # noqa: F401  (keeps PIL importable in workers)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _render_one(args: tuple) -> tuple[str, bool, str]:
    """Worker: exec cadquery code → 4-view 256×256 PNG → write to disk.
    Returns (cache_path, ok, status)."""
    cache_path_str, code, color_rgb, img_size = args
    cache_path = Path(cache_path_str)
    if cache_path.exists():
        return cache_path_str, True, 'cached'
    if not code or not code.strip():
        return cache_path_str, False, 'empty'

    def _on_alarm(signum, frame): raise TimeoutError('cadquery timeout')
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(15)
    try:
        import trimesh
        import open3d
        import cadquery as cq  # noqa: F401
        from PIL import Image, ImageOps
        from common.datasets import mesh_to_image

        try:
            code_obj = compile(code, '<string>', 'exec')
        except SyntaxError:
            return cache_path_str, False, 'syntax'
        captured = {}
        g = {'show_object': lambda obj, *a, **kw: captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        res = g.get('r') or g.get('result') or captured.get('r')
        if res is None:
            return cache_path_str, False, 'no_r'
        compound = res.val()
        verts, faces = compound.tessellate(0.001, 0.1)
        if len(faces) < 3:
            return cache_path_str, False, 'empty_mesh'
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        # Normalize to [0,1]^3 — matches common.meshio.render_img exactly
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        ext = float(max(mesh.extents))
        if ext > 1e-7:
            mesh.apply_scale(2.0 / ext)        # → [-1, 1]
        mesh.apply_scale(0.5)                  # → [-0.5, 0.5]
        mesh.apply_translation([0.5, 0.5, 0.5])  # → [0, 1]
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        # Use the canonical "yellow" color from common.meshio.render_img
        o3d.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d.compute_vertex_normals()
        # 4-view 2×2 tile — replicate common.meshio.render_img exactly
        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        sub = img_size // 2  # each tile is half (so total = img_size×img_size)
        imgs = [ImageOps.expand(
            mesh_to_image(o3d, camera_distance=-0.9, front=f_, img_size=sub),
            border=3, fill='black') for f_ in fronts]
        combined = Image.fromarray(np.vstack((
            np.hstack((np.array(imgs[0]), np.array(imgs[1]))),
            np.hstack((np.array(imgs[2]), np.array(imgs[3]))))))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(cache_path, format='PNG', optimize=True)
        return cache_path_str, True, 'ok'
    except TimeoutError:
        return cache_path_str, False, 'timeout'
    except Exception as e:
        return cache_path_str, False, f'err:{type(e).__name__}'
    finally:
        signal.alarm(0)


def collect_tasks(source: str, root: Path, splits: list[str], img_size: int) -> list[tuple]:
    """Return list of (png_path, code, color_rgb, img_size) ready for the worker."""
    out: list[tuple] = []
    for split in splits:
        pkl = root / f'{split}.pkl'
        if not pkl.exists():
            print(f'[{source}] {pkl} missing, skipping', flush=True); continue
        rows = pickle.load(open(pkl, 'rb'))
        cadquery_dir = root / 'cadquery'
        n_skipped, n_no_code = 0, 0
        for r in rows:
            uid = r['uid']
            png_path = root / split / f'{uid}_render.png'
            if png_path.exists():
                n_skipped += 1; continue
            if 'code' in r and r['code']:
                code = r['code']
            else:
                code_p = cadquery_dir / f'{uid}.py'
                if not code_p.exists():
                    n_no_code += 1; continue
                code = code_p.read_text()
            out.append((str(png_path), code, (136, 200, 255), img_size))
        print(f'[{source}/{split}] {len(rows)} rows, {n_skipped} cached, '
              f'{n_no_code} no-code, {sum(1 for t in out if t[0].startswith(str(root/split)))} to render',
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--splits', nargs='+', default=['train', 'test', 'val'])
    ap.add_argument('--sources', nargs='+', default=['bench', 'legacy'],
                    choices=['bench', 'legacy'])
    args = ap.parse_args()

    src_map = {
        'bench':  REPO_ROOT / 'data' / 'text2cad-bench',
        'legacy': REPO_ROOT / 'data' / 'text2cad',
    }
    tasks: list[tuple] = []
    for s in args.sources:
        tasks.extend(collect_tasks(s, src_map[s], args.splits, args.img_size))

    if not tasks:
        print('nothing to render (all cached?)', flush=True); return

    print(f'\ntotal tasks: {len(tasks)}, workers: {args.workers}', flush=True)
    t0 = time.time()
    n_ok = 0; n_failed = 0
    fail_reasons: dict[str, int] = {}
    last_print = t0
    # Use multiprocessing.Pool with maxtasksperchild — recycles workers
    # periodically to avoid memory leaks AND survives a single worker crash
    # (replaces the dead worker on next iteration).
    with mp.Pool(processes=args.workers, maxtasksperchild=200) as pool:
        for i, (path, ok, status) in enumerate(pool.imap_unordered(
                _render_one, tasks, chunksize=4)):
            if ok: n_ok += 1
            else:
                n_failed += 1
                fail_reasons[status] = fail_reasons.get(status, 0) + 1
            if time.time() - last_print > 60:  # log every minute
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f'  [{i+1}/{len(tasks)}] ok={n_ok} failed={n_failed}  '
                      f'rate={rate:.1f}/s  eta={eta/60:.0f} min', flush=True)
                last_print = time.time()

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed/60:.1f} min — ok={n_ok}, failed={n_failed}',
          flush=True)
    if fail_reasons:
        print(f'failure breakdown: {fail_reasons}', flush=True)


if __name__ == '__main__':
    main()
