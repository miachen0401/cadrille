"""Fill in missing composite_png renders in BenchCAD/benchcad-easy on HF.

Current state (2026-04-28):
  - Total rows: 109 804
  - With composite_png: 12 356 (11.3 %)
  - Without composite_png: 97 448 (88.7 %)

Each row has gt_code (cadquery). For rows with empty composite_png, we exec
the code, build the mesh, render it as a 4-view 268×268 PNG (matching the
existing samples), and write back the parquet.

Usage:
    uv run python -m data_prep.render_benchcad_easy --workers 8

Steps:
    1. Download data/test-00000-of-00001.parquet from HF
    2. Identify rows with empty composite_png
    3. Render gt_code → 4-view 268×268 PNG (parallel, mp.Pool)
    4. Write back: create new parquet with composite_png filled
    5. (optional --upload) push back to HF
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image as _PIL  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPO = 'BenchCAD/benchcad-easy'
PARQUET_PATH_IN_REPO = 'data/test-00000-of-00001.parquet'


def _render_one(args: tuple) -> tuple[int, bytes | None, str]:
    """Worker: row_idx + gt_code → 4-view 268×268 PNG bytes (or None)."""
    row_idx, code = args
    if not code or not code.strip():
        return row_idx, None, 'empty'

    def _on_alarm(s, f): raise TimeoutError('cadquery timeout')
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
            return row_idx, None, 'syntax'
        captured = {}
        g = {'show_object': lambda obj, *a, **kw: captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        res = g.get('r') or g.get('result') or captured.get('r')
        if res is None:
            return row_idx, None, 'no_r'
        compound = res.val()
        # Loose tessellation: 268×268 thumbnails don't need 0.001/0.1 precision.
        # 0.01/0.5 is ~10× faster and visually identical at this resolution.
        verts, faces = compound.tessellate(0.01, 0.5)
        if len(faces) < 3:
            return row_idx, None, 'empty_mesh'
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        # Normalize to [0,1]^3 — matches common.meshio.render_img exactly
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        ext = float(max(mesh.extents))
        if ext > 1e-7:
            mesh.apply_scale(2.0 / ext)
        mesh.apply_scale(0.5)
        mesh.apply_translation([0.5, 0.5, 0.5])
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        o3d.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d.compute_vertex_normals()
        # 4-view 2×2, each 128px → final ~268×268 with 3px borders
        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        sub = 128
        imgs = [ImageOps.expand(
            mesh_to_image(o3d, camera_distance=-0.9, front=f_, img_size=sub),
            border=3, fill='black') for f_ in fronts]
        combined = Image.fromarray(np.vstack((
            np.hstack((np.array(imgs[0]), np.array(imgs[1]))),
            np.hstack((np.array(imgs[2]), np.array(imgs[3])))))).convert('RGB')
        buf = io.BytesIO(); combined.save(buf, format='PNG', optimize=True)
        return row_idx, buf.getvalue(), 'ok'
    except TimeoutError:
        return row_idx, None, 'timeout'
    except Exception as e:
        return row_idx, None, f'err:{type(e).__name__}'
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-parquet', default=str(REPO_ROOT / 'data/_hf_upload/benchcad-easy/data/test-00000-of-00001.parquet'))
    ap.add_argument('--renders-out', default='',
                    help='Write {row_idx: png_bytes} pickle (for multi-VM split). '
                         'When set, parquet step is skipped.')
    ap.add_argument('--upload', action='store_true',
                    help='Upload back to HF (otherwise just write locally)')
    ap.add_argument('--limit', type=int, default=0,
                    help='Render at most N missing rows (0 = all)')
    ap.add_argument('--start-idx', type=int, default=0,
                    help='Start index into the missing-rows todo list (inclusive)')
    ap.add_argument('--end-idx', type=int, default=0,
                    help='End index into the missing-rows todo list (exclusive, '
                         '0 = until end). Range slicing happens BEFORE --limit.')
    # Shard semantics: slice over the FULL parquet (matches VM1 import script)
    ap.add_argument('--start-shard', type=int, default=-1,
                    help='Start shard (inclusive) over the FULL parquet. '
                         'When set with --end-shard, overrides --start-idx/--end-idx.')
    ap.add_argument('--end-shard', type=int, default=-1,
                    help='End shard (exclusive) over the FULL parquet.')
    ap.add_argument('--shard-size', type=int, default=2000,
                    help='Rows per shard (default 2000, matches VM1)')
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pickle as _pickle
    from huggingface_hub import hf_hub_download, HfApi

    print(f'Downloading {REPO}/{PARQUET_PATH_IN_REPO} ...', flush=True)
    parquet_in = hf_hub_download(REPO, PARQUET_PATH_IN_REPO, repo_type='dataset')
    print(f'  → {parquet_in}', flush=True)
    table = pq.read_table(parquet_in)
    n_total = len(table)
    imgs = table['composite_png'].to_pylist()
    codes = table['gt_code'].to_pylist()

    # Determine row-index range over the FULL parquet
    if args.start_shard >= 0 and args.end_shard >= 0:
        row_lo = args.start_shard * args.shard_size
        row_hi = min(args.end_shard * args.shard_size, n_total)
        print(f'shards [{args.start_shard}:{args.end_shard}) × {args.shard_size} '
              f'→ parquet rows [{row_lo}:{row_hi})', flush=True)
    else:
        row_lo, row_hi = 0, n_total

    # Find rows needing render within [row_lo, row_hi)
    todo_full = [(i, codes[i]) for i, r in enumerate(imgs)
                 if not (r and r.get('bytes'))]
    n_with = n_total - len(todo_full)
    n_full = len(todo_full)

    # Filter by row range (shards), then optionally by todo-list slice
    todo = [(i, c) for (i, c) in todo_full if row_lo <= i < row_hi]
    s, e = row_lo, row_hi
    if args.start_shard < 0 and args.end_shard < 0:
        # Legacy todo-list slicing (only when shards not used)
        s = args.start_idx
        e = args.end_idx if args.end_idx > 0 else len(todo)
        todo = todo[s:e]
    if args.limit > 0:
        todo = todo[:args.limit]
    # Shuffle todo (deterministic seed) so pathological-family clusters get
    # spread across workers — otherwise 12 workers can hit the same slow
    # family block at once and stall the pool.
    import random as _random
    _random.Random(20260429).shuffle(todo)

    print(f'rows: total={n_total}, with_png={n_with}, '
          f'todo_full={n_full}, todo_in_range={len(todo)} '
          f'(limit={args.limit or "all"})', flush=True)

    if not todo:
        print('Nothing to render — all rows already have composite_png '
              '(or empty slice).', flush=True)
        return

    # Render in parallel
    print(f'\nRendering {len(todo)} cadquery codes with {args.workers} workers...',
          flush=True)
    t0 = time.time()
    n_ok = 0; n_failed = 0
    fail_reasons: dict[str, int] = {}
    last_print = t0
    new_pngs: dict[int, bytes] = {}
    with mp.Pool(processes=args.workers, maxtasksperchild=200) as pool:
        for i, (row_idx, png, status) in enumerate(pool.imap_unordered(
                _render_one, todo, chunksize=4)):
            if png:
                new_pngs[row_idx] = png; n_ok += 1
            else:
                n_failed += 1
                fail_reasons[status] = fail_reasons.get(status, 0) + 1
            if time.time() - last_print > 60:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                print(f'  [{i+1}/{len(todo)}] ok={n_ok} failed={n_failed}  '
                      f'rate={rate:.1f}/s  eta={eta/60:.0f} min', flush=True)
                last_print = time.time()

    elapsed = time.time() - t0
    print(f'\nRender done in {elapsed/60:.1f} min — ok={n_ok}, failed={n_failed}')
    if fail_reasons:
        print(f'  failures: {fail_reasons}')

    # If --renders-out is given, dump just {row_idx: bytes} for later merge
    # across VMs, and stop here.
    if args.renders_out:
        out_p = Path(args.renders_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, 'wb') as f:
            _pickle.dump({'renders': new_pngs,
                          'slice': [s, e],
                          'n_ok': n_ok, 'n_failed': n_failed,
                          'fail_reasons': fail_reasons}, f)
        print(f'Wrote renders pickle → {out_p} '
              f'({out_p.stat().st_size // (1024*1024)} MB, {len(new_pngs)} pngs)')
        return

    # Build new composite_png list with both existing + new
    new_imgs = []
    for i, r in enumerate(imgs):
        if i in new_pngs:
            new_imgs.append({'bytes': new_pngs[i], 'path': None})
        else:
            new_imgs.append(r)

    # Construct new table (preserve all other columns)
    new_cols = {}
    for col in table.column_names:
        if col == 'composite_png':
            new_cols[col] = new_imgs
        else:
            new_cols[col] = table[col].to_pylist()

    new_table = pa.table(new_cols, schema=table.schema)
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, out, compression='snappy')
    print(f'\nWrote new parquet → {out}  ({out.stat().st_size // (1024*1024)} MB)')

    if args.upload:
        token = os.environ.get('HF_TOKEN')
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=PARQUET_PATH_IN_REPO,
            repo_id=REPO, repo_type='dataset',
            commit_message=f'Fill composite_png: +{n_ok} renders ({n_with}→{n_with+n_ok}/{n_total})',
        )
        print(f'Uploaded to HF {REPO}/{PARQUET_PATH_IN_REPO}')


if __name__ == '__main__':
    main()
