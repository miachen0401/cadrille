"""Phase F: import BenchCAD/cad_simple_ops_100k → render PNG → push to
Hula0401/cad-sft/benchcad-simple-100k.

Pipeline (per row):
  1. Read row from upstream parquet shard (stem, code, step_bytes, family,
     difficulty, n_ops, ops_json, base_plane)
  2. Strip OCP HashCode shim header from `code` (model shouldn't learn that)
  3. Load STEP from `step_bytes` → tessellate → trimesh
     (much faster than exec'ing the python: ~30ms vs ~1s)
  4. Render 4-view 268×268 PNG via common.meshio.render_img
  5. Pack (stem, code, render_img, family, difficulty, n_ops, ops_json,
     base_plane) into output parquet shard
  6. Upload to Hula0401/cad-sft/benchcad-simple-100k/

Reuses the RAM-safe scaffolding from prepare_hf_cadrecode_v2.py:
  - max_tasks_per_child=100 worker recycle (OCP/open3d leak mitigation)
  - incremental shard upload, free memory after each shard
  - RAM floor abort

Code is **already** BenchCAD shell style (no rewrite needed). We just clean
the shim and trust the upstream geometry.

Usage (smoke):
  set -a; source .env; set +a
  uv run python -m data_prep.import_benchcad_simple --n 50 --workers 2 --no-upload

Usage (full ~99k):
  uv run python -m data_prep.import_benchcad_simple \\
      --workers 4 --shard-size 2000 --max-tasks-per-child 100
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


SHIM_END_MARKER = '# --- end shim ---'


def _clean_code(code: str) -> str:
    """Strip OCP HashCode shim from code header. Keep show_object trailer
    (matches recode-bench format for training-data consistency)."""
    if SHIM_END_MARKER in code:
        code = code.split(SHIM_END_MARKER, 1)[1].lstrip('\n')
    return code


def _free_ram_mb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _step_to_mesh(step_bytes: bytes):
    """Load STEP from bytes → trimesh. Returns Trimesh or raises."""
    import cadquery as cq
    import trimesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
        f.write(step_bytes)
        step_path = f.name
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != IFSelect_RetDone:
            raise ValueError('STEP read failed')
        reader.TransferRoots()
        shape = reader.OneShape()
        compound = cq.Compound._fromTopoDS(shape) if hasattr(cq.Compound, '_fromTopoDS') \
            else cq.Shape.cast(shape)
        verts, faces = compound.tessellate(0.001, 0.1)
        if len(verts) < 4 or len(faces) < 4:
            raise ValueError('degenerate mesh')
        return trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
    finally:
        try: os.unlink(step_path)
        except Exception: pass


def _render_png_bytes(mesh) -> bytes:
    """Render mesh → 4-view PNG bytes via common.meshio.render_img."""
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl_path = f.name
    try:
        mesh.export(stl_path)
        from common.meshio import render_img
        result = render_img(stl_path)
        img = result['video'][0]
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    finally:
        try: os.unlink(stl_path)
        except Exception: pass


def _process_row(row: dict) -> dict:
    """Per-row worker: clean code + STEP→mesh + render → packed dict.

    Returns {stem, code, png_bytes, family, difficulty, n_ops, ops_json,
             base_plane, error}. On any failure, error is set; other fields
    may be None.
    """
    stem = row['stem']
    try:
        code_clean = _clean_code(row['code'])
        mesh = _step_to_mesh(row['step_bytes'])
        png_bytes = _render_png_bytes(mesh)
    except Exception as e:
        return {'stem': stem, 'error': f'{type(e).__name__}: {str(e)[:120]}',
                'code': None, 'png_bytes': None,
                'family': row.get('family'), 'difficulty': row.get('difficulty'),
                'n_ops': row.get('n_ops'), 'ops_json': row.get('ops_json'),
                'base_plane': row.get('base_plane')}
    return {
        'stem': stem,
        'code': code_clean,
        'png_bytes': png_bytes,
        'family': row.get('family'),
        'difficulty': row.get('difficulty'),
        'n_ops': row.get('n_ops'),
        'ops_json': row.get('ops_json'),
        'base_plane': row.get('base_plane'),
        'error': None,
    }


def _push_shard(rows, shard_idx, total_shards, repo_id, dst_prefix, out_dir, dry_run):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    fname_local = out_dir / f'train-{shard_idx:05d}-of-{total_shards:05d}.parquet'
    fname_remote = f'{dst_prefix}/train-{shard_idx:05d}-of-{total_shards:05d}.parquet'

    table = pa.table({
        'stem': [r['stem'] for r in rows],
        'code': [r['code'] for r in rows],
        'render_img': [{'bytes': r['png_bytes'], 'path': None} for r in rows],
        'family': [r['family'] for r in rows],
        'difficulty': [r['difficulty'] for r in rows],
        'n_ops': [r['n_ops'] for r in rows],
        'ops_json': [r['ops_json'] for r in rows],
        'base_plane': [r['base_plane'] for r in rows],
    })
    pq.write_table(table, str(fname_local), compression='snappy')
    size_mb = fname_local.stat().st_size / 1024 / 1024
    print(f'  shard {shard_idx + 1}/{total_shards}: {len(rows)} rows, {size_mb:.1f} MB', flush=True)

    if not dry_run:
        api = HfApi()
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(fname_local),
            path_in_repo=fname_remote,
            repo_id=repo_id,
            repo_type='dataset',
            token=os.environ.get('HF_TOKEN'),
            commit_message=f'phase-F shard {shard_idx + 1}/{total_shards}',
        )
        print(f'    uploaded in {time.time()-t0:.1f}s', flush=True)
    fname_local.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src-repo', default='BenchCAD/cad_simple_ops_100k')
    ap.add_argument('--src-prefix', default='data')
    ap.add_argument('--src-shard-prefix', default='train',
                    help='Per-shard filename prefix; e.g. cad_simple_ops_100k '
                         "uses 'train-XXXXX-of-NNNNN.parquet', cad_iso_106 "
                         "uses 'data-XXXXX-of-NNNNN.parquet'.")
    ap.add_argument('--rows-per-src-shard', type=int, default=8240,
                    help='Used only for shard-count estimation; default '
                         'matches cad_simple_ops_100k (~8240 rows/shard).')
    ap.add_argument('--total-src-shards', type=int, default=12)
    ap.add_argument('--repo-id', default='Hula0401/cad-sft')
    ap.add_argument('--dst-prefix', default='benchcad-simple-100k')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--shard-size', type=int, default=2000)
    ap.add_argument('--out-dir', default='data/_phase_f_out')
    ap.add_argument('--cache-dir', default='data/_cache_phase_f')
    ap.add_argument('--start-shard', type=int, default=0,
                    help='Skip first N output shards (resume after crash)')
    ap.add_argument('--n', type=int, default=None,
                    help='Process at most N rows total (for smoke test)')
    ap.add_argument('--no-upload', action='store_true')
    ap.add_argument('--max-tasks-per-child', type=int, default=100)
    ap.add_argument('--ram-floor-mb', type=int, default=1500)
    args = ap.parse_args()

    if not args.no_upload and not os.environ.get('HF_TOKEN'):
        print('ERROR: HF_TOKEN not set'); sys.exit(1)

    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Estimate total output shards
    rows_per_src_shard = args.rows_per_src_shard
    total_src_rows = rows_per_src_shard * args.total_src_shards
    if args.n is not None:
        total_src_rows = min(total_src_rows, args.n)
    total_shards_out = (total_src_rows + args.shard_size - 1) // args.shard_size
    n_skip_for_resume = args.start_shard * args.shard_size

    print(f'workers={args.workers}  shard_size={args.shard_size}  '
          f'estimated_output_shards={total_shards_out}', flush=True)
    print(f'(resume: skipping first {n_skip_for_resume} successes for start-shard={args.start_shard})', flush=True)

    # Stream rows from upstream parquet shards, dispatch to workers, batch into output shards.
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    shard_idx = args.start_shard
    successes_buf = []
    n_success = 0
    n_error = 0
    error_kinds = {}
    n_dispatched = 0
    skip_remaining = n_skip_for_resume

    t_start = time.time()
    last_log = t_start

    # Use multiprocessing.Pool — has `maxtasksperchild` on Python 3.10
    # (concurrent.futures.ProcessPoolExecutor's `max_tasks_per_child` is 3.11+).
    pool = mp.Pool(processes=args.workers, maxtasksperchild=args.max_tasks_per_child)
    try:
        in_flight = []  # list of AsyncResult
        max_in_flight = 4 * args.workers

        # Iterate over upstream parquet shards lazily
        for src_shard_i in range(args.total_src_shards):
            if args.n is not None and n_dispatched >= args.n:
                break
            src_fname = f'{args.src_prefix}/{args.src_shard_prefix}-{src_shard_i:05d}-of-{args.total_src_shards:05d}.parquet'
            print(f'[src shard {src_shard_i + 1}/{args.total_src_shards}] downloading {src_fname} ...', flush=True)
            t0 = time.time()
            p = hf_hub_download(args.src_repo, src_fname, repo_type='dataset',
                                 token=os.environ.get('HF_TOKEN'),
                                 local_dir=str(cache_dir))
            t = pq.read_table(p)
            rows = t.to_pylist()
            print(f'  loaded {len(rows)} rows in {time.time()-t0:.1f}s', flush=True)

            # Dispatch + drain loop
            row_idx = 0
            while row_idx < len(rows) or in_flight:
                # Top up
                while len(in_flight) < max_in_flight and row_idx < len(rows):
                    if args.n is not None and n_dispatched >= args.n:
                        break
                    in_flight.append(pool.apply_async(_process_row, (rows[row_idx],)))
                    row_idx += 1
                    n_dispatched += 1
                if not in_flight:
                    break

                done = [r for r in in_flight if r.ready()]
                if not done:
                    time.sleep(0.1)
                    continue
                for r in done:
                    in_flight.remove(r)
                    try:
                        result = r.get(timeout=1.0)
                    except Exception as e:
                        n_error += 1
                        error_kinds['fut_err'] = error_kinds.get('fut_err', 0) + 1
                        continue
                    if result['error']:
                        n_error += 1
                        kind = result['error'].split(':')[0]
                        error_kinds[kind] = error_kinds.get(kind, 0) + 1
                        continue
                    if skip_remaining > 0:
                        skip_remaining -= 1
                        n_success += 1
                        continue
                    successes_buf.append(result)
                    n_success += 1

                    if len(successes_buf) >= args.shard_size:
                        _push_shard(successes_buf, shard_idx, total_shards_out,
                                    args.repo_id, args.dst_prefix, out_dir,
                                    dry_run=args.no_upload)
                        successes_buf = []
                        shard_idx += 1
                        free = _free_ram_mb()
                        if free is not None and free < args.ram_floor_mb:
                            print(f'  !! RAM floor {free} MB < {args.ram_floor_mb} MB — aborting', flush=True)
                            return

                    now = time.time()
                    if now - last_log > 30:
                        elapsed = now - t_start
                        rate = n_success / max(elapsed, 1)
                        free = _free_ram_mb()
                        print(f'  progress: success={n_success}  errors={n_error}  '
                              f'rate={rate:.1f}/s  free_ram={free}MB  in_flight={len(in_flight)}',
                              flush=True)
                        last_log = now

            # Cleanup downloaded source shard
            try: Path(p).unlink()
            except Exception: pass
    finally:
        pool.close()
        pool.join()

    # Flush remaining buffer
    if successes_buf:
        _push_shard(successes_buf, shard_idx, total_shards_out,
                    args.repo_id, args.dst_prefix, out_dir, dry_run=args.no_upload)
        shard_idx += 1

    elapsed = time.time() - t_start
    print(f'\n=== Phase F done ===', flush=True)
    print(f'  total successes: {n_success}', flush=True)
    print(f'  total errors:    {n_error}  by kind: {error_kinds}', flush=True)
    print(f'  shards uploaded: {shard_idx} (start_shard={args.start_shard})', flush=True)
    print(f'  elapsed:         {elapsed:.1f}s ({elapsed/60:.1f} min)', flush=True)
    print(f'  rate:            {n_success/max(elapsed,1):.2f}/s', flush=True)


if __name__ == '__main__':
    main()
