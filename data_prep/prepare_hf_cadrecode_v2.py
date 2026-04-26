"""Phase B: prepare new bench-style cad-recode SFT data and push to HF.

Pipeline (per sample):
  1. Sample .py from data/cad-recode-v1.5/train (sorted+shuffled w/ seed=42,
     offset+over-sample like the original commit-2f6396c packer)
  2. Exec the raw .py → STL via cadquery → trimesh
  3. Render 4-view 268×268 PNG via common.meshio.render_img
  4. Apply v2 rewriter to get bench-style code (fallback to raw on rewrite_fail)
  5. Pack into parquet shard
  6. Upload to Hula0401/cad-sft/cad-recode-bench/train-XXXXX-of-YYYYY.parquet

Uses incremental sharding: every `--shard-size` successes (default 2000), close
a shard, push to HF, free memory, continue. This keeps RAM bounded and gives
crash safety (resume by inspecting HF for last shard idx).

Resource defaults (12-core 16 GB box):
  --workers 2   (rendering is single-threaded but hits Open3D / OCP heavy)
  --shard-size 2000

Sampling matches commit-2f6396c semantics:
  candidates = sorted(rglob('*.py'))[shuffle(seed=42)][int(offset*1.3) : int((offset+n)*1.3)]
  Process candidates, count successes; stop when successes ≥ n.

Usage (smoke):
  set -a; source .env; set +a
  uv run python -m data_prep.prepare_hf_cadrecode_v2 --n 100 --workers 2 --no-upload

Usage (full 80k, resumes from last uploaded shard if interrupted):
  uv run python -m data_prep.prepare_hf_cadrecode_v2 \\
      --offset 20000 --n 80000 --workers 2 --shard-size 2000

Resume after crash:
  --start-shard <idx>   skip first N successful shards (still consumes candidates)
"""
from __future__ import annotations

import argparse
import io
import os
import random
import resource
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _free_ram_mb():
    """Return free RAM in MB (linux /proc/meminfo)."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _exec_render_rewrite(args):
    """Per-sample worker: exec → tessellate → render PNG → v2 rewrite.

    Returns dict with keys: stem, code (bench-style), png_bytes, error.
    On any exec/render failure, error is set and other fields may be None.
    """
    py_path = args
    stem_full = Path(py_path).parent.name + '/' + Path(py_path).stem
    try:
        raw = Path(py_path).read_text()
    except Exception as e:
        return {'stem': stem_full, 'code': None, 'png_bytes': None, 'error': f'read: {e}'}

    # Exec → mesh
    try:
        import cadquery as cq
        import trimesh
        ns = {'cq': cq}
        exec(raw, ns)
        r = ns.get('r')
        if r is None:
            return {'stem': stem_full, 'code': None, 'png_bytes': None, 'error': 'no r'}
        compound = r.val()
        verts, faces = compound.tessellate(0.001, 0.1)
        if len(verts) < 4 or len(faces) < 4:
            return {'stem': stem_full, 'code': None, 'png_bytes': None, 'error': 'degenerate'}
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
    except Exception as e:
        return {'stem': stem_full, 'code': None, 'png_bytes': None, 'error': f'exec: {type(e).__name__}'}

    # Render via tmpfile (render_img takes a path)
    import tempfile
    stl_tmp = None
    png_bytes = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tf:
            stl_tmp = tf.name
        mesh.export(stl_tmp)
        from common.meshio import render_img
        result = render_img(stl_tmp)
        img = result['video'][0]
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
    except Exception as e:
        return {'stem': stem_full, 'code': None, 'png_bytes': None, 'error': f'render: {type(e).__name__}'}
    finally:
        if stl_tmp and os.path.exists(stl_tmp):
            try: os.remove(stl_tmp)
            except Exception: pass

    # Rewrite
    try:
        from data_prep.rewrite_recode_to_benchcad_v2 import rewrite_source
        new_code = rewrite_source(raw)
    except Exception:
        new_code = raw  # fallback: keep raw recode style

    return {'stem': stem_full, 'code': new_code, 'png_bytes': png_bytes, 'error': None}


def _push_shard(rows, shard_idx, total_shards, repo_id, dst_prefix, out_dir, dry_run):
    """Pack `rows` into a parquet shard, upload to HF, cleanup local."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    fname_local = out_dir / f'train-{shard_idx:05d}-of-{total_shards:05d}.parquet'
    fname_remote = f'{dst_prefix}/train-{shard_idx:05d}-of-{total_shards:05d}.parquet'

    table = pa.table({
        'stem': [r['stem'] for r in rows],
        'code': [r['code'] for r in rows],
        'render_img': [{'bytes': r['png_bytes'], 'path': None} for r in rows],
    })
    pq.write_table(table, str(fname_local), compression='snappy')
    size_mb = fname_local.stat().st_size / 1024 / 1024
    print(f'  shard {shard_idx + 1}/{total_shards}: {len(rows)} rows, {size_mb:.1f} MB → {fname_local}', flush=True)

    if not dry_run:
        api = HfApi()
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(fname_local),
            path_in_repo=fname_remote,
            repo_id=repo_id,
            repo_type='dataset',
            token=os.environ.get('HF_TOKEN'),
            commit_message=f'phase-B shard {shard_idx + 1}/{total_shards}',
        )
        print(f'    uploaded in {time.time()-t0:.1f}s', flush=True)
    fname_local.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='data/cad-recode-v1.5/train')
    ap.add_argument('--n', type=int, default=80000)
    ap.add_argument('--offset', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--shard-size', type=int, default=2000)
    ap.add_argument('--out-dir', default='data/_phase_b_out')
    ap.add_argument('--repo-id', default='Hula0401/cad-sft')
    ap.add_argument('--dst-prefix', default='cad-recode-bench')
    ap.add_argument('--start-shard', type=int, default=0,
                    help='Resume — skip first N completed shards (consumes candidates)')
    ap.add_argument('--no-upload', action='store_true')
    ap.add_argument('--ram-floor-mb', type=int, default=1500,
                    help='Abort if /proc/meminfo MemAvailable drops below this')
    ap.add_argument('--max-tasks-per-child', type=int, default=100,
                    help='Recycle each worker after N tasks — mitigates '
                         'cadquery/open3d memory leaks (default 100)')
    args = ap.parse_args()

    if not args.no_upload and not os.environ.get('HF_TOKEN'):
        print('ERROR: HF_TOKEN not set; run `set -a; source .env; set +a`'); sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

    # Sampling: matches commit 2f6396c
    all_py = sorted(glob(os.path.join(args.data_dir, '**', '*.py'), recursive=True))
    print(f'discovered {len(all_py)} .py files in {args.data_dir}', flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(all_py)
    start = int(args.offset * 1.3)
    end = start + int(args.n * 1.3)
    candidates = all_py[start:end]
    print(f'sampling {len(candidates)} candidates (offset={args.offset}, target n={args.n})', flush=True)

    n_target = args.n
    n_per_shard = args.shard_size
    total_shards = (n_target + n_per_shard - 1) // n_per_shard

    # Bookkeeping
    successes_buf = []
    n_success = 0
    n_skip_for_resume = args.start_shard * n_per_shard
    n_error = 0
    error_kinds = {}
    shard_idx = args.start_shard

    print(f'\nLaunching workers={args.workers}, shard_size={n_per_shard}, total_shards={total_shards}', flush=True)
    print(f'(resume: skipping first {n_skip_for_resume} successes for start-shard={args.start_shard})', flush=True)

    t_start = time.time()
    last_log = t_start

    # Iterate over candidates, dispatch to worker pool.
    # max_tasks_per_child recycles workers periodically to mitigate memory
    # leaks in cadquery/OCP/open3d (observed: ~3 GB worker RSS after 3k tasks
    # in the first run, abort at ~12k successes when RAM dropped below 1.5 GB).
    with ProcessPoolExecutor(max_workers=args.workers,
                              max_tasks_per_child=args.max_tasks_per_child) as pool:
        future_iter = (pool.submit(_exec_render_rewrite, c) for c in candidates)
        # Throttle: keep at most 4*workers in flight to bound queue memory
        in_flight = []
        cand_iter = iter(candidates)
        for _ in range(min(4 * args.workers, len(candidates))):
            try:
                in_flight.append(pool.submit(_exec_render_rewrite, next(cand_iter)))
            except StopIteration:
                break

        while in_flight and n_success < n_target:
            done = []
            for fut in list(in_flight):
                if fut.done():
                    done.append(fut)
            if not done:
                # No completion yet — sleep briefly
                time.sleep(0.2)
                continue
            for fut in done:
                in_flight.remove(fut)
                try:
                    result = fut.result(timeout=1.0)
                except Exception as e:
                    n_error += 1
                    error_kinds['fut_err'] = error_kinds.get('fut_err', 0) + 1
                    result = None
                # Top up the queue
                try:
                    in_flight.append(pool.submit(_exec_render_rewrite, next(cand_iter)))
                except StopIteration:
                    pass

                if result is None:
                    continue
                if result['error']:
                    n_error += 1
                    kind = result['error'].split(':')[0]
                    error_kinds[kind] = error_kinds.get(kind, 0) + 1
                    continue

                # Already past target — discard extras (workers in flight may
                # still complete); no over-shoot into a phantom final shard.
                if n_success >= n_target:
                    continue

                # Success — but skip if still in resume-skip window
                if n_skip_for_resume > 0:
                    n_skip_for_resume -= 1
                    n_success += 1
                    continue

                successes_buf.append(result)
                n_success += 1

                # Shard boundary?
                if len(successes_buf) >= n_per_shard:
                    _push_shard(successes_buf, shard_idx, total_shards,
                                args.repo_id, args.dst_prefix, out_dir,
                                dry_run=args.no_upload)
                    successes_buf = []
                    shard_idx += 1
                    free = _free_ram_mb()
                    if free is not None and free < args.ram_floor_mb:
                        print(f'  !! RAM floor {free} MB < {args.ram_floor_mb} MB — aborting', flush=True)
                        # Cancel pending
                        for f in in_flight: f.cancel()
                        return

                # Periodic log
                now = time.time()
                if now - last_log > 30:
                    elapsed = now - t_start
                    rate = n_success / max(elapsed, 1)
                    free = _free_ram_mb()
                    print(f'  progress: {n_success}/{n_target}  rate={rate:.1f}/s  '
                          f'errors={n_error}  free_ram={free}MB  in_flight={len(in_flight)}',
                          flush=True)
                    last_log = now

    # Flush any leftover (could be < shard_size on final shard)
    if successes_buf:
        _push_shard(successes_buf, shard_idx, total_shards,
                    args.repo_id, args.dst_prefix, out_dir, dry_run=args.no_upload)
        shard_idx += 1

    elapsed = time.time() - t_start
    print(f'\n=== Phase B done ===', flush=True)
    print(f'  total successes: {n_success}/{n_target}', flush=True)
    print(f'  total errors: {n_error}  by kind: {error_kinds}', flush=True)
    print(f'  shards uploaded: {shard_idx} (start_shard={args.start_shard})', flush=True)
    print(f'  elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)  rate={n_success/max(elapsed,1):.2f}/s', flush=True)


if __name__ == '__main__':
    main()
