"""Import BenchCAD/benchcad-easy → repackage into Hula0401/cad-sft/benchcad-easy/.

Mixed pipeline because only ~12k of 109,804 rows ship with composite_png.

Per row:
  • If `composite_png` already present  → passthrough the PNG bytes directly.
  • Else (97k rows)                     → exec `gt_code` → cadquery compound →
                                          tessellate → trimesh → 4-view 268×268
                                          render via common.meshio.render_img.

Renders run in a 6-process pool with max_tasks_per_child=100 to bound the
known cadquery / open3d memory growth, and a per-task SIGALRM timeout to
abandon individual pathological shapes (twist/sweep/helix surfaces whose
tessellation is dense enough to hang Open3D for minutes).

Output schema matches the existing benchcad-simple-100k / cad-iso-106-175k
prefixes on Hula0401/cad-sft:
    stem, code, render_img{bytes,path}, family, difficulty,
    n_ops, ops_json, base_plane

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a

    # smoke (no upload)
    uv run python -m data_prep.import_benchcad_easy --n 30 --workers 4 --no-upload

    # full ~110k, resume from shard 4 (the 4 already-uploaded bellows shards)
    nohup uv run python -m data_prep.import_benchcad_easy \\
        --workers 6 --shard-size 2000 \\
        --start-shard 4 \\
        --per-task-timeout-sec 60 \\
        > logs/benchcad_easy_inner.log 2>&1 &
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


SRC_REPO       = 'BenchCAD/benchcad-easy'
SRC_PARQUET    = 'data/test-00000-of-00001.parquet'
DST_REPO       = 'Hula0401/cad-sft'
DST_PREFIX     = 'benchcad-easy'
UPLOAD_TIMEOUT = 300  # seconds — abandon hung HfApi.upload_file threads


# ---------------------------------------------------------------------------
# Worker side: exec gt_code → tessellate → render
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout('task exceeded budget')


def _render_from_gtcode(gt_code: str) -> bytes:
    """Exec cadquery gt_code → trimesh → 4-view 268×268 PNG bytes."""
    import cadquery as cq  # noqa: F401  (used implicitly via exec)
    import trimesh

    # Stub show_object so BenchCAD-style trailers don't NameError
    g = {'show_object': lambda *a, **kw: None}
    exec(gt_code, g)
    res = g.get('result') or g.get('r')
    if res is None:
        raise ValueError("no 'result' or 'r' in gt_code")
    compound = res.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    if len(verts) < 4 or len(faces) < 4:
        raise ValueError('degenerate mesh')
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)

    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    try:
        mesh.export(stl)
        from common.meshio import render_img
        result = render_img(stl)
        img = result['video'][0]
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    finally:
        try: os.unlink(stl)
        except Exception: pass


def _process_render_task(args) -> dict:
    """ProcessPool worker: render one row from gt_code (slow path).

    Inputs that already have an image are NOT dispatched here — see main loop.
    """
    idx, row, timeout_sec = args
    if timeout_sec and timeout_sec > 0:
        import signal
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(timeout_sec))

    try:
        png_bytes = _render_from_gtcode(row['gt_code'])
    except Exception as e:
        kind = 'timeout' if isinstance(e, _Timeout) else type(e).__name__
        return {'idx': idx, 'error': f'{kind}: {str(e)[:120]}', 'png_bytes': None}
    finally:
        if timeout_sec and timeout_sec > 0:
            import signal
            signal.alarm(0)
    return {'idx': idx, 'error': None, 'png_bytes': png_bytes}


# ---------------------------------------------------------------------------
# Main side: dispatch + shard packing + upload
# ---------------------------------------------------------------------------

def _build_record(row: dict, png_bytes: bytes) -> dict:
    return {
        'stem':       row['stem'],
        'code':       row['gt_code'],
        'render_img': {'bytes': png_bytes, 'path': None},
        'family':     row.get('family') or '',
        'difficulty': row.get('difficulty') or '',
        'n_ops':      int(row.get('feature_count') or 0),
        'ops_json':   row.get('ops_used') or '[]',
        'base_plane': row.get('base_plane') or '',
    }


def _push_shard(rows: list[dict], shard_idx: int, total_shards: int,
                out_dir: Path, dry_run: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fname_remote = f'{DST_PREFIX}/train-{shard_idx:05d}-of-{total_shards:05d}.parquet'
    fname_local  = out_dir / fname_remote.replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ('stem',       pa.string()),
        ('code',       pa.string()),
        ('render_img', pa.struct([
            ('bytes', pa.binary()),
            ('path',  pa.null()),
        ])),
        ('family',     pa.string()),
        ('difficulty', pa.string()),
        ('n_ops',      pa.int64()),
        ('ops_json',   pa.string()),
        ('base_plane', pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, fname_local, compression='snappy')
    sz = fname_local.stat().st_size / 1024 / 1024
    print(f'  shard {shard_idx + 1}/{total_shards}: {len(rows)} rows, {sz:.1f} MB '
          f'→ {fname_remote}', flush=True)

    if dry_run:
        fname_local.unlink()
        return

    from huggingface_hub import HfApi
    api = HfApi()
    for attempt in range(3):
        result = {'ok': False, 'err': None}
        def task():
            try:
                api.upload_file(
                    path_or_fileobj=str(fname_local),
                    path_in_repo=fname_remote,
                    repo_id=DST_REPO, repo_type='dataset',
                    token=os.environ.get('HF_TOKEN'),
                    commit_message=f'benchcad-easy shard {shard_idx + 1}/{total_shards}')
                result['ok'] = True
            except Exception as e:
                result['err'] = f'{type(e).__name__}: {e}'
        t0 = time.time()
        th = threading.Thread(target=task, daemon=True); th.start()
        th.join(timeout=UPLOAD_TIMEOUT)
        if result['ok']:
            print(f'    uploaded in {time.time()-t0:.1f}s', flush=True)
            break
        if th.is_alive():
            print(f'    !! hang > {UPLOAD_TIMEOUT}s, retry {attempt+1}/3', flush=True)
        elif result['err']:
            print(f'    upload err: {result["err"]}; retry {attempt+1}/3', flush=True)
    else:
        raise RuntimeError(f'upload failed after 3 retries: {fname_remote}')
    fname_local.unlink()


def _free_ram_mb() -> Optional[int]:
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',           type=int, default=None,
                    help='Cap total rows processed (default: all 109,804)')
    ap.add_argument('--shard-size',  type=int, default=2000)
    ap.add_argument('--workers',     type=int, default=6)
    ap.add_argument('--start-shard', type=int, default=0,
                    help='Skip first N output shards (= first N*shard_size rows). '
                         'Use 4 to resume after the bellows-prefix run that '
                         'crashed at row 8132.')
    ap.add_argument('--per-task-timeout-sec', type=int, default=60,
                    help='SIGALRM cap per render task. 60s skips pathological '
                         'shapes that would otherwise stall a worker.')
    ap.add_argument('--max-tasks-per-child', type=int, default=100)
    ap.add_argument('--ram-floor-mb', type=int, default=1500)
    ap.add_argument('--out-dir',     type=str, default='/tmp/benchcad_easy_shards')
    ap.add_argument('--no-upload',   action='store_true')
    args = ap.parse_args()

    if not args.no_upload and not os.environ.get('HF_TOKEN'):
        print('HF_TOKEN not set', file=sys.stderr); sys.exit(1)

    # ── 1. Read upstream parquet directly (no Image decode upfront)  ──────
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    print(f'Downloading {SRC_REPO}/{SRC_PARQUET} ...', flush=True)
    t0 = time.time()
    cache_dir = Path('/tmp/benchcad_easy_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_path = hf_hub_download(
        repo_id=SRC_REPO, filename=SRC_PARQUET,
        repo_type='dataset', token=os.environ.get('HF_TOKEN'),
        local_dir=str(cache_dir))
    sz = Path(src_path).stat().st_size / 1024 / 1024
    print(f'  done in {time.time()-t0:.1f}s ({sz:.1f} MB)', flush=True)

    # We only need a few columns; keep composite_png raw (struct of bytes).
    cols = ['stem', 'gt_code', 'composite_png',
            'family', 'difficulty', 'feature_count', 'ops_used', 'base_plane']
    table = pq.read_table(src_path, columns=cols)
    rows  = table.to_pylist()
    n_total = len(rows) if args.n is None else min(len(rows), args.n)
    rows = rows[:n_total]
    total_shards = (n_total + args.shard_size - 1) // args.shard_size
    print(f'  {len(rows)} rows in scope, {total_shards} output shards', flush=True)

    n_skip = args.start_shard * args.shard_size
    if n_skip:
        print(f'(resume: skipping rows 0..{n_skip-1} for start-shard={args.start_shard})',
              flush=True)
        rows = rows[n_skip:]

    # Stats: how many rows already have an image?
    n_have = sum(1 for r in rows if isinstance(r['composite_png'], dict)
                                    and r['composite_png'].get('bytes'))
    n_render = len(rows) - n_have
    print(f'  rows-to-process: {len(rows)}  '
          f'passthrough(have_img)={n_have}  render={n_render}', flush=True)

    out_dir = Path(args.out_dir)
    shard_buf: list[dict] = []
    shard_idx = args.start_shard
    n_done = 0
    n_render_err = 0
    err_kinds: dict[str, int] = {}
    t_start = time.time()
    last_log = time.time()

    # ── 2. Pool dispatch: only render-needing rows go through workers  ────
    # Passthrough rows are packed inline in main process — saves dispatch cost.
    # Even so, results must come out in the original order, so we keep an
    # in-flight queue of (idx, future) and drain in submission order.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        max_tasks_per_child=args.max_tasks_per_child,
    ) as pool:
        in_flight: list[tuple[int, dict, object]] = []  # (idx, row, future_or_None)
        max_in_flight = 4 * args.workers
        cursor = 0

        def _drain_one_in_order() -> bool:
            """Pop the head of in_flight if its result is ready; emit rec."""
            nonlocal n_done, n_render_err, shard_idx, last_log
            if not in_flight:
                return False
            idx, row, fut = in_flight[0]
            if fut is not None and not fut.done():
                return False
            in_flight.pop(0)
            if fut is None:
                # Passthrough row
                rec = _build_record(row, row['composite_png']['bytes'])
            else:
                res = fut.result()
                if res['error']:
                    n_render_err += 1
                    kind = res['error'].split(':', 1)[0]
                    err_kinds[kind] = err_kinds.get(kind, 0) + 1
                    n_done += 1
                    return True  # skip this row
                rec = _build_record(row, res['png_bytes'])
            shard_buf.append(rec)
            n_done += 1
            if len(shard_buf) >= args.shard_size:
                _push_shard(shard_buf, shard_idx, total_shards, out_dir,
                            dry_run=args.no_upload)
                shard_buf.clear()
                shard_idx += 1
            now = time.time()
            if now - last_log > 30:
                rate = n_done / (now - t_start + 1e-6)
                remaining = len(rows) - n_done
                eta = remaining / max(rate, 1e-6) / 60
                ram = _free_ram_mb()
                print(f'  [{n_done}/{len(rows)}] {rate:.2f}/s '
                      f'ETA {eta:.1f}min err={n_render_err} '
                      f'ram_free={ram}MB',
                      flush=True)
                last_log = now
            return True

        # Main dispatch loop
        while cursor < len(rows) or in_flight:
            # Submit until in_flight is full
            while cursor < len(rows) and len(in_flight) < max_in_flight:
                row = rows[cursor]
                if isinstance(row['composite_png'], dict) and row['composite_png'].get('bytes'):
                    # Fast passthrough — no pool dispatch
                    in_flight.append((cursor, row, None))
                else:
                    fut = pool.submit(_process_render_task,
                                      (cursor, row, args.per_task_timeout_sec))
                    in_flight.append((cursor, row, fut))
                cursor += 1
            # Drain at least one in-order result before we try more
            if not _drain_one_in_order():
                # Head not ready and pool is full — sleep briefly
                time.sleep(0.05)
                # If RAM is dangerously low, abort gracefully
                ram = _free_ram_mb()
                if ram is not None and ram < args.ram_floor_mb:
                    print(f'!! RAM_FLOOR breached (free={ram}MB < {args.ram_floor_mb}MB), '
                          f'aborting after current shard', flush=True)
                    break

    # Flush final partial shard
    if shard_buf:
        _push_shard(shard_buf, shard_idx, total_shards, out_dir,
                    dry_run=args.no_upload)
        shard_idx += 1

    total_min = (time.time() - t_start) / 60
    print(f'\nDone in {total_min:.1f}min. processed={n_done} '
          f'render_errors={n_render_err} '
          f'output_shards={shard_idx - args.start_shard}', flush=True)
    if err_kinds:
        print(f'  error breakdown: {err_kinds}', flush=True)
    print(f'  → https://huggingface.co/datasets/{DST_REPO}/tree/main/{DST_PREFIX}',
          flush=True)


if __name__ == '__main__':
    main()
