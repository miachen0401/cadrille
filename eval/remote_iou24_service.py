"""Remote IoU-24 scoring service for commercial-LLM CAD code generations.

Polls an HF dataset of model-generated cadquery codes (each row = one bench
case, each `cq_<model>` column = one LLM's prediction), computes
rotation-invariant IoU-24 vs the GT meshes, and pushes per-case + summary
results to an HF score repo.

Architecture:

    GT repo         (read-only, static 200 cases)
        └── BenchCAD/cad_bench_200
            data/train-00000-of-00001.parquet
              columns: stem, gt_code, ...

    Predictions    (the user updates this; we poll for new commits)
        └── qixiaoqi/cad_bench_200            ← --pred-repo
            data/train-00000-of-00001.parquet
              columns: stem, gt_code, cq_<model_a>, cq_<model_b>, ...
              Each `cq_*` column is one LLM's prediction code.

    Score repo     (we write here)
        └── Hula0401/bench_score              ← --score-repo
            per_case/{sha[:8]}/{model}.parquet
              one row per stem with iou, iou_24, rot_idx, exec_ok, ...
            summary.parquet
              one row per (sha[:8], model) with mean/median/pass-rate

Usage:

    # one-shot (debug)
    uv run python -m eval.remote_iou24_service \\
        --pred-repo qixiaoqi/cad_bench_200 \\
        --once

    # polling daemon (30 min default)
    nohup uv run python -m eval.remote_iou24_service \\
        --pred-repo qixiaoqi/cad_bench_200 \\
        --poll-interval 1800 \\
        --workers 12 \\
        > logs/remote_iou24.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.metrics import compute_metrics_24  # noqa: E402

# ─── Defaults (CLI-overridable, NOT hardcoded into business logic) ────────────
DEFAULT_GT_REPO    = 'BenchCAD/cad_bench_200'
DEFAULT_SCORE_REPO = 'Hula0401/bench_score'
DEFAULT_PARQUET    = 'data/train-00000-of-00001.parquet'

# Where to keep local state + GT-mesh cache (not on root partition)
DEFAULT_STATE_DIR  = Path('/ephemeral/data/remote_iou24_state')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _code_hash(code: str | None) -> str:
    """16-char hash of normalized code; '' → 'none'."""
    if not code:
        return 'none'
    norm = ' '.join((code or '').split())
    return hashlib.md5(norm.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _pred_columns(table) -> list[str]:
    """Return column names that hold model predictions (heuristic: `cq_*`).

    Excludes the GT (`gt_code`). The user-supplied schema uses one cq_<model>
    column per LLM, so this covers the public layout. If a future repo adds
    predictions under a different naming convention, extend here.
    """
    return [c for c in table.column_names
            if c.startswith('cq_') and c != 'gt_code']


def _gt_template(gt_code: str) -> str:
    """Wrap GT code into a self-contained subprocess script that writes STL."""
    return (
        'import cadquery as cq, sys, trimesh\n'
        + gt_code
        + '\n'
        + ('try: r = result\n'
           'except NameError: r = show_object_arg if "show_object_arg" in dir() else None\n'
           'compound = r.val(); v, f = compound.tessellate(0.001, 0.1)\n'
           'mesh = trimesh.Trimesh([(p.x, p.y, p.z) for p in v], f)\n'
           'mesh.apply_translation(-(mesh.bounds[0]+mesh.bounds[1])/2.0)\n'
           'ext = float(max(mesh.extents))\n'
           'if ext > 1e-7: mesh.apply_scale(2.0/ext)\n'
           'mesh.export(sys.argv[1])\n')
    )


def _exec_gt_to_stl(gt_code: str, out_stl: Path, timeout: float = 60.0) -> bool:
    """Subprocess-exec gt_code, normalize mesh, write STL. True on success."""
    import subprocess, tempfile
    if out_stl.exists() and out_stl.stat().st_size > 100:
        return True
    script = _gt_template(gt_code)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run([sys.executable, '-c', script, str(out_stl)],
                           capture_output=True, timeout=timeout)
        return r.returncode == 0 and out_stl.exists() and out_stl.stat().st_size > 100
    except Exception:
        return False


# ─── HF helpers (small wrappers, kept thin so testing is trivial) ─────────────

def _hf_api(token_env: str = 'HF_TOKEN'):
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get(token_env))


def _head_sha(api, repo: str) -> str | None:
    try:
        refs = api.list_repo_refs(repo, repo_type='dataset')
        for b in refs.branches:
            if b.name == 'main':
                return b.target_commit
    except Exception as e:
        print(f'[warn] head_sha probe failed for {repo}: {e}', flush=True)
    return None


def _download_parquet(api, repo: str, path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, path_in_repo, repo_type='dataset',
                                 token=os.environ.get('HF_TOKEN')))


# ─── Scoring (one prediction at a time, parallel-safe) ────────────────────────

def _score_one(args: tuple) -> dict:
    """Worker: run compute_metrics_24 for one (model, stem, code) vs cached STL.

    Returns a dict ready to drop into the per_case parquet.
    """
    (model, stem, gen_code, gt_stl_path, code_hash,
     timeout, early_stop, sha_short, scored_at) = args
    if not gen_code or not gen_code.strip():
        return {
            'sha_short': sha_short, 'model': model, 'stem': stem,
            'code_hash': code_hash, 'iou': -1.0, 'iou_24': None, 'rot_idx': -1,
            'exec_ok': False, 'reason': 'empty_pred', 'latency_s': 0.0,
            'scored_at': scored_at,
        }
    t0 = time.time()
    try:
        iou, _cd, iou_24, rot_idx = compute_metrics_24(
            gen_code, gt_stl_path, timeout=timeout,
            iou_24_early_stop=early_stop)
    except Exception as e:
        return {
            'sha_short': sha_short, 'model': model, 'stem': stem,
            'code_hash': code_hash, 'iou': -1.0, 'iou_24': None, 'rot_idx': -1,
            'exec_ok': False, 'reason': f'err:{type(e).__name__}',
            'latency_s': round(time.time() - t0, 3), 'scored_at': scored_at,
        }
    return {
        'sha_short': sha_short, 'model': model, 'stem': stem,
        'code_hash': code_hash,
        'iou': float(iou),
        'iou_24': None if iou_24 is None else round(float(iou_24), 4),
        'rot_idx': int(rot_idx),
        'exec_ok': iou >= 0,
        'reason': 'ok' if iou >= 0 else 'pred_exec_fail',
        'latency_s': round(time.time() - t0, 3),
        'scored_at': scored_at,
    }


# ─── State ────────────────────────────────────────────────────────────────────

def _load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {'last_sha': None, 'scored': {}}
    with state_file.open() as f:
        s = json.load(f)
    s.setdefault('scored', {})
    return s


def _save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_file)


def _scored_key(model: str, stem: str, code_hash: str) -> str:
    return f'{model}::{stem}::{code_hash}'


# ─── Push to score repo ───────────────────────────────────────────────────────

def _push_results(api, score_repo: str, sha_short: str,
                   per_model_rows: dict[str, list[dict]],
                   summary_rows: list[dict],
                   state_dir: Path) -> None:
    """Write per_case/{sha}/{model}.parquet + a fresh summary.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = state_dir / '_push_staging' / sha_short
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_case
    for model, rows in per_model_rows.items():
        if not rows:
            continue
        safe_model = model.replace('/', '_').replace(' ', '_')
        out_p = out_dir / f'{safe_model}.parquet'
        pq.write_table(pa.Table.from_pylist(rows), out_p, compression='snappy')
        print(f'  staged per_case/{sha_short}/{safe_model}.parquet '
              f'({len(rows)} rows)', flush=True)

    # summary (full rebuild — small)
    summary_p = out_dir / 'summary.parquet'
    if summary_rows:
        pq.write_table(pa.Table.from_pylist(summary_rows), summary_p,
                        compression='snappy')

    # upload all files in one folder push
    print(f'Pushing → {score_repo}/per_case/{sha_short}/ ...', flush=True)
    for model in per_model_rows:
        if not per_model_rows[model]:
            continue
        safe_model = model.replace('/', '_').replace(' ', '_')
        api.upload_file(
            path_or_fileobj=str(out_dir / f'{safe_model}.parquet'),
            path_in_repo=f'per_case/{sha_short}/{safe_model}.parquet',
            repo_id=score_repo, repo_type='dataset',
            commit_message=f'iou_24 results: {model} @ pred_sha={sha_short}',
        )
    if summary_rows:
        api.upload_file(
            path_or_fileobj=str(summary_p),
            path_in_repo='summary.parquet',
            repo_id=score_repo, repo_type='dataset',
            commit_message=f'iou_24 summary update @ pred_sha={sha_short}',
        )


def _build_summary_row(sha_short: str, model: str, rows: list[dict]) -> dict:
    """Roll-up per-(sha, model) for summary.parquet."""
    import statistics as _st
    iou24s = [r['iou_24'] for r in rows if r['iou_24'] is not None]
    n = len(rows)
    n_exec = sum(1 for r in rows if r['exec_ok'])
    return {
        'sha_short': sha_short,
        'model': model,
        'n_total': n,
        'n_exec_ok': n_exec,
        'exec_rate': n_exec / max(n, 1),
        'mean_iou_24': float(_st.mean(iou24s)) if iou24s else 0.0,
        'median_iou_24': float(_st.median(iou24s)) if iou24s else 0.0,
        'pass_iou_24_0.5': sum(1 for v in iou24s if v > 0.5) / max(n, 1),
        'pass_iou_24_0.7': sum(1 for v in iou24s if v > 0.7) / max(n, 1),
        'updated_at': _now_iso(),
    }


# ─── Main scoring pass ────────────────────────────────────────────────────────

def score_pass(args, state: dict) -> bool:
    """Pull predictions, score new (model, stem, code_hash) tuples, push.

    Returns True if anything was scored this pass (= caller can save state).
    """
    api = _hf_api()

    # (1) Detect new commit on prediction repo
    sha = _head_sha(api, args.pred_repo)
    if sha is None:
        print('  could not get head_sha; skipping', flush=True)
        return False
    sha_short = sha[:8]
    print(f'  pred_repo={args.pred_repo} sha={sha_short} '
          f'(last={(state.get("last_sha") or "(none)")[:8]})', flush=True)
    if state.get('last_sha') == sha and not args.force:
        print('  no commit change since last pass; skipping (use --force to override).',
              flush=True)
        return False

    # (2) Pull GT + predictions parquets
    print(f'  downloading GT parquet from {args.gt_repo} ...', flush=True)
    gt_p = _download_parquet(api, args.gt_repo, args.parquet_path)
    print(f'  downloading predictions parquet from {args.pred_repo} ...', flush=True)
    pred_p = _download_parquet(api, args.pred_repo, args.parquet_path)

    import pyarrow.parquet as pq
    gt_t = pq.read_table(gt_p, columns=['stem', 'gt_code'])
    pred_t = pq.read_table(pred_p)

    # Build {stem: gt_code} from GT repo (authoritative)
    gt_lookup = {row['stem']: row['gt_code']
                 for row in gt_t.to_pylist()}
    print(f'  GT cases: {len(gt_lookup)}', flush=True)

    # (3) Materialize GT meshes (cache on disk, reused across polls)
    print('  materializing GT meshes (cached) ...', flush=True)
    gt_dir = args.state_dir / 'gt_meshes'
    gt_stl_paths: dict[str, Path] = {}
    n_built, n_failed = 0, 0
    for stem, gt_code in gt_lookup.items():
        stl_p = gt_dir / f'{stem}.stl'
        if not stl_p.exists() or stl_p.stat().st_size <= 100:
            ok = _exec_gt_to_stl(gt_code, stl_p)
            if ok: n_built += 1
            else: n_failed += 1
        if stl_p.exists() and stl_p.stat().st_size > 100:
            gt_stl_paths[stem] = stl_p
    print(f'  GT meshes: {len(gt_stl_paths)} ready, {n_built} newly built, '
          f'{n_failed} failed', flush=True)

    # (4) For each cq_<model> column × each row, score if not already scored
    pred_cols = _pred_columns(pred_t)
    print(f'  prediction columns: {pred_cols}', flush=True)

    pred_rows = pred_t.to_pylist()
    tasks: list[tuple] = []
    scored_at = _now_iso()
    early_stop = args.early_stop
    timeout = args.eval_timeout
    for col in pred_cols:
        model = col[3:] if col.startswith('cq_') else col  # cq_gpt4o → gpt4o
        for row in pred_rows:
            stem = row['stem']
            if stem not in gt_stl_paths:
                continue
            gen_code = row.get(col)
            ch = _code_hash(gen_code)
            key = _scored_key(model, stem, ch)
            if key in state['scored'] and not args.force:
                continue
            tasks.append((model, stem, gen_code or '', str(gt_stl_paths[stem]),
                          ch, timeout, early_stop, sha_short, scored_at))

    if not tasks:
        print('  no new (model, stem, code_hash) tuples to score', flush=True)
        # but still update last_sha so we don't repeat the diff next poll
        state['last_sha'] = sha
        return True

    print(f'  scoring {len(tasks)} new predictions with {args.workers} workers ...',
          flush=True)
    new_results: list[dict] = []
    t_start = time.time()
    last_log = t_start
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_score_one, task) for task in tasks]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            new_results.append(r)
            if time.time() - last_log > 30:
                rate = (i + 1) / (time.time() - t_start)
                eta_s = (len(tasks) - i - 1) / max(rate, 1e-3)
                print(f'    [{i+1}/{len(tasks)}] {rate:.1f}/s  eta={eta_s/60:.1f} min',
                      flush=True)
                last_log = time.time()

    print(f'  scored {len(new_results)} in {(time.time()-t_start)/60:.1f} min',
          flush=True)

    # (5) Update scored set + push
    for r in new_results:
        key = _scored_key(r['model'], r['stem'], r['code_hash'])
        state['scored'][key] = {
            'iou_24': r['iou_24'], 'rot_idx': r['rot_idx'],
            'exec_ok': r['exec_ok'], 'sha_short': r['sha_short'],
        }

    # group by model for per_case files
    per_model: dict[str, list[dict]] = {}
    for r in new_results:
        per_model.setdefault(r['model'], []).append(r)

    # build summary rows from CURRENT pass results (NOT the cumulative state).
    # An aggregate roll-up across all SHAs would require either re-pulling the
    # existing summary or storing all per_case rows in state — keep simple.
    summary_rows = [_build_summary_row(sha_short, m, rows)
                    for m, rows in per_model.items()]

    _push_results(api, args.score_repo, sha_short, per_model, summary_rows,
                  args.state_dir)

    state['last_sha'] = sha
    return True


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-repo', required=True,
                    help='HF dataset id of predictions, e.g. qixiaoqi/cad_bench_200')
    ap.add_argument('--gt-repo', default=DEFAULT_GT_REPO,
                    help=f'HF dataset id of GT (default {DEFAULT_GT_REPO})')
    ap.add_argument('--score-repo', default=DEFAULT_SCORE_REPO,
                    help=f'HF dataset id to push results to (default {DEFAULT_SCORE_REPO})')
    ap.add_argument('--parquet-path', default=DEFAULT_PARQUET)
    ap.add_argument('--state-dir', type=Path, default=DEFAULT_STATE_DIR,
                    help='Local dir for state.json + cached GT STLs')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--eval-timeout', type=float, default=300.0,
                    help='subprocess timeout per scoring call (s)')
    ap.add_argument('--early-stop', type=float, default=0.95,
                    help='iou_24 early-stop threshold across the 24 rotations')
    ap.add_argument('--poll-interval', type=int, default=1800,
                    help='seconds between polls (default 30 min)')
    ap.add_argument('--once', action='store_true',
                    help='run a single scoring pass and exit')
    ap.add_argument('--force', action='store_true',
                    help='ignore state cache; rescore everything')
    args = ap.parse_args()

    args.state_dir = Path(args.state_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_dir / 'state.json'

    print(f'[remote-iou24] start  pred_repo={args.pred_repo}  '
          f'score_repo={args.score_repo}  state_dir={args.state_dir}',
          flush=True)

    if args.once:
        state = _load_state(state_file)
        if score_pass(args, state):
            _save_state(state_file, state)
        return

    while True:
        state = _load_state(state_file)
        try:
            changed = score_pass(args, state)
            if changed:
                _save_state(state_file, state)
        except KeyboardInterrupt:
            print('[remote-iou24] interrupted', flush=True)
            raise
        except Exception as e:
            print(f'[remote-iou24] pass failed: {type(e).__name__}: {e}',
                  flush=True)
        print(f'[remote-iou24] sleeping {args.poll_interval}s ...', flush=True)
        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
