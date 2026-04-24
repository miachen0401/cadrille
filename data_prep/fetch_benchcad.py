"""Materialize BenchCAD/cad_bench into local files matching CadRecodeDataset-style layout.

Input:  BenchCAD/cad_bench (public, 20143 rows, parquet)
Output: data/benchcad/
          train/{stem}.py             gt_code
          train/{stem}_render.png     composite_png (pre-rendered, 4-view composite)
          train/{stem}.stl            exec(gt_code) → tessellate → trimesh.export
          train.pkl                   list of {uid, py_path, mesh_path, png_path, description}
          val/… same shape
          _skipped.jsonl              rows whose STL generation failed

Split: 90/10 train/val by hash(stem).

Run: set -a; source .env; set +a; uv run python tools/fetch_benchcad.py
Flags:
  --out DIR          default data/benchcad
  --parquet PATH     already-cached parquet (skip HF download)
  --workers N        parallel STL generation (default 8)
  --limit N          cap number of rows (smoke)
  --force            overwrite existing files
"""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import traceback
from pathlib import Path


def _split(stem: str, val_frac: float = 0.1) -> str:
    """Deterministic split by stem hash — same stem always lands in same split."""
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(1000 * val_frac) else "train"


def _compose_description(row: dict) -> str:
    """A short natural-language prompt from metadata. Same shape as Text2CADDataset 'description'."""
    family = row.get("family", "") or ""
    diff = row.get("difficulty", "") or ""
    ops = row.get("ops_used", "") or ""
    parts = []
    if family:
        parts.append(f"{family.replace('_', ' ')}")
    if diff:
        parts.append(f"[{diff}]")
    if ops:
        parts.append(f"ops={ops}")
    s = " ".join(parts).strip()
    return s or "Generate cadquery code"


def _exec_to_stl(args):
    """Worker: exec gt_code, compound.tessellate → trimesh.export(.stl)."""
    py_path, stl_path = args
    try:
        import cadquery as cq  # noqa
        import trimesh
        src = Path(py_path).read_text()
        ns = {"cq": cq, "__name__": "__main__"}
        # gt_code uses show_object — stub it to a no-op, capture 'result' instead
        captured = {}
        ns["show_object"] = lambda x, *a, **k: captured.setdefault("r", x)
        exec(src, ns)
        obj = captured.get("r") or ns.get("result") or ns.get("r")
        if obj is None:
            return (stl_path, "no result")
        compound = obj.val() if hasattr(obj, "val") else obj
        vertices, faces = compound.tessellate(0.001, 0.1)
        if not vertices or not faces:
            return (stl_path, "empty tessellation")
        mesh = trimesh.Trimesh(
            [(v.x, v.y, v.z) for v in vertices],
            faces,
        )
        mesh.export(stl_path)
        return (stl_path, None)
    except Exception as e:
        return (stl_path, f"{type(e).__name__}: {e}".splitlines()[0][:200])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchcad")
    ap.add_argument("--parquet", default=None,
                    help="Pre-cached parquet path; default: hf_hub_download from BenchCAD/cad_bench")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="For smoke; cap row count")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)

    # Step 1 — locate parquet
    if args.parquet:
        parquet = Path(args.parquet)
    else:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("BenchCAD_HF_TOKEN") or os.environ.get("HF_TOKEN")
        cache_dir = out_root.parent / "_cache_benchcad"
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet = Path(hf_hub_download(
            "BenchCAD/cad_bench",
            "data/test-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=token,
        ))
    print(f"parquet: {parquet} ({parquet.stat().st_size / 1024**2:.1f} MB)")

    # Step 2 — stream rows, write .py + .png, queue STL jobs
    import pyarrow.parquet as pq
    t = pq.read_table(parquet)
    cols = t.schema.names
    print(f"rows: {t.num_rows}; cols: {cols}")

    rows = t.to_pylist() if args.limit is None else t.slice(0, args.limit).to_pylist()

    # Per-row materialization (fast, sequential — just file I/O)
    stl_jobs: list[tuple[str, str]] = []
    annotations: dict[str, list[dict]] = {"train": [], "val": []}

    for i, row in enumerate(rows):
        stem = row["stem"]
        code = row["gt_code"]
        png_bytes = row["composite_png"]
        if isinstance(png_bytes, dict):  # arrow struct with "bytes" key
            png_bytes = png_bytes.get("bytes", b"")
        split = _split(stem)
        split_dir = out_root / split

        py_path = split_dir / f"{stem}.py"
        png_path = split_dir / f"{stem}_render.png"
        stl_path = split_dir / f"{stem}.stl"

        if args.force or not py_path.exists():
            py_path.write_text(code)
        if png_bytes and (args.force or not png_path.exists()):
            png_path.write_bytes(png_bytes)
        if args.force or not stl_path.exists():
            stl_jobs.append((str(py_path), str(stl_path)))

        annotations[split].append({
            "uid": stem,
            "py_path": str(py_path.relative_to(out_root)),
            "mesh_path": str(stl_path.relative_to(out_root)),
            "png_path": str(png_path.relative_to(out_root)),
            "description": _compose_description(row),
            "family": row.get("family", ""),
            "difficulty": row.get("difficulty", ""),
        })

        if (i + 1) % 2000 == 0:
            print(f"  wrote py+png {i+1}/{len(rows)}")

    print(f"py+png done. train={len(annotations['train'])}, val={len(annotations['val'])}, "
          f"stl jobs={len(stl_jobs)}")

    # Step 3 — STL generation (parallel)
    skipped_path = out_root / "_skipped.jsonl"
    skipped_path.unlink(missing_ok=True)
    if stl_jobs:
        print(f"generating STLs with {args.workers} workers ...")
        ok = 0
        fail = 0
        with mp.Pool(args.workers) as pool:
            for i, (stl_path, err) in enumerate(pool.imap_unordered(_exec_to_stl, stl_jobs)):
                if err is None:
                    ok += 1
                else:
                    fail += 1
                    with skipped_path.open("a") as fp:
                        fp.write(json.dumps({"stl_path": stl_path, "error": err}) + "\n")
                if (i + 1) % 500 == 0:
                    print(f"  stl {i+1}/{len(stl_jobs)}  ok={ok}  fail={fail}")
        print(f"STL gen done. ok={ok}, fail={fail} (→ {skipped_path})")

    # Step 4 — filter annotations to ones with a valid STL, write pkl
    for split in ("train", "val"):
        kept = [a for a in annotations[split]
                if (out_root / a["mesh_path"]).exists() and
                   (out_root / a["mesh_path"]).stat().st_size > 200]
        dropped = len(annotations[split]) - len(kept)
        pkl = out_root / f"{split}.pkl"
        with pkl.open("wb") as fp:
            pickle.dump(kept, fp)
        print(f"{split}.pkl: {len(kept)} kept, {dropped} dropped (no STL)")

    print(f"\nDone. {out_root}/train.pkl  {out_root}/val.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
