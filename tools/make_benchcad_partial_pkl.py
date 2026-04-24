"""Write a partial benchcad train.pkl/val.pkl from the STLs that exist NOW.

fetch_benchcad.py only writes pkl after all 20143 STLs finish — this helper lets
us kick off SFT smoke using whatever subset is already materialized.

Re-running fetch_benchcad.py to completion will overwrite these pkls with the
full set, so this script is safe to run concurrently.
"""
from __future__ import annotations
import pickle
from pathlib import Path


def main() -> None:
    root = Path("data/benchcad")
    for split in ("train", "val"):
        split_dir = root / split
        rows = []
        for stl in sorted(split_dir.glob("*.stl")):
            if stl.stat().st_size < 200:
                continue
            stem = stl.stem
            py = split_dir / f"{stem}.py"
            png = split_dir / f"{stem}_render.png"
            if not (py.exists() and png.exists()):
                continue
            rows.append({
                "uid": stem,
                "py_path": f"{split}/{stem}.py",
                "mesh_path": f"{split}/{stem}.stl",
                "png_path": f"{split}/{stem}_render.png",
                "description": "Generate cadquery code",  # fetch_benchcad.py builds a richer one; smoke is fine with default
            })
        pkl = root / f"{split}.pkl"
        with pkl.open("wb") as fp:
            pickle.dump(rows, fp)
        print(f"{split}.pkl: {len(rows)} rows → {pkl}")


if __name__ == "__main__":
    main()
