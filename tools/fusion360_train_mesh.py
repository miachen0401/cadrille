#!/usr/bin/env python3
"""
Download, convert, and upload Fusion360 Gallery reconstruction train split.

Pipeline:
  1. Download r1.0.1.zip (2.0 GB) from Fusion360 Gallery S3
  2. Extract, discover folder structure
  3. Parse train_test.json → train design IDs
  4. Convert OBJ → normalized STL (center + scale to [0,1]^3)
  5. Zip → data/fusion360_train_mesh.zip
  6. Upload to Hula0401/fusion360_train_mesh on HuggingFace

Usage:
    python tools/fusion360_train_mesh.py [--skip-download] [--skip-convert] [--skip-upload]
"""

import os
import sys
import json
import zipfile
import argparse
import tempfile
import shutil
import trimesh
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path


S3_URL  = "https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip"
ZIP_PATH  = "data/_zips/r1.0.1.zip"
EXTRACT_DIR = "data/_zips/r1.0.1_extracted"
OUT_DIR   = "data/fusion360_train_mesh"
OUT_ZIP   = "data/fusion360_train_mesh.zip"
HF_REPO   = "Hula0401/fusion360_train_mesh"
HF_FILE   = "fusion360_train_mesh.zip"


def download(url: str, dest: str):
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  Already exists ({size // 1024**2} MB): {dest}")
        return
    print(f"  Downloading {url} ...")
    def progress(count, block, total):
        pct = count * block / total * 100
        mb = count * block / 1024**2
        print(f"\r  {mb:.0f} / {total/1024**2:.0f} MB  ({pct:.0f}%)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()
    print(f"  Saved → {dest}  ({os.path.getsize(dest)//1024**2} MB)")


def extract(zip_path: str, out_dir: str):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"  Already extracted → {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting {zip_path} → {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"  Done.")


def load_train_ids(extract_dir: str):
    """Parse train_test.json and return set of train design IDs."""
    # Search recursively for train_test.json
    matches = list(Path(extract_dir).rglob("train_test.json"))
    if not matches:
        raise FileNotFoundError(f"train_test.json not found under {extract_dir}")
    path = matches[0]
    print(f"  Found split file: {path}")
    with open(path) as f:
        data = json.load(f)

    # Format varies — try common patterns
    if isinstance(data, dict) and "train" in data:
        ids = set(data["train"])
    elif isinstance(data, list):
        # list of {"design": "...", "split": "train"/"test"}
        ids = {d["design"] for d in data if d.get("split") == "train"}
    else:
        raise ValueError(f"Unrecognized train_test.json format: {list(data)[:3]}")

    print(f"  Train IDs: {len(ids)}")
    return ids


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center and scale to [0,1]^3 — matches deepcad_train_mesh convention."""
    bounds = mesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation(-center)
    ext = max(mesh.extents)
    if ext > 1e-7:
        mesh.apply_scale(1.0 / ext)        # → [-0.5, 0.5]
    mesh.apply_translation([0.5, 0.5, 0.5])  # → [0, 1]
    return mesh


def convert_obj_to_stl(obj_path: str, out_path: str) -> bool:
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            # Scene with multiple geometries — merge
            if hasattr(mesh, 'dump'):
                meshes = mesh.dump()
                if not meshes:
                    return False
                mesh = trimesh.util.concatenate(meshes)
            else:
                return False
        if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            return False
        mesh = normalize_mesh(mesh)
        mesh.export(out_path)
        return True
    except Exception as e:
        return False


def convert_split(extract_dir: str, train_ids: set, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Find all OBJ files under extract_dir
    all_objs = list(Path(extract_dir).rglob("*.obj"))
    print(f"  Total OBJ files found: {len(all_objs)}")

    # Filter to train split by checking if any parent component of the path
    # starts with a known train ID prefix.
    # Fusion360 naming: {project}_{file}_{component}.obj
    # train_test.json "design" entries look like "{project}_{file}"
    def get_design_id(obj_path: Path) -> str:
        stem = obj_path.stem  # e.g. "100155_57ec5fc6_0000"
        parts = stem.rsplit("_", 1)  # split off trailing component index
        return parts[0] if len(parts) == 2 else stem

    train_objs = [p for p in all_objs if get_design_id(p) in train_ids]
    print(f"  OBJ files in train split: {len(train_objs)}")

    if len(train_objs) == 0:
        # Fallback: print a sample of found design IDs vs train_ids to debug
        sample_found = {get_design_id(p) for p in all_objs[:20]}
        sample_train = list(train_ids)[:5]
        print(f"  WARNING: no matches. Sample found IDs: {sample_found}")
        print(f"  Sample train IDs: {sample_train}")
        return 0

    ok = 0
    fail = 0
    for obj_path in tqdm(train_objs, desc="OBJ→STL"):
        stem = obj_path.stem  # e.g. "100155_57ec5fc6_0000"
        out_path = os.path.join(out_dir, stem + ".stl")
        if os.path.exists(out_path):
            ok += 1
            continue
        if convert_obj_to_stl(str(obj_path), out_path):
            ok += 1
        else:
            fail += 1

    print(f"  Converted: {ok} ok, {fail} failed  ({100*ok/(ok+fail+1e-9):.1f}% success)")
    return ok


def make_zip(in_dir: str, zip_path: str):
    stls = glob(os.path.join(in_dir, "*.stl"))
    if os.path.exists(zip_path):
        print(f"  Zip already exists ({os.path.getsize(zip_path)//1024**2} MB): {zip_path}")
        return
    print(f"  Zipping {len(stls)} STL files → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, stl in enumerate(stls):
            zf.write(stl, os.path.basename(stl))
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(stls)} done", flush=True)
    print(f"  Done. Size: {os.path.getsize(zip_path)/1024**2:.2f} MB")


def upload_to_hf(zip_path: str, repo_id: str, filename: str):
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"  Creating repo {repo_id} (if not exists) ...")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"  Uploading {zip_path} → {repo_id}/{filename} ...")
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Upload complete: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-convert",  action="store_true")
    parser.add_argument("--skip-zip",      action="store_true")
    parser.add_argument("--skip-upload",   action="store_true")
    args = parser.parse_args()

    print("\n=== Step 1: Download ===")
    if not args.skip_download:
        download(S3_URL, ZIP_PATH)
    else:
        print("  Skipped.")

    print("\n=== Step 2: Extract ===")
    if not args.skip_download:
        extract(ZIP_PATH, EXTRACT_DIR)
    else:
        print("  Skipped.")

    print("\n=== Step 3: Load train/test split ===")
    train_ids = load_train_ids(EXTRACT_DIR)

    print("\n=== Step 4: Convert OBJ → STL ===")
    if not args.skip_convert:
        n = convert_split(EXTRACT_DIR, train_ids, OUT_DIR)
        print(f"  Total STLs in {OUT_DIR}: {n}")
    else:
        existing = len(glob(os.path.join(OUT_DIR, "*.stl")))
        print(f"  Skipped. Existing STLs: {existing}")

    print("\n=== Step 5: Zip ===")
    if not args.skip_zip:
        make_zip(OUT_DIR, OUT_ZIP)
    else:
        print("  Skipped.")

    print("\n=== Step 6: Upload to HuggingFace ===")
    if not args.skip_upload:
        upload_to_hf(OUT_ZIP, HF_REPO, HF_FILE)
    else:
        print("  Skipped.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
