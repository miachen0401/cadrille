#!/usr/bin/env bash
# scripts/pack_datasets.sh — zip mesh datasets and upload to HuggingFace as single files
#
# Why: huggingface-cli download resolves metadata for every file before downloading.
# deepcad_test_mesh has 8048 files — this blows past HF's 5000 req/5min rate limit
# for free accounts and returns a misleading LocalEntryNotFoundError.
# Uploading as a single zip means download only needs 1 resolver request.
#
# Run once from the repo root after downloading the raw datasets:
#   bash scripts/pack_datasets.sh
#
# Requires: huggingface-cli login (write access to the dataset repos)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

pack_and_upload() {
    local name="$1"          # e.g. deepcad_test_mesh
    local src_dir="$2"       # e.g. data/deepcad_test_mesh
    local repo_id="$3"       # e.g. Hula0401/deepCAD_test
    local zip_file="${name}.zip"

    if [[ ! -d "$src_dir" ]]; then
        echo "SKIP $name — $src_dir not found"
        return
    fi

    echo "=== $name ==="
    echo "  Files : $(ls "$src_dir" | wc -l)"
    echo "  Size  : $(du -sh "$src_dir" | cut -f1)"

    echo "  Zipping → $zip_file ..."
    python3 - "$src_dir" "$zip_file" <<'PYEOF'
import sys, zipfile, os
from pathlib import Path
src, dst = Path(sys.argv[1]), sys.argv[2]
stls = sorted(src.glob("*.stl"))
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
    for i, f in enumerate(stls, 1):
        zf.write(f, f.name)
        if i % 500 == 0:
            print(f"    {i}/{len(stls)} ...")
print(f"    {len(stls)} files zipped")
PYEOF
    echo "  Zip size: $(du -sh "$zip_file" | cut -f1)"

    echo "  Uploading to hf://datasets/$repo_id/$zip_file ..."
    huggingface-cli upload "$repo_id" "$zip_file" "$zip_file" --repo-type dataset

    rm "$zip_file"
    echo "  Done."
}

pack_and_upload \
    "deepcad_test_mesh" \
    "data/deepcad_test_mesh" \
    "Hula0401/deepCAD_test"

pack_and_upload \
    "fusion360_test_mesh" \
    "data/fusion360_test_mesh" \
    "Hula0401/fusion360_test_mesh"

echo ""
echo "Both zips uploaded. setup.sh --data already pulls them via hf_hub_download."
