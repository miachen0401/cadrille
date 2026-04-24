#!/usr/bin/env python3
"""Prepare 10k CAD-Recode samples for HuggingFace upload.

Pipeline: .py → execute → .stl → render 4-view PNG → HF dataset

Usage:
    uv run python -m data_prep.prepare_hf_cadrecode --n 10000 --workers 2 --out data/cadrecode_hf_10k
    uv run python -m data_prep.prepare_hf_cadrecode --n 10000 --workers 2 --out data/cadrecode_hf_10k --upload Hula0401/cad-sft
"""

import argparse
import os
import random
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

from tqdm import tqdm

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))


def _exec_and_render(args):
    """Execute .py → STL → render PNG. Returns (py_path, code, png_bytes, error)."""
    py_path, out_dir = args
    stem = Path(py_path).stem
    batch = Path(py_path).parent.name

    try:
        code = Path(py_path).read_text()

        # Execute CadQuery code to get mesh
        import cadquery as cq
        import trimesh
        import numpy as np

        ns = {'cq': cq}
        exec(code, ns)
        r = ns.get('r')
        if r is None:
            return py_path, code, None, 'no result variable r'

        compound = r.val()
        verts, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)

        if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            return py_path, code, None, 'degenerate mesh'

        # Save STL to temp, render, clean up
        stl_path = os.path.join(out_dir, f'{batch}_{stem}.stl')
        mesh.export(stl_path)

        from rl.dataset import render_img
        result = render_img(stl_path)
        img = result['video'][0]

        # Save PNG to bytes
        import io
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()

        # Remove STL to save disk
        os.remove(stl_path)

        return py_path, code, png_bytes, None

    except Exception as e:
        return py_path, None, None, str(e)[:200]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/cad-recode-v1.5/train',
                        help='Directory with batch_XX/*.py files')
    parser.add_argument('--n', type=int, default=10000, help='Number of samples')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers (keep low to avoid IO saturation)')
    parser.add_argument('--out', default='data/cadrecode_hf_10k', help='Output directory')
    parser.add_argument('--upload', default=None, help='HF repo ID to upload to')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Collect all .py files
    all_py = sorted(glob(os.path.join(args.data_dir, '**', '*.py'), recursive=True))
    print(f'Found {len(all_py)} .py files in {args.data_dir}')

    # Sample
    rng = random.Random(args.seed)
    rng.shuffle(all_py)
    # Over-sample to account for failures
    candidates = all_py[:int(args.n * 1.3)]
    print(f'Sampling {len(candidates)} candidates (target {args.n})')

    os.makedirs(args.out, exist_ok=True)

    # Process
    results = []
    errors = 0
    tasks = [(p, args.out) for p in candidates]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_exec_and_render, t): t for t in tasks}
        pbar = tqdm(total=args.n, desc='Processing')
        for future in as_completed(futures):
            py_path, code, png_bytes, error = future.result()
            if error:
                errors += 1
            else:
                results.append({
                    'py_path': py_path,
                    'code': code,
                    'png_bytes': png_bytes,
                })
                pbar.update(1)
                if len(results) >= args.n:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
        pbar.close()

    print(f'Collected {len(results)} successful samples, {errors} errors')

    # Build HF dataset
    from datasets import Dataset, Features, Value, Image as HFImage
    from PIL import Image
    import io

    records = []
    for i, r in enumerate(tqdm(results[:args.n], desc='Building dataset')):
        img = Image.open(io.BytesIO(r['png_bytes']))
        batch_stem = Path(r['py_path']).parent.name + '/' + Path(r['py_path']).stem
        records.append({
            'stem': batch_stem,
            'code': r['code'],
            'render_img': img,
        })

    ds = Dataset.from_list(records)
    print(f'Dataset: {ds}')
    print(f'Columns: {ds.column_names}')

    # Save locally
    ds.save_to_disk(args.out)
    print(f'Saved to {args.out}')

    # Upload as a subset/config in the existing repo
    if args.upload:
        ds.push_to_hub(args.upload, config_name='cad-recode-10k', split='train',
                        private=False)
        print(f'Uploaded to {args.upload} (config=cad-recode-10k, split=train)')


if __name__ == '__main__':
    main()
