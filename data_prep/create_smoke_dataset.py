"""create_smoke_dataset.py — Build a small smoke-test training dataset.

Selects N simple STL+code pairs from cad-recode-v1.5/train/ where the GT code
achieves IoU≥0.95 on its own mesh (verified via rl/reward.py subprocess).
Copies the STLs to data/smoke_train/ and writes smoke_train.pkl for use with
rl/train.py --config configs/rl/smoke.yaml.

Usage
-----
    python3 tools/create_smoke_dataset.py [--n N] [--out-dir PATH] [--no-verify]

Options
-------
    --n N           Number of examples to include (default: 100)
    --out-dir PATH  Output directory for STL files (default: ./data/smoke_train)
    --no-verify     Skip IoU verification (use all candidates, faster)
    --src-root PATH Root of cad-recode-v1.5 dataset (default: ./data/cad-recode-v1.5)

Output
------
    data/smoke_train/{stem}.stl       — GT mesh files (copied from cad-recode-v1.5)
    data/smoke_train/smoke_train.pkl  — list of {gt_mesh_path, file_name, py_path}
"""

import argparse
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='Build smoke-test training dataset from cad-recode-v1.5.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--n', type=int, default=100,
                        help='Number of examples (default: %(default)s)')
    parser.add_argument('--out-dir', default='./data/smoke_train',
                        help='Output directory for STL files (default: %(default)s)')
    parser.add_argument('--src-root', default='./data/cad-recode-v1.5',
                        help='cad-recode-v1.5 root (default: %(default)s)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip IoU verification (faster but less rigorous)')
    args = parser.parse_args()

    pkl_path = os.path.join(args.src_root, 'train.pkl')
    if not os.path.exists(pkl_path):
        print(f'ERROR: {pkl_path} not found. Run setup.sh --data first.')
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        rows = pickle.load(f)

    # Sort by STL file size — smallest = simplest geometry → fastest IoU verification
    rows = sorted(
        rows,
        key=lambda r: os.path.getsize(
            os.path.join(args.src_root, r['mesh_path'])
        ),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_verify:
        from common.metrics import compute_reward

    examples = []
    n_tried = 0
    n_failed_verify = 0

    for r in rows:
        if len(examples) >= args.n:
            break
        n_tried += 1

        src_stl = os.path.join(args.src_root, r['mesh_path'])
        src_py  = os.path.join(args.src_root, r['py_path'])
        if not os.path.exists(src_stl) or not os.path.exists(src_py):
            continue

        # Stem: use a flat name (batch_XX_stem) to avoid sub-directory collisions
        rel = r['mesh_path']  # e.g. train/batch_00/308.stl
        batch = os.path.basename(os.path.dirname(rel))  # batch_00
        stem  = os.path.splitext(os.path.basename(rel))[0]  # 308
        flat_stem = f'{batch}_{stem}'  # batch_00_308

        dst_stl = os.path.join(args.out_dir, f'{flat_stem}.stl')

        if not args.no_verify:
            with open(src_py) as f:
                code = f.read()
            iou = compute_reward(code, src_stl, timeout=30.0)
            if iou < 0.95:
                n_failed_verify += 1
                continue

        if not os.path.exists(dst_stl):
            shutil.copy2(src_stl, dst_stl)

        examples.append({
            'gt_mesh_path': dst_stl,
            'file_name':    flat_stem,
            'py_path':      src_py,
        })
        if len(examples) % 10 == 0:
            print(f'  {len(examples)}/{args.n} collected (tried {n_tried}, failed_verify={n_failed_verify})')

    out_pkl = os.path.join(args.out_dir, 'smoke_train.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(examples, f)

    print(f'\nSmoke dataset: {len(examples)} examples saved to {args.out_dir}')
    print(f'  pkl: {out_pkl}')
    print(f'  tried: {n_tried}  failed_verify: {n_failed_verify}')
    if len(examples) < args.n:
        print(f'  WARNING: only {len(examples)}/{args.n} collected — try --n {len(examples)}')


if __name__ == '__main__':
    main()
