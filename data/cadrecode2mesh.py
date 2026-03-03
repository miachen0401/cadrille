import os
import pickle
import trimesh
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def py_file_to_mesh_file(py_path):
    try:
        with open(py_path, 'r') as f:
            py_string = f.read()
        exec(py_string, globals())
        compound = globals()['r'].val()
        mesh = compound_to_mesh(compound)
        mesh.export(py_path[:-3] + '.stl')
    except:
        pass


def run_split(path, py_paths, split, workers):
    pool = Pool(workers)
    list(tqdm(pool.imap(py_file_to_mesh_file, py_paths),
              total=len(py_paths), desc=split))
    pool.close()
    pool.join()

    annotations = []
    for py_path in py_paths:
        mesh_path = py_path[:-3] + '.stl'
        if os.path.exists(mesh_path):
            annotations.append(dict(
                py_path=py_path[len(path) + 1:],
                mesh_path=mesh_path[len(path) + 1:]))
    with open(os.path.join(path, f'{split}.pkl'), 'wb') as f:
        pickle.dump(annotations, f)
    print(f'{split}: {len(annotations)} / {len(py_paths)} converted successfully')


def run(path, val_batch, workers):
    all_train_py = glob(os.path.join(path, 'train', '*', '*.py'))

    if val_batch:
        # Use a specific batch directory as the val split (files stay in place).
        # The generated val.pkl paths are relative to `path`, so CadRecodeDataset
        # loads them correctly regardless of which subdirectory they live in.
        val_dir = os.path.join(path, 'train', val_batch)
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f'val_batch directory not found: {val_dir}')
        val_py = glob(os.path.join(val_dir, '*.py'))
        val_set = set(val_py)
        train_py = [p for p in all_train_py if p not in val_set]
        print(f'Using {val_batch} as val split  '
              f'({len(val_py)} files, {len(train_py)} train files)')
    else:
        # Original behaviour: expects a flat val/ directory
        val_py = glob(os.path.join(path, 'val', '*.py'))
        train_py = all_train_py
        print(f'train: {len(train_py)}  val: {len(val_py)}')

    run_split(path, train_py, 'train', workers)
    if val_py:
        run_split(path, val_py, 'val', workers)
    else:
        print('No val files found — skipping val.pkl generation')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./cad-recode-v1.5',
                        help='Path to cad-recode-v1.5 dataset root')
    parser.add_argument('--val-batch', type=str, default=None,
                        help='Batch directory to use as val split, e.g. batch_10. '
                             'If omitted, uses the val/ directory (original behaviour).')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of parallel worker processes')
    args = parser.parse_args()
    run(args.path, args.val_batch, args.workers)
