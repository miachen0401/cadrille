"""Dataset classes and mesh→input helpers for RL training."""

import os
import json
import pickle
import random

import numpy as np
from glob import glob
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------

def render_img(gt_mesh_path: str) -> dict:
    """Render 4-view image grid from a mesh path (image-mode examples)."""
    import trimesh
    import open3d
    from PIL import Image, ImageOps
    from dataset import mesh_to_image

    mesh = trimesh.load(gt_mesh_path)
    # Normalize to [0,1]^3 — matches reference code (transform_real_mesh + scale/translate)
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(2.0 / max(mesh.extents))   # → [-1, 1]
    mesh.apply_scale(0.5)                         # → [-0.5, 0.5]
    mesh.apply_translation([0.5, 0.5, 0.5])       # → [0, 1]
    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = open3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
    o3d_mesh.compute_vertex_normals()
    fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
    imgs = [ImageOps.expand(mesh_to_image(o3d_mesh, camera_distance=-0.9, front=f, img_size=128),
                            border=3, fill='black')
            for f in fronts]
    combined = Image.fromarray(np.vstack((
        np.hstack((np.array(imgs[0]), np.array(imgs[1]))),
        np.hstack((np.array(imgs[2]), np.array(imgs[3]))))))
    return {'video': [combined]}


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class MeshDataset:
    """Load GT meshes directly from a directory of .stl files.

    Primary dataset for RL fine-tuning on real handcrafted meshes
    (e.g. DeepCAD train, Fusion360 train).

    modality='pc'  — point cloud input (paper: unstable for RL)
    modality='img' — 4-view rendered image input (paper default for RL)

    Image mode is lazy: meshes are rendered on first __getitem__ access,
    so startup is instant regardless of dataset size.
    """

    def __init__(self, data_dir: str, n_points: int = 256,
                 noise_scale: float = 0.01, size: int = None,
                 modality: str = 'img'):
        self.modality = modality
        self.noise_scale = noise_scale
        self.n_points = n_points

        stl_files = sorted(glob(os.path.join(data_dir, '**', '*.stl'), recursive=True))
        if size is not None:
            rng = random.Random(42)
            rng.shuffle(stl_files)
            stl_files = stl_files[:size]

        if modality == 'img':
            # Lazy image mode — store paths only, render on demand.
            # No upfront cost; rendering takes ~0.2s per mesh at access time.
            self.examples = [
                {
                    'description': 'Generate cadquery code',
                    'file_name': os.path.splitext(os.path.basename(p))[0],
                    'gt_mesh_path': p,
                }
                for p in stl_files
            ]
            print(f'MeshDataset (img): {len(self.examples)} meshes from {data_dir}')
        else:
            # Point-cloud mode — load and process all meshes at init.
            import trimesh
            from dataset import mesh_to_point_cloud

            self.examples = []
            print(f'Loading {len(stl_files)} meshes from {data_dir} ...')
            for path in tqdm(stl_files, desc='mesh→pc'):
                try:
                    mesh = trimesh.load(path)
                    pc = mesh_to_point_cloud(mesh, n_points)
                    pc = (pc - 0.5) * 2
                    if noise_scale > 0:
                        pc = pc + np.random.randn(*pc.shape).astype(np.float32) * noise_scale
                    self.examples.append({
                        'point_cloud': pc,
                        'description': 'Generate cadquery code',
                        'file_name': os.path.splitext(os.path.basename(path))[0],
                        'gt_mesh_path': path,
                    })
                except Exception:
                    pass
            print(f'  → loaded {len(self.examples)} valid examples')

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ex = self.examples[index]
        if self.modality == 'img':
            item = dict(ex)
            item.update(render_img(ex['gt_mesh_path']))
            return item
        return ex


class RLDataset:
    """Loads hard-mined examples from rl/mine.py output pkl."""

    def __init__(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            self.examples = pickle.load(f)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ex = self.examples[index]
        item = {
            'description': 'Generate cadquery code',
            'file_name': ex['file_name'],
            'gt_mesh_path': ex['gt_mesh_path'],
        }
        if ex.get('is_pc', True):
            item['point_cloud'] = ex['point_cloud']
        else:
            item.update(render_img(ex['gt_mesh_path']))
        return item


class DPODataset:
    """Precomputed preference pairs for DPO training.

    JSONL: {"description", "point_cloud"|null, "file_name", "gt_mesh_path",
            "y_w", "y_l", "ref_logp_w", "ref_logp_l"}
    """

    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        item = {
            'description': rec['description'],
            'file_name': rec['file_name'],
            'gt_mesh_path': rec['gt_mesh_path'],
            'y_w': rec['y_w'],
            'y_l': rec['y_l'],
            'ref_logp_w': float(rec['ref_logp_w']),
            'ref_logp_l': float(rec['ref_logp_l']),
        }
        if rec.get('point_cloud') is not None:
            item['point_cloud'] = np.array(rec['point_cloud'], dtype=np.float32)
        return item
