"""Mesh → model-input helpers shared by train/, eval/, and tools/.

Originally lived at rl/dataset.py:17-135 (`render_img` + `MeshDataset`). Moved
here so eval/ can import without depending on rl/.

rl/dataset.py re-exports both names for backwards compatibility during the
refactor; drop the shim once no caller references `rl.dataset.render_img`
or `rl.dataset.MeshDataset`.
"""
from __future__ import annotations

import os
import random
from glob import glob

import numpy as np
from tqdm import tqdm


def render_img(gt_mesh_path: str) -> dict:
    """Render 4-view image grid from a mesh path (image-mode examples).

    Loads from pre-rendered PNG cache if available ({stem}_render.png),
    otherwise renders on-the-fly via open3d Visualizer (~1 s per mesh).
    Run tools/prerender_dataset.py once to populate the cache.
    """
    png_path = gt_mesh_path[:-4] + '_render.png'
    if os.path.exists(png_path):
        from PIL import Image
        return {'video': [Image.open(png_path).convert('RGB')]}

    import trimesh
    import open3d
    from PIL import Image, ImageOps
    from common.datasets import mesh_to_image

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
            from common.datasets import mesh_to_point_cloud

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
            item['modality'] = 'img'
            item.update(render_img(ex['gt_mesh_path']))
            return item
        item = dict(ex)
        item['modality'] = 'pc'
        return item
