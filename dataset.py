import os
import pickle
import open3d
import trimesh

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
import skimage
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points


def mesh_to_point_cloud(mesh, n_points, n_pre_points=8192):
    vertices, faces = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = vertices[ids]
    return np.asarray(vertices)


_offscreen_renderer: "open3d.visualization.rendering.OffscreenRenderer | None" = None

def _get_offscreen_renderer(width: int = 500, height: int = 500):
    """Return a per-process singleton OffscreenRenderer (creates it once, silently)."""
    global _offscreen_renderer
    if _offscreen_renderer is None:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        try:
            _offscreen_renderer = open3d.visualization.rendering.OffscreenRenderer(width, height)
            _offscreen_renderer.render_to_image()  # force Filament to fully initialize while fds are suppressed
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull_fd)
    return _offscreen_renderer


def mesh_to_image(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=128):
    renderer = _get_offscreen_renderer(width, height)
    renderer.scene.clear_geometry()

    mat = open3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    lookat = np.array([0.5, 0.5, 0.5])
    front_n = np.array(front, dtype=float)
    front_n /= np.linalg.norm(front_n)
    eye = lookat + front_n * camera_distance
    up = np.array([0.0, 1.0, 0.0])

    renderer.setup_camera(60.0, lookat, eye, up)
    image = np.asarray(renderer.render_to_image())

    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)


class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc, img_size,
                normalize_std_img, noise_scale_img, num_imgs, mode, n_samples=None, ext='stl'):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.n_samples = n_samples
        self.n_points = n_points  
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        if self.split in ['train', 'val']:
            pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
            with open(pkl_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            paths = os.listdir(os.path.join(self.root_dir, self.split))
            self.annotations = [
                {'mesh_path': os.path.join(self.split, f)}
                for f in paths if f.endswith('.stl')
            ]

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        if self.mode == 'pc':
            input_item = self.get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self.get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(item)
            else:
                input_item = self.get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['file_name'] = os.path.basename(item['mesh_path'])[:-4]

        if self.split in ['train', 'val']:
            py_path = item['py_path']
            py_path = os.path.join(self.root_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer

        return input_item

    def get_img(self, item):
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        if self.split in ['train', 'val']:
            mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
            mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(vertices)
        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            image = mesh_to_image(
                mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
            images.append(image)
                
        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        if self.num_imgs == 1:
            images = [images[0]]
        elif self.num_imgs == 2:
            images = [Image.fromarray(np.hstack((
                np.array(images[0]), np.array(images[1])
            )))]
        elif self.num_imgs == 4:
            images = [Image.fromarray(np.vstack((
                np.hstack((np.array(images[0]), np.array(images[1]))),
                np.hstack((np.array(images[2]), np.array(images[3])))
            )))]
        else:
            raise ValueError(f'Invalid number of images: {self.num_imgs}')

        input_item = {
            'video': images,
            'description': 'Generate cadquery code'
        }
        return input_item

    def get_point_cloud(self, item):
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        
        if self.split in ['train', 'val']:
            point_cloud = point_cloud / self.normalize_std_pc
        else:
            point_cloud = (point_cloud - 0.5) * 2
        
        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh


class Text2CADDataset(Dataset):
    def __init__(self, root_dir, split, code_dir='cadquery', n_samples=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_samples = n_samples
        self.code_dir = code_dir
        pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        input_item = {
            'description': item['description'],
            'file_name': item['uid']
        }

        if self.split in ['train', 'val']:
            py_path = f'{item["uid"]}.py'
            py_path = os.path.join(self.root_dir, self.code_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer
        return input_item
