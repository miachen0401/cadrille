import json
import os
import pickle
import open3d
import trimesh
import skimage
import numpy as np
from PIL import Image, ImageOps

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

import torch
from torch.utils.data import Dataset


def mesh_to_point_cloud(mesh, n_points, n_pre_points=8192):
    from pytorch3d.ops import sample_farthest_points
    vertices, faces = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = vertices[ids]
    return np.asarray(vertices)


def mesh_to_image(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=128):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)


def _filter_by_code_len(root_dir, split, annotations, max_code_len, py_path_fn):
    """Drop annotations whose .py file exceeds max_code_len chars.

    Caches the filtered index list to {root_dir}/.filter_cache/{split}_len{max}.json
    so subsequent runs skip the full scan (~1 min for 980k files).
    """
    cache_dir = os.path.join(root_dir, '.filter_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{split}_len{max_code_len}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            keep_idx = json.load(f)
        return [annotations[i] for i in keep_idx]

    keep_idx = []
    for i, item in enumerate(annotations):
        py = py_path_fn(item)
        try:
            with open(py) as f:
                n = len(f.read())
        except Exception:
            continue
        if n <= max_code_len:
            keep_idx.append(i)
    with open(cache_file, 'w') as f:
        json.dump(keep_idx, f)
    print(f'[filter] {split}: {len(keep_idx)}/{len(annotations)} kept '
          f'(len<={max_code_len}) → {cache_file}')
    return [annotations[i] for i in keep_idx]


def _compute_lengths(annotations, root_dir, py_path_fn, desc_getter=lambda a: ''):
    """Cheap length estimate per sample: code file size + description char count.

    Used for length-grouped batching (group_by_length=True). os.path.getsize is
    O(1) per file on a warm fs cache.
    """
    lengths = []
    for item in annotations:
        try:
            py = py_path_fn(item)
            code_len = os.path.getsize(py)
        except Exception:
            code_len = 500
        lengths.append(code_len + len(desc_getter(item) or ''))
    return lengths


class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc, img_size,
                normalize_std_img, noise_scale_img, num_imgs, mode, n_samples=None, ext='stl',
                max_code_len=None):
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
            if max_code_len is not None:
                self.annotations = _filter_by_code_len(
                    root_dir, split, self.annotations, max_code_len,
                    lambda item: os.path.join(root_dir, item['py_path']),
                )
            self.lengths = _compute_lengths(
                self.annotations, root_dir,
                lambda item: os.path.join(root_dir, item['py_path']))
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
    def __init__(self, root_dir, split, code_dir='cadquery', n_samples=None, max_code_len=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_samples = n_samples
        self.code_dir = code_dir
        pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)
        if max_code_len is not None:
            self.annotations = _filter_by_code_len(
                root_dir, f'text2cad_{split}', self.annotations, max_code_len,
                lambda item: os.path.join(root_dir, code_dir, f'{item["uid"]}.py'),
            )
        self.lengths = _compute_lengths(
            self.annotations, root_dir,
            lambda item: os.path.join(root_dir, code_dir, f'{item["uid"]}.py'),
            desc_getter=lambda a: a.get('description', ''))

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


class BenchCadDataset(Dataset):
    """BenchCAD/cad_bench → file-based loader. Mirrors CadRecodeDataset shape.

    Built by tools/fetch_benchcad.py, which materializes the HF parquet into:
        {root_dir}/{split}/{uid}.py           gt_code
        {root_dir}/{split}/{uid}.stl          exec(gt_code) → tessellate
        {root_dir}/{split}/{uid}_render.png   composite_png (4-view, 268×268 RGB)
        {root_dir}/{split}.pkl                list[{uid, py_path, mesh_path, png_path, description}]

    img mode loads the pre-rendered PNG (resized to img_size), skipping Open3D
    render. pc mode samples from the STL via pytorch3d.
    """

    def __init__(self, root_dir, split, n_points=256, normalize_std_pc=100,
                 noise_scale_pc=None, img_size=128, normalize_std_img=200,
                 noise_scale_img=None, num_imgs=4, mode='pc_img',
                 n_samples=None, max_code_len=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.img_size = img_size
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        self.n_samples = n_samples

        pkl_path = os.path.join(root_dir, f'{split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)
        if max_code_len is not None:
            self.annotations = _filter_by_code_len(
                root_dir, f'benchcad_{split}', self.annotations, max_code_len,
                lambda item: os.path.join(root_dir, item['py_path']),
            )
        self.lengths = _compute_lengths(
            self.annotations, root_dir,
            lambda item: os.path.join(root_dir, item['py_path']),
            desc_getter=lambda a: a.get('description', ''))

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        if self.mode == 'pc':
            input_item = self._get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self._get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self._get_point_cloud(item)
            else:
                input_item = self._get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['file_name'] = item['uid']

        py_path = os.path.join(self.root_dir, item['py_path'])
        with open(py_path, 'r') as f:
            input_item['answer'] = f.read()
        return input_item

    def _get_img(self, item):
        png_path = os.path.join(self.root_dir, item['png_path'])
        img = Image.open(png_path).convert('RGB')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        # num_imgs is a no-op here because composite_png already bakes 4 views
        # into a single image; mirror CadRecodeDataset's return shape.
        return {
            'video': [img],
            'description': item.get('description', 'Generate cadquery code'),
        }

    def _get_point_cloud(self, item):
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices = mesh.vertices + np.random.normal(
                loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        pc = mesh_to_point_cloud(mesh, self.n_points)
        pc = pc / self.normalize_std_pc
        return {
            'point_cloud': pc,
            'description': item.get('description', 'Generate cadquery code'),
        }


class CadRecode20kDataset(Dataset):
    """Hula0401/cad-sft → cad-recode-20k → file-based loader. Img-mode only.

    Materialised by data_prep/fetch_cad_sft.py:
        {root}/{split}/{uid}.py              code
        {root}/{split}/{uid}_render.png      pre-rendered 4-view composite
        {root}/{split}.pkl                   [{uid, py_path, png_path}]

    No STL, no mesh — pc mode is not supported. The `mode` arg is accepted for
    API uniformity with the other loaders but only 'img' works.
    """

    def __init__(self, root_dir, split, img_size=128, n_samples=None,
                 max_code_len=None, mode='img', **_unused):
        super().__init__()
        if mode != 'img':
            raise ValueError(
                f'CadRecode20kDataset supports img only (got mode={mode!r}); '
                'no STL is materialised for this corpus.')
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.n_samples = n_samples

        pkl_path = os.path.join(root_dir, f'{split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)
        if max_code_len is not None:
            self.annotations = _filter_by_code_len(
                root_dir, f'recode20k_{split}', self.annotations, max_code_len,
                lambda item: os.path.join(root_dir, item['py_path']),
            )
        self.lengths = _compute_lengths(
            self.annotations, root_dir,
            lambda item: os.path.join(root_dir, item['py_path']))

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        img = Image.open(os.path.join(self.root_dir, item['png_path'])).convert('RGB')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        input_item = {
            'video': [img],
            'description': 'Generate cadquery code',
            'file_name': item['uid'],
        }
        if self.split in ('train', 'val'):
            with open(os.path.join(self.root_dir, item['py_path'])) as f:
                input_item['answer'] = f.read()
        return input_item
