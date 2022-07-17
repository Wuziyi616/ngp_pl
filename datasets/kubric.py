import os
import glob
import json
import numpy as np
from PIL import Image

import torch
from einops import rearrange
from tqdm import tqdm

from .base import BaseDataset
from .ray_utils import get_ray_directions, get_rays


class KubricDataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        timestep,
        split='train',
        downsample=1.0,
        **kwargs,
    ):
        super().__init__(root_dir, split, downsample)
        self.timestep = timestep

        meta_fn = f"transforms_{self.split}_{self.timestep:05d}.json"
        with open(os.path.join(root_dir, meta_fn)) as f:
            self.meta = json.load(f)
        w = int(self.meta['w'] * downsample)
        h = int(self.meta['h'] * downsample)
        self.img_wh = (w, h)
        xyz_min, xyz_max = np.array(self.meta['bbox']).reshape(2, 3)
        self.shift = (xyz_max + xyz_min) / 2.
        self.scale = (xyz_max - xyz_min).max() / 2. * 1.05  # enlarge a little
        self.blender2opencv = np.array([
            [1., 0, 0, 0],
            [0, -1., 0, 0],
            [0, 0, -1., 0],
            [0, 0, 0, 1.],
        ])

        fx = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x']) * downsample
        self.K = np.float32([
            [fx, 0., w / 2.],
            [0., fx, h / 2.],
            [0., 0., 1.],
        ])
        self.directions = get_ray_directions(h, w, self.K)

        if split == 'train':
            rays_train = self.read_meta('train')
            self.rays = torch.cat(list(rays_train.values()))
        else:  # val, test, test_train
            if split == 'test_train':
                split = 'train'
            self.rays = self.read_meta(split)

    def read_meta(self, split):
        img_paths = [
            os.path.join(self.root_dir, f['file_path'])
            for f in self.meta['frames']
        ]
        camera_poses = [f['transform_matrix'] for f in self.meta['frames']]

        rays = {}  # {frame_idx: ray tensor}
        self.poses = []
        print(f'Loading {len(img_paths)} {split} images ...')
        for idx, (img, pose) in enumerate(tqdm(zip(img_paths, camera_poses))):
            # to bound the scene inside [-0.5, 0.5]
            c2w = np.array(pose)[:3] @ self.blender2opencv
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= (2. * self.scale)
            self.poses.append(c2w)

            rays_o, rays_d = get_rays(self.directions,
                                      torch.cuda.FloatTensor(c2w))

            img = Image.open(img)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img).cuda()  # (c, h, w)
            img = rearrange(img, 'c h w -> (h w) c')
            # blend A to RGB
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1. - img[:, -1:])
            img[img < 0.01] = 0.

            rays[idx] = torch.cat([rays_o, rays_d, img], 1).cpu()  # (h*w, 9)
        self.poses = np.float32(self.poses)

        return rays
