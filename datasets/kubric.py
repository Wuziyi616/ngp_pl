import os
import json

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from einops import rearrange

from .base import BaseDataset
from .ray_utils import get_ray_directions, get_rays, get_img_grids, inverse_c2w


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
        fy = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y']) * downsample \
            if 'camera_angle_y' in self.meta else fx
        self.fx, self.fy = fx, fy
        self.K = np.float32([
            [fx, 0., w / 2.],
            [0., fy, h / 2.],
            [0., 0., 1.],
        ])
        directions = get_ray_directions(h, w, self.K)
        self.directions = F.normalize(directions, p=2, dim=-1)

        if split == 'train':
            rays_train = self.read_meta('train')
            self.rays = torch.cat(list(rays_train.values()))
            # rays: [num_imgs * (w*h), (rays_o, rays_d, rgb)]
            self.idx2imgidx = np.arange(self.rays.shape[0]) // (w * h)
        else:  # val, test
            self.rays = self.read_meta(split)
            # a dict, key: img_idx, value: [w*h, (rays_o, rays_d, rgb)]

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
            # TODO:
            img[img < 0.01] = 0.

            rays[idx] = torch.cat([rays_o, rays_d, img], 1).cpu()  # (h*w, 9)
        self.poses = np.float32(self.poses)

        return rays


class KubricFlowDataset(KubricDataset):

    def read_meta(self, split):
        img_paths = [
            os.path.join(self.root_dir, f['file_path'])
            for f in self.meta['frames']
        ]
        flow_paths = [
            path.replace('rgba_', 'backward_flow_').replace('.png', '.npy')
            for path in img_paths
        ]
        camera_poses = [f['transform_matrix'] for f in self.meta['frames']]
        img_grids = get_img_grids(self.img_wh[1], self.img_wh[0], self.K)
        img_grids[..., 0] = img_grids[..., 0] / self.img_wh[0]
        img_grids[..., 1] = img_grids[..., 1] / self.img_wh[1]
        self.img_grids = img_grids

        rays = {}  # {frame_idx: ray tensor}
        self.poses, self.inv_poses = [], []
        print(f'Loading {len(img_paths)} {split} images ...')
        for idx, (img, flow, pose) in enumerate(
                tqdm(zip(img_paths, flow_paths, camera_poses))):
            # to bound the scene inside [-0.5, 0.5]
            c2w = np.array(pose)[:3] @ self.blender2opencv  # (3, 4)
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= (2. * self.scale)
            self.poses.append(c2w)
            c2w = torch.cuda.FloatTensor(c2w)
            self.inv_poses.append(inverse_c2w(c2w).cpu().flatten())
            rays_o, rays_d = get_rays(self.directions, c2w)

            img = Image.open(img)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img).cuda()  # (c, h, w)
            img = rearrange(img, 'c h w -> (h w) c')
            # blend A to RGB
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1. - img[:, -1:])
            # TODO:
            img[img < 0.01] = 0.  # black background

            # load flow
            if self.timestep == 0:
                flow = np.zeros((self.img_wh[1], self.img_wh[0], 2))
            else:
                flow = np.load(flow)
                # TODO:
                flow[np.abs(flow) < 1e-4] = 0.
            # [dx, dy] to [x + dx, y + dy] in pinhole camera format
            flow = torch.from_numpy(flow).float().cuda()  # (h, w, 2)
            # TODO: normalize the flow
            flow[..., 0] = flow[..., 0] / self.img_wh[0]
            flow[..., 1] = flow[..., 1] / self.img_wh[1]
            flow = flow + self.img_grids
            flow = rearrange(flow, 'h w c -> (h w) c')

            # (h*w, 9)
            rays[idx] = torch.cat([rays_o, rays_d, img, flow], 1).cpu()
        self.poses = np.float32(self.poses)
        self.inv_poses = torch.stack(self.inv_poses)  # (num_imgs, 3*4)

        return rays

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # load backward flow
        # I2[x, y] = (u, v) --> I2[x + v, y + u] == I1[x, y]
        # this is suitable for pinhole camera whose x is W and y is H
        if self.split.startswith('train'):
            idxs = sample['idxs']
            sample['flow'] = self.rays[idxs, 9:11]
            # w2c matrix
            imgidxs = self.idx2imgidx[idxs]
            sample['w2c'] = self.inv_poses[imgidxs]  # [N, 3*4]
        else:
            sample['flow'] = self.rays[idx][:, 9:11]
            sample['w2c'] = self.inv_poses[idx]  # [3*4]
        return sample
