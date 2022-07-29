import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from datasets import dataset_dict
from cvbase.optflow import flow2rgb


def build_dataloader(hparams):
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {
        'root_dir': hparams.root_dir,
        'downsample': hparams.downsample,
        'timestep': hparams.timestep,
        'use_depth': hparams.use_depth,
    }
    train_dataset = dataset(split=hparams.split, **kwargs)
    train_dataset.batch_size = hparams.batch_size
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        persistent_workers=True,
        batch_size=None,
        pin_memory=True,
    )
    test_dataset = dataset(split='test', **kwargs)
    test_loader = DataLoader(
        test_dataset,
        num_workers=8,
        batch_size=None,
        pin_memory=True,
    )
    return train_dataset, train_loader, test_loader


def extract_model_state_dict(ckpt_path,
                             model_name='model',
                             prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name,
                                           prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = []
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips') or \
                k in ['weights', 'model.density_grid', 'model.grid_coords']:
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k)
    return ckpt['state_dict']


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def flow2img(flow):
    """flow: [H, W, 2], torch.Tensor, in camera coordinate."""
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    # TODO:
    flow[np.abs(flow) < 1e-3] = 0.
    flow = flow2rgb(flow)
    flow = np.round(flow * 255.).astype(np.uint8)
    return flow


def show_flow(flow, h=256, w=256, show=True):
    """Plot the flow map.

    Inputs:
        flow: [H, W, 2]
    """
    if len(flow.shape) == 2:
        flow = flow.reshape(h, w, 2)
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = flow + 127.5
    flow = np.round(flow).astype(np.uint8)
    flow = np.concatenate([flow, np.zeros((h, w, 1), dtype=np.uint8)], axis=-1)
    plt.figure()
    plt.imshow(flow)
    if show:
        plt.show()


@torch.no_grad()
def make_img_grids(imgs, pad_value=1):
    """Rearrange images into grids.

    Inputs:
        imgs: [N, H, W, 3]
    """
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs).float()
    if imgs.max().item() > 1.:
        imgs = imgs / 255.
    if imgs.shape[-1] == 3:
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
    N = imgs.shape[0]
    nrow = int(np.floor(np.sqrt(N)))
    imgs = vutils.make_grid(
        imgs, nrow=nrow, normalize=False, pad_value=pad_value)
    # back to numpy savesable image
    imgs = imgs.permute(1, 2, 0).contiguous().cpu().numpy()
    imgs = np.round(imgs * 255.).astype(np.uint8)
    return imgs
