import os
import glob
import time

import imageio
import warnings
import numpy as np
import wandb

import torch
from einops import rearrange

# data
from utils import build_dataloader

# models
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (PeakSignalNoiseRatio,
                          StructuralSimilarityIndexMeasure)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

# misc.
from opt import get_opts
from utils import depth2img, make_img_grids

warnings.filterwarnings("ignore")
seed_everything(19870203)


class NeRFSystem(LightningModule):

    def __init__(self, hparams, train_dataset):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = train_dataset

        self.loss = NeRFLoss(lambda_opa=hparams.occ_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        self.model = NGP(scale=hparams.scale, black_bg=hparams.black_bg)

        self.S = 16  # the interval to update density grid

    def forward(self, batch_data, split):
        kwargs = {'test_time': split != 'train'}
        if self.hparams.dataset_name == 'colmap':
            kwargs['exp_step_factor'] = 1. / 256.

        return render(self.model, batch_data['rays'], **kwargs)

    def configure_optimizers(self):
        self.opt = FusedAdam(
            self.model.parameters(), self.hparams.lr, eps=1e-15)
        self.sch = CosineAnnealingLR(self.opt,
                                     int(self.hparams.num_epochs * 1000),
                                     self.hparams.lr / 10)

        return ([self.opt], [{
            'scheduler': self.sch,
            'interval': 'step',
        }])

    def on_train_start(self):
        self.start_t = time.time()
        # mask cells according to the camera poses
        K = torch.cuda.FloatTensor(self.train_dataset.K)
        poses = torch.cuda.FloatTensor(self.train_dataset.poses)
        self.model.mark_invisible_cells(K, poses, self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb):
        if self.global_step % self.S == 0:
            self.model.update_density_grid(
                0.01 * MAX_SAMPLES / 3**0.5,
                warmup=self.global_step < 256,
                erode=self.hparams.dataset_name == 'colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch, **{'step': self.global_step})
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])

            if self.global_rank == 0:
                s_per_ray = results['total_samples'] / len(batch['rays'])
                log_dict = {
                    'train/lr': self.opt.param_groups[0]['lr'],
                    'train/loss': loss.detach().item(),
                    'train/psnr': self.train_psnr.compute().item(),
                    'train/s_per_ray': s_per_ray,
                }
                log_dict.update({
                    f'train/{k}_loss': v.mean().item()
                    for k, v in loss_d.items()
                })
                wandb.log(log_dict, step=self._get_log_step())

        return loss

    def on_train_end(self):
        used_t = time.time() - self.start_t
        wandb.log({'train/used_time': used_t}, step=self._get_log_step())

    def _get_log_step(self):
        return self.global_step

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'ckpts/{self.hparams.exp_name}/val'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(
                torch.clip(rgb_pred * 2 - 1, -1, 1),
                torch.clip(rgb_gt * 2 - 1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        # save test image to disk
        if not self.hparams.no_save_test and self.global_rank == 0:
            idx = batch['idx']
            rgb_pred = rearrange(
                results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = np.round(rgb_pred * 255.).astype(np.uint8)
            # depth = depth2img(
            #     rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            fn = f'{self.hparams.timestep}-{idx:03d}-pred_img.png'
            imageio.imsave(os.path.join(self.val_dir, fn), rgb_pred)
            logs['rgb_pred_img'] = rgb_pred
            # imageio.imsave(
            #     os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)
            rgb_gt = rearrange(
                batch['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_gt = np.round(rgb_gt * 255.).astype(np.uint8)
            fn = f'{self.hparams.timestep}-{idx:03d}-gt_img.png'
            imageio.imsave(os.path.join(self.val_dir, fn), rgb_gt)
            logs['rgb_gt_img'] = rgb_gt

        return logs

    def validation_epoch_end(self, outputs):
        log_dict = {}
        for k in outputs[0].keys():
            # numerical results
            if not k.endswith('_img'):
                v = torch.stack([x[k] for x in outputs])
                v = all_gather_ddp_if_available(v).mean()
                log_dict[f'test/{k}'] = v.item()

        print('\n\n')
        print('\n'.join(f'{k}: {v:.4f}' for k, v in log_dict.items()))
        print('\n\n')

        if self.global_rank == 0:
            for k in outputs[0].keys():
                # visualization results
                if k.endswith('_img'):
                    v = np.stack([x[k] for x in outputs])
                    v = make_img_grids(v, pad_value=0 if 'flow' in k else 1)
                    log_dict[f'test/{k[:-4]}'] = wandb.Image(v)

        wandb.log(log_dict, step=self._get_log_step(), commit=True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    train_set, train_loader, test_loader = build_dataloader(hparams)
    system = NeRFSystem(hparams, train_set)

    dir_path = f'ckpts/{hparams.exp_name}'
    os.makedirs(dir_path, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=dir_path,
        every_n_train_steps=int(hparams.num_epochs * 1000),
        save_last=True,
        save_weights_only=True,
    )
    callbacks = [ckpt_cb]

    wandb.init(project='ngp_pl', name=hparams.exp_name, dir=dir_path)
    logger = False

    trainer = Trainer(
        max_steps=int(hparams.num_epochs * 1000),
        check_val_every_n_epoch=1000000,  # no validation in timing
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator='gpu',
        devices=hparams.num_gpus,
        strategy=DDPPlugin(
            find_unused_parameters=False) if hparams.num_gpus > 1 else None,
        num_sanity_val_steps=-1 if hparams.val_only else 0,
        precision=16,
    )

    start_t = time.time()
    trainer.fit(system, train_loader, ckpt_path=hparams.ckpt_path)
    end_t = time.time()
    print(f'Training took {end_t - start_t:.2f} seconds')
    trainer.validate(system, test_loader)

    # save video
    if not hparams.no_save_test and hparams.dataset_name == 'nsvf':
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(
            os.path.join(system.val_dir, 'rgb.mp4'),
            [imageio.imread(img) for img in imgs[::2]],
            fps=30,
            macro_block_size=1)
        imageio.mimsave(
            os.path.join(system.val_dir, 'depth.mp4'),
            [imageio.imread(img) for img in imgs[1::2]],
            fps=30,
            macro_block_size=1)
