import os
import glob
import time

import cv2
import imageio
import warnings
import numpy as np

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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

# misc.
from opt import get_opts
from utils import slim_ckpt

warnings.filterwarnings("ignore")
seed_everything(19870203)


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):

    def __init__(self, hparams, train_dataset):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = train_dataset

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        self.model = NGP(scale=hparams.scale, black_bg=hparams.black_bg)

        self.S = 16  # the interval to update density grid

    def forward(self, rays, split):
        kwargs = {'test_time': split != 'train'}
        if hparams.dataset_name == 'colmap':
            kwargs['exp_step_factor'] = 1. / 256.

        return render(self.model, rays, **kwargs)

    def configure_optimizers(self):
        opt = FusedAdam(self.model.parameters(), hparams.lr, eps=1e-15)
        sch = CosineAnnealingLR(opt, hparams.num_epochs * 1000,
                                hparams.lr / 30)

        return ([opt], [{
            'scheduler': sch,
            'interval': 'step',
        }])

    def on_train_start(self):
        K = torch.cuda.FloatTensor(self.train_dataset.K)
        poses = torch.cuda.FloatTensor(self.train_dataset.poses)
        self.model.mark_invisible_cells(K, poses, self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb):
        if self.global_step % self.S == 0:
            self.model.update_density_grid(
                0.01 * MAX_SAMPLES / 3**0.5,
                warmup=self.global_step < 256,
                erode=hparams.dataset_name == 'colmap')

        results = self(batch['rays'], split='train')
        loss_d = self.loss(results, batch, **{'step': self.global_step})
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('train/loss', loss)
        self.log('train/s_per_ray',
                 results['total_samples'] / len(batch['rays']))
        self.log('train/psnr', self.train_psnr, prog_bar=True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not hparams.no_save_test:
            self.val_dir = f'ckpts/{hparams.exp_name}/val'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch['rays'], split='test')

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
        if hparams.eval_lpips:
            self.val_lpips(
                torch.clip(rgb_pred * 2 - 1, -1, 1),
                torch.clip(rgb_gt * 2 - 1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not hparams.no_save_test:  # save test image to disk
            idx = batch['idx']
            rgb_pred = rearrange(
                results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = np.round(rgb_pred * 255.).astype(np.uint8)
            # depth = depth2img(
            #     rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(
                os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            # imageio.imsave(
            #     os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, prog_bar=True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

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
        filename='{epoch:d}',
        save_weights_only=True,
        every_n_epochs=hparams.num_epochs,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )
    lr_cb = LearningRateMonitor(logging_interval='step')
    callbacks = [ckpt_cb, lr_cb]

    logger = WandbLogger(
        project='ngp_pl',
        name=hparams.exp_name,
        save_dir=dir_path,
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
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

    if not hparams.val_only:  # save slimmed ckpt for the last epoch
        ckpt_ = slim_ckpt(f'{dir_path}/epoch={hparams.num_epochs-1}.ckpt')
        torch.save(ckpt_, f'{dir_path}/epoch={hparams.num_epochs-1}_slim.ckpt')
