import os
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
from models.deform_networks import DeformNGP
from models.rendering import render

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

warnings.filterwarnings("ignore")
seed_everything(19870203)


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):

    def __init__(self, hparams, train_dataset, ngp_model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = train_dataset

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
        for p in self.val_lpips.net.parameters():
            p.requires_grad = False

        self.model = DeformNGP(ngp_model)

    def forward(self, rays, split):
        kwargs = {'test_time': split != 'train'}
        if hparams.dataset_name == 'colmap':
            kwargs['exp_step_factor'] = 1. / 256.

        return render(self.model, rays, **kwargs)

    def configure_optimizers(self):
        opt = FusedAdam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            hparams.lr,
            eps=1e-15)
        sch = CosineAnnealingLR(opt, hparams.num_epochs * 1000,
                                hparams.lr / 30)

        return ([opt], [{
            'scheduler': sch,
            'interval': 'step',
        }])

    def training_step(self, batch, batch_nb):
        rays, rgb = batch['rays'], batch['rgb']
        results = self(rays, split='train')
        loss_d = self.loss(results, rgb)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], rgb)
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, prog_bar=True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not hparams.no_save_test:
            self.val_dir = f'ckpts/{hparams.exp_name}/val'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rays, rgb_gt = batch['rays'], batch['rgb']
        results = self(rays, split='test')

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
        self.val_lpips(
            torch.clip(rgb_pred * 2. - 1., -1., 1.),
            torch.clip(rgb_gt * 2. - 1., -1., 1.))
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
        ssims = torch.stack([x['ssim'] for x in outputs])
        lpipss = torch.stack([x['lpips'] for x in outputs])

        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        mean_lpips = all_gather_ddp_if_available(lpipss).mean()

        self.log('test/psnr', mean_psnr, prog_bar=True)
        self.log('test/ssim', mean_ssim)
        self.log('test/lpips_vgg', mean_lpips)

        print(f'Test PSNR: {mean_psnr.item():.2f}')
        print(f'Test SSIM: {mean_ssim.item():.2f}')
        print(f'Test LPIPS: {mean_lpips.item():.2f}')


def build_trainer(hparams):
    return Trainer(
        max_steps=int(hparams.ft_num_epochs * 1000),
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


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

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

    trainer = build_trainer(hparams)

    # pretrained NGP as template model
    ngp_model = NGP(scale=hparams.scale, black_bg=hparams.black_bg).eval()
    ckpt = torch.load(hparams.ckpt_path, map_location='cpu')['state_dict']
    ckpt = {k[6:]: v for k, v in ckpt.items() if k.startswith('model.')}
    ngp_model.load_state_dict(ckpt)

    system = None
    for timestep in range(hparams.timestep_start + 1, hparams.timestep_end):
        print('#######################################################')
        print(f'Timestep: {timestep}')
        hparams.timestep = timestep
        train_set, train_loader, test_loader = build_dataloader(hparams)
        if system is None:
            system = NeRFSystem(hparams, train_set, ngp_model)
        else:
            state_dict = system.model.state_dict()
            system = NeRFSystem(hparams, train_set, ngp_model)
            system.model.load_state_dict(state_dict)
            trainer = build_trainer(hparams)

        start_t = time.time()
        trainer.fit(system, train_loader)
        end_t = time.time()
        print(f'\tTraining timestep {timestep} took {end_t - start_t:.2f}s\n')
        trainer.validate(system, test_loader)
