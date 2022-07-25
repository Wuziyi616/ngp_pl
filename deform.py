import os
import time

import warnings
import wandb

import torch

# data
from utils import build_dataloader

# models
from models.rendering import MAX_SAMPLES
from models.deform_networks import DeformNGP

# optimizer, losses
from apex.optimizers import FusedAdam
from losses import DeformNeRFLoss

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

# misc.
from finetune import NeRFSystem as BaseNeRFSystem
from opt import get_opts

warnings.filterwarnings("ignore")
seed_everything(19870203)


class NeRFSystem(BaseNeRFSystem):

    def __init__(self, hparams, train_dataset):
        super().__init__(hparams, train_dataset)

        self.epoch_it = int(hparams.def_num_epochs * 1000)
        self.loss = DeformNeRFLoss()
        self.model = DeformNGP(
            ckpt_path=hparams.ckpt_path,
            scale=hparams.scale,
            black_bg=hparams.black_bg,
            ft_rgb=hparams.ft_rgb,
        )

    def configure_optimizers(self):
        if not self.hparams.ft_rgb:
            params_list = filter(lambda p: p.requires_grad,
                                 self.model.parameters())
        else:
            deform_params = list(self.model.delta_xyz.parameters())
            rgb_params = list(self.model.rgb_net.parameters())
            params_list = [
                {
                    'params': deform_params,
                },
                {
                    'params': rgb_params,
                    'lr': self.hparams.ft_lr,
                },
            ]
        self.opt = FusedAdam(params_list, lr=self.hparams.def_lr, eps=1e-15)
        return self.opt

    def on_train_start(self):
        # TODO: we should update new density grid for new inputs
        K = torch.cuda.FloatTensor(self.train_dataset.K)
        poses = torch.cuda.FloatTensor(self.train_dataset.poses)
        self.model.mark_invisible_cells(K, poses, self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb):
        if self.global_step % self.S == 0:
            self.model.update_density_grid(
                0.01 * MAX_SAMPLES / 3**0.5,
                warmup=self.global_step < 256,
                erode=hparams.dataset_name == 'colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])

            step = self.epoch_it * self.hparams.timestep + self.global_step
            log_dict = {
                'train/lr': self.opt.param_groups[0]['lr'],
                'train/loss': loss.detach().item(),
                'train/psnr': self.train_psnr.compute().item(),
                'train/s_per_ray':
                results['total_samples'] / len(batch['rays']),
            }
            log_dict.update({
                f'train/{k}_loss': v.mean().item()
                for k, v in loss_d.items()
            })
            wandb.log(log_dict, step=step)

        return loss


def build_trainer(hparams, callbacks=None):
    return Trainer(
        max_steps=int(hparams.def_num_epochs * 1000),
        check_val_every_n_epoch=1000000,  # no validation in timing
        callbacks=callbacks,
        logger=False,
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
    callbacks = None

    wandb.init(project='ngp_pl', name=hparams.exp_name, dir=dir_path)

    trainer = build_trainer(hparams, callbacks=callbacks)

    system = None
    for timestep in range(hparams.timestep_start, hparams.timestep_end):
        print('#######################################################')
        print(f'Timestep: {timestep}')
        hparams.timestep = timestep
        train_set, train_loader, test_loader = build_dataloader(hparams)
        if system is None:
            system = NeRFSystem(hparams, train_set)
            trainer.validate(system, test_loader)
            continue
        else:
            state_dict = system.model.state_dict()
            system = NeRFSystem(hparams, train_set)
            system.model.load_state_dict(state_dict)
            trainer = build_trainer(hparams, callbacks=callbacks)

        start_t = time.time()
        trainer.fit(system, train_loader)
        end_t = time.time()
        print(f'\tTraining timestep {timestep} took {end_t - start_t:.2f}s\n')
        trainer.validate(system, test_loader)
