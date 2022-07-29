import os
import time

import warnings
import wandb

import torch

# data
from utils import build_dataloader

# optimizer, losses
from apex.optimizers import FusedAdam

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

# misc.
from train import NeRFSystem as BaseNeRFSystem
from opt import get_opts

warnings.filterwarnings("ignore")
seed_everything(19870203)


class NeRFSystem(BaseNeRFSystem):

    def __init__(self, hparams, train_dataset):
        hparams.eval_lpips = True

        super().__init__(hparams, train_dataset)

        self.epoch_it = int(hparams.ft_num_epochs * 1000)

    def configure_optimizers(self):
        self.opt = FusedAdam(
            self.model.parameters(), self.hparams.ft_lr, eps=1e-15)
        return self.opt

    def _get_log_step(self):
        return self.epoch_it * self.hparams.timestep + self.global_step


def build_trainer(hparams, callbacks=None):
    return Trainer(
        max_steps=int(hparams.ft_num_epochs * 1000),
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
        # timestep 0 load pretrained weight
        if system is None:
            system = NeRFSystem(hparams, train_set)
            system.load_state_dict(
                torch.load(hparams.ckpt_path,
                           map_location='cpu')['state_dict'])
            trainer.validate(system, test_loader)
            continue
        else:
            state_dict = system.model.state_dict()
            system = NeRFSystem(hparams, train_set)
            system.model.load_state_dict(state_dict)
            system.model.reset_density_field()
            trainer = build_trainer(hparams, callbacks=callbacks)

        start_t = time.time()
        trainer.fit(system, train_loader)
        end_t = time.time()
        print(f'\tTraining timestep {timestep} took {end_t - start_t:.2f}s\n')
        trainer.validate(system, test_loader)
