import os
import time

import warnings
import wandb

# data
from utils import build_dataloader

# models
from models.voxel_networks import SVOXNGP

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
        super().__init__(hparams, train_dataset)

        self.model = SVOXNGP(
            scale=hparams.scale, init_reso=128, black_bg=hparams.black_bg)


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    train_set, train_loader, test_loader = build_dataloader(hparams)
    system = NeRFSystem(hparams, train_set)

    dir_path = f'ckpts/{hparams.exp_name}'
    os.makedirs(dir_path, exist_ok=True)

    wandb.init(project='ngp_pl', name=hparams.exp_name, dir=dir_path)
    logger = False

    trainer = Trainer(
        max_steps=int(hparams.num_epochs * 1000),
        check_val_every_n_epoch=1000000,  # no validation in timing
        callbacks=None,
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

    system.model.save(os.path.join(dir_path, 'last.ckpt'))
