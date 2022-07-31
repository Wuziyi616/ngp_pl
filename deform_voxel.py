import os
import time

import warnings
import wandb

# data
from utils import build_dataloader

# models
from models.deform_voxel_network import DeformSVOXNGP

# optimizer, losses
from apex.optimizers import FusedAdam
from losses import DeformNeRFLoss

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

# misc.
from finetune_voxel import NeRFSystem as BaseNeRFSystem
from opt import get_opts

warnings.filterwarnings("ignore")
seed_everything(19870203)


class NeRFSystem(BaseNeRFSystem):

    def __init__(self, hparams, train_dataset):
        super().__init__(hparams, train_dataset)

        self.epoch_it = int(hparams.def_num_epochs * 1000)
        self.loss = DeformNeRFLoss(
            lambda_opa=hparams.occ_loss_w, lambda_flow=hparams.flow_loss_w)
        self.model = DeformSVOXNGP(
            ckpt_path=hparams.ckpt_path,
            scale=hparams.scale,
            init_reso=128,
            black_bg=hparams.black_bg,
            ft_rgb=hparams.ft_rgb,
        )

    def configure_optimizers(self):
        if not self.hparams.ft_rgb:
            params_list = filter(lambda p: p.requires_grad,
                                 self.model.parameters())
        else:
            deform_params = [
                kv[1] for kv in self.model.named_parameters()
                if 'deform_grid' in kv[0]
            ]
            rgb_params = [
                kv[1] for kv in self.model.named_parameters()
                if 'sh_grid' in kv[0]
            ]
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
            # sanity-check
            # system.model.merge_deformation()
            # trainer.validate(system, test_loader)
            continue
        else:
            # merge previous deformation field to grid data
            system.model.merge_deformation()
            # keep the same model params, update dataset
            state_dict = system.model.state_dict()
            system = NeRFSystem(hparams, train_set)
            system.model.load_state_dict(state_dict)
            trainer = build_trainer(hparams, callbacks=callbacks)

        start_t = time.time()
        trainer.fit(system, train_loader)
        end_t = time.time()
        print(f'\tTraining timestep {timestep} took {end_t - start_t:.2f}s\n')
        trainer.validate(system, test_loader)
