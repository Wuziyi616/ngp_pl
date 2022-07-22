import os
import time

import warnings
import wandb

# data
from utils import build_dataloader

# models
from models.networks import NGP

# pytorch-lightning
from pytorch_lightning.utilities.seed import seed_everything

# misc.
from opt import get_opts
from finetune import build_trainer as build_ft_trainer
from finetune import NeRFSystem as FTNeRFSystem
from deform import build_trainer as build_def_trainer
from deform import NeRFSystem as DefNeRFSystem
from deform import get_ckpt

warnings.filterwarnings("ignore")
seed_everything(19870203)

if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    dir_path = f'ckpts/{hparams.exp_name}'
    os.makedirs(dir_path, exist_ok=True)
    callbacks = None

    wandb.init(project='ngp_pl', name=hparams.exp_name, dir=dir_path)

    trainer = build_def_trainer(hparams, callbacks=callbacks)

    # pretrained NGP as template model
    ngp_model = NGP(scale=hparams.scale, black_bg=hparams.black_bg).eval()
    ngp_model.load_state_dict(get_ckpt(hparams))

    system = None
    for timestep in range(hparams.timestep_start, hparams.timestep_end):
        print('#######################################################')
        print(f'Timestep: {timestep}')
        hparams.timestep = timestep
        train_set, train_loader, test_loader = build_dataloader(hparams)
        if system is None:
            system = DefNeRFSystem(hparams, train_set, ngp_model)
            trainer.validate(system, test_loader)
            continue
        else:
            # do finetuning on keyframes
            if timestep % hparams.ft_interval == 0:
                state_dict = system.model.ngp_model.state_dict()
                system = FTNeRFSystem(hparams, train_set)
                system.model.load_state_dict(state_dict)
                system.model.reset_density_field()
                trainer = build_ft_trainer(hparams, callbacks=callbacks)
            # still do deformation field
            else:
                # just finish one finetuning
                if (timestep - 1) % hparams.ft_interval == 0 and timestep != 1:
                    state_dict = system.model.state_dict()
                    system = DefNeRFSystem(hparams, train_set, ngp_model)
                    system.model.ngp_model.load_state_dict(state_dict)
                    trainer = build_def_trainer(hparams, callbacks=callbacks)
                else:
                    state_dict = system.model.state_dict()
                    system = DefNeRFSystem(hparams, train_set, ngp_model)
                    system.model.load_state_dict(state_dict)
                    trainer = build_def_trainer(hparams, callbacks=callbacks)

        start_t = time.time()
        trainer.fit(system, train_loader)
        end_t = time.time()
        print(f'\tTraining timestep {timestep} took {end_t - start_t:.2f}s\n')
        trainer.validate(system, test_loader)
