import os
import time

import imageio
import warnings
import wandb

from einops import rearrange

# data
from utils import build_dataloader

# models
from models.deform_networks import DeformNGP
from models.deform_rendering import deform_render

# metrics
from torchmetrics import MeanSquaredError

# pytorch-lightning
from pytorch_lightning.utilities.seed import seed_everything

# misc.
from deform import build_trainer
from deform import NeRFSystem as BaseNeRFSystem
from opt import get_opts
from utils import flow2img

warnings.filterwarnings("ignore")
seed_everything(19870203)


class NeRFSystem(BaseNeRFSystem):

    def __init__(self, hparams, train_dataset):
        super().__init__(hparams, train_dataset)

        self.val_flow_mse = MeanSquaredError()
        self.model = DeformNGP(
            ckpt_path=hparams.ckpt_path,
            scale=hparams.scale,
            black_bg=hparams.black_bg,
            ft_rgb=hparams.ft_rgb,
            ret_xyz=True,
        )

    def forward(self, batch_data, split):
        kwargs = {'test_time': split != 'train'}
        if self.hparams.dataset_name == 'colmap':
            kwargs['exp_step_factor'] = 1. / 256.

        return deform_render(
            self.model,
            batch_data['rays'],
            w2c=batch_data['w2c'],
            bg_flow=batch_data['bg_flow'],
            fx=self.train_dataset.fx,
            fy=self.train_dataset.fy,
            wh=self.train_dataset.img_wh,
            **kwargs,
        )

    def validation_step(self, batch, batch_nb):
        logs = super().validation_step(batch, batch_nb)

        flow_gt = batch['flow']
        results = self(batch, split='test')

        # save test image to disk
        if not self.hparams.no_save_test and self.global_rank == 0:
            idx = batch['idx']
            w, h = self.train_dataset.img_wh
            # flow visualization
            flow_pred = rearrange(
                results['flow'] - batch['bg_flow'], '(h w) c -> h w c', h=h)
            flow_pred = flow2img(flow_pred)
            flow_gt = rearrange(
                flow_gt - batch['bg_flow'], '(h w) c -> h w c', h=h)
            flow_gt = flow2img(flow_gt)
            fn = f'{self.hparams.timestep}-{idx:03d}-pred_flow.png'
            imageio.imsave(os.path.join(self.val_dir, fn), flow_pred)
            logs['flow_pred_img'] = flow_pred
            fn = f'{self.hparams.timestep}-{idx:03d}-gt_flow.png'
            imageio.imsave(os.path.join(self.val_dir, fn), flow_gt)
            logs['flow_gt_img'] = flow_gt

        return logs


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
