import torch
from torch.utils.data import DataLoader

from datasets import dataset_dict


def build_dataloader(hparams):
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {
        'root_dir': hparams.root_dir,
        'downsample': hparams.downsample,
        'timestep': hparams.timestep,
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
