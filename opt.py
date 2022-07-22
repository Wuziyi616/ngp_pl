import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_dir',
        type=str,
        required=True,
        help='root directory of dataset')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='nsvf',
        choices=['nsvf', 'colmap', 'kubric', 'kubric_flow'],
        help='which dataset to train/test')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'trainval'],
        help='use which split to train')
    parser.add_argument(
        '--timestep', type=int, default=-1, help='in video nerf')
    parser.add_argument('--timestep_start', type=int, default=-1)
    parser.add_argument('--timestep_end', type=int, default=-1)
    parser.add_argument(
        '--downsample',
        type=float,
        default=1.0,
        help='downsample factor (<=1.0) for the images')

    parser.add_argument(
        '--black_bg',
        action='store_true',
        default=False,
        help='image background is black')
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='scene scale (whole scene must lie in [-scale, scale]^3')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16384,
        help='number of rays in a batch')
    parser.add_argument(
        '--num_epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument(
        '--ft_num_epochs', type=float, default=0.2, help='number of FT epochs')
    parser.add_argument(
        '--def_num_epochs',
        type=float,
        default=1,
        help='number of deformation epochs')
    parser.add_argument(
        '--ft_interval', type=float, default=4, help='keyframing interval')
    parser.add_argument(
        '--num_gpus', type=int, default=1, help='number of gpus')

    parser.add_argument(
        '--eval_lpips',
        action='store_true',
        default=False,
        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument(
        '--val_only',
        action='store_true',
        default=False,
        help='run only validation (need to provide ckpt_path)')
    parser.add_argument(
        '--no_save_test',
        action='store_true',
        default=False,
        help='whether to save test image and video')

    parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
    parser.add_argument(
        '--ft_lr', type=float, default=1e-2, help='finetuning learning rate')
    parser.add_argument(
        '--def_lr', type=float, default=1e-3, help='deformation learning rate')

    parser.add_argument(
        '--exp_name', type=str, default='exp', help='experiment name')
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='pretrained checkpoint to load (including optimizers, etc)')

    return parser.parse_args()
