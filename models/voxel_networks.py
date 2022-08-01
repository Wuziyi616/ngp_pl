import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from kornia.utils.grid import create_meshgrid3d

from .networks import NGP
from .sh_utils import eval_sh
from .custom_functions import TruncExp


def grid_sample(data, grid, align_corners=False):
    """Wrapper over pytorch's grid_sample.

    Inputs:
        data: [1, C, D, H, W]
        grid: [1, d, h, w, 3] in [-1, 1]

    Outputs:
        data: [1, C, d, h, w]
    """
    # https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/3
    # the 3-tuple of `grid` is treated as (k, j, i), i.e. (z, y, x)
    return F.grid_sample(
        data.permute(0, 1, 4, 3, 2),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=align_corners,
    )


def point_sample(data, points, transpose=True):
    """Wrapper over pytorch's grid_sample.

    Inputs:
        data: [1, C, D, H, W]
        points: [N, 3] in [-1, 1]

    Outputs:
        data: [N, C] or [C, N] if not `transpose`
    """
    N = points.shape[0]
    points = points.view(1, N, 1, 1, 3)
    sample_data = grid_sample(data, points, align_corners=True).view(-1, N)
    if not transpose:
        return sample_data
    return sample_data.transpose(0, 1).contiguous()


class SimpleGrid(nn.Module):
    """Simple 3D voxel grid storing data.

    Supports grid_sample, upsample.
    """

    def __init__(self, reso, basis_dim):
        super().__init__()

        self.grid_size = reso
        self.basis_dim = basis_dim

        self.density_grid = nn.Parameter(
            torch.zeros((1, 1, reso, reso, reso), dtype=torch.float32))
        self.sh_grid = nn.Parameter(
            torch.zeros((1, self.basis_dim * 3, reso, reso, reso),
                        dtype=torch.float32))

    def sample(self, points: torch.Tensor, want_colors=True):
        """
        Grid sampling with trilinear interpolation.

        :param points: torch.Tensor, (N, 3)
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (density, color)
        """
        sigmas = point_sample(self.density_grid, points)
        if not want_colors:
            return sigmas, None
        colors = point_sample(self.sh_grid, points)
        return sigmas, colors

    @torch.no_grad()
    def refine(self, src_coords, dst_coords, mask):
        """Refine `density_grid` and `sh_grid` by applying deformation.

        Inputs:
            src_coords: coordinates before deformation, [N, 3]
            dst_coords: coordinates after deformation, [N, 3].
                Both coords are in [-1, 1].
            mask: [D, H, W], 1 is the coords to be refined
        """
        # assert dst_coords.shape[1:4] == self.density_grid.shape[-3:]
        assert mask.shape == self.density_grid.shape[-3:]
        # breakpoint()
        if not mask.any():
            return
        # resample to update grids
        self.density_grid.data[0][:, mask] = point_sample(
            self.density_grid.data, src_coords, transpose=False)
        self.sh_grid.data[0][:, mask] = point_sample(
            self.sh_grid.data, src_coords, transpose=False)

    @torch.no_grad()
    def upsample2x(self):
        """Upsample grids by 2x."""
        print(f'Upsample {self.grid_size} --> {self.grid_size * 2}')
        print('Please re-init optimizer in the Trainer!')
        self.density_grid.data = F.interpolate(
            self.density_grid.data,
            scale_factor=2,
            mode='trilinear',
            align_corners=False,
        )
        self.sh_grid.data = F.interpolate(
            self.sh_grid.data,
            scale_factor=2,
            mode='trilinear',
            align_corners=False,
        )
        self.grid_size *= 2


class SVOXNGP(NGP):
    """Explicit voxel representation + NGP like occupancy modeling."""

    def __init__(self, scale=0.5, init_reso=128, black_bg=False):
        nn.Module.__init__(self)

        # scene bounding box
        self.scale = scale
        assert self.scale == 0.5, 'SVOXNGP only supports 0.5 scale'

        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2.)

        # background color
        self.bg_color = 0. if black_bg else 1.

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        assert self.cascades == 1, 'SVOXNGP currently only supports 1 cascade'
        self.grid_size = init_reso
        self._init_occ_grids()

        # explicit voxel grid
        self.sh_basis_dim = 9
        self.svox2_grid = SimpleGrid(
            reso=self.grid_size,
            basis_dim=self.sh_basis_dim,
        )

        self.sigma_act = TruncExp.apply
        self.rgb_act = nn.Sigmoid()

    def _init_occ_grids(self):
        self.register_buffer(
            'density_bitfield',
            torch.zeros(
                self.cascades * self.grid_size**3 // 8, dtype=torch.uint8))
        self.register_buffer('density_grid',
                             torch.zeros(self.cascades, self.grid_size**3))
        self.register_buffer(
            'grid_coords',
            create_meshgrid3d(
                self.grid_size,
                self.grid_size,
                self.grid_size,
                False,
                dtype=torch.int32).reshape(-1, 3))

    def _normalize_x(self, x):
        return x / self.scale

    def _denormalize_x(self, x):
        return x * self.scale

    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]

        Outputs:
            sigmas: (N)
        """
        x = self._normalize_x(x)
        sigmas = self.svox2_grid.sample(x, want_colors=False)[0]
        sigmas = self.sigma_act(sigmas[:, 0])
        return sigmas

    def forward(self, x, d):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        x = self._normalize_x(x)
        sigmas, shs = self.svox2_grid.sample(x)
        sigmas = self.sigma_act(sigmas[:, 0])
        rgbs = eval_sh(2, shs.unflatten(-1, (3, self.sh_basis_dim)), d)
        rgbs = self.rgb_act(rgbs)
        return sigmas, rgbs

    def upsample2x(self):
        self.svox2_grid.upsample2x()
        self.grid_size *= 2
        self._init_occ_grids()
