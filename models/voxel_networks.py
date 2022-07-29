import numpy as np

import torch
from torch import nn

from kornia.utils.grid import create_meshgrid3d

import svox2

from .networks import NGP
from .sh_utils import eval_sh
from .custom_functions import TruncExp


class SVOXNGP(NGP):

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

        # explicit voxel grid
        self.sh_basis_dim = 9
        self.svox2_grid = svox2.SparseGrid(
            reso=init_reso,
            center=self.center,
            radius=self.scale,
            use_sphere_bound=False,
            basis_dim=self.sh_basis_dim,
            use_z_order=True,
            background_nlayers=0,
        )

        self.sigma_act = TruncExp.apply
        # self.sigma_act = nn.ReLU()  # ReLU-Field

    def density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]

        Outputs:
            sigmas: (N)
        """
        sigmas = self.svox2_grid.sample(x, want_colors=False)
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
        sigmas, shs = self.svox2_grid.sample(x)
        sigmas = self.sigma_act(sigmas[:, 0])
        rgbs = eval_sh(2, shs.unflatten(-1, (3, self.sh_basis_dim)), d)
        return sigmas, rgbs
