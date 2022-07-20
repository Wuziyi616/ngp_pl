import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn

from .networks import NGP


class DeformNGP(nn.Module):

    def __init__(self, ngp_model: NGP):
        super().__init__()

        self.ngp_model = ngp_model.eval()
        for p in self.ngp_model.parameters():
            p.requires_grad = False

        # constants, same as NGP
        L = 16
        F = 2
        log2_T = 19
        N_min = 16

        self.delta_xyz = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale":
                np.exp(np.log(2048 * ngp_model.scale / N_min) / (L - 1)),
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            })

    def deform(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]

        Outputs:
            x': (N, 3), xyz after deformation
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        dx = self.delta_xyz(x)  # sigmoid output
        # to [-1, 1]
        dx = dx * 2. - 1.
        return x + dx

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = self.deform(x)
        h = self.ngp_model.xyz_encoder(x)
        sigmas = self.ngp_model.sigma_act(h[:, 0])
        if return_feat:
            return sigmas, h
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
        sigmas, h = self.density(x, return_feat=True)
        # d /= torch.norm(d, dim=-1, keepdim=True)
        d = self.ngp_model.dir_encoder((d + 1.) / 2.)
        rgbs = self.ngp_model.rgb_net(torch.cat([d, h], 1))
        return sigmas, rgbs

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        # ngp_model should always in be in eval mode
        self.ngp_model.eval()
        return self

    @property
    def scale(self):
        return self.ngp_model.scale

    @property
    def center(self):
        return self.ngp_model.center

    @property
    def xyz_min(self):
        return self.ngp_model.xyz_min

    @property
    def xyz_max(self):
        return self.ngp_model.xyz_max

    @property
    def half_size(self):
        return self.ngp_model.half_size

    @property
    def cascades(self):
        return self.ngp_model.cascades

    @property
    def grid_size(self):
        return self.ngp_model.grid_size

    @property
    def bg_color(self):
        return self.ngp_model.bg_color

    @property
    def density_bitfield(self):
        return self.ngp_model.density_bitfield
