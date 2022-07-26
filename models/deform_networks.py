import numpy as np

import torch
import tinycudann as tcnn

from .networks import NGP


def get_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt = {k[6:]: v for k, v in ckpt.items() if k.startswith('model.')}
    return ckpt


class DeformNGP(NGP):
    """NGP model with deformation field."""

    def __init__(
        self,
        ckpt_path,
        scale=0.5,
        black_bg=False,
        ft_rgb=False,
        ret_xyz=False,
    ):
        super().__init__(scale, black_bg)
        self.load_state_dict(get_ckpt(ckpt_path))

        self.ft_rgb = ft_rgb
        # fix params of ngp_model
        for kv in self.named_parameters():
            if self.ft_rgb and 'rgb_net' in kv[0]:
                continue
            kv[1].requires_grad = False

        # constants, same as NGP
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        b = np.exp(np.log(2048 * self.scale / N_min) / (L - 1))

        self.delta_xyz = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            })

        self.ret_xyz = ret_xyz  # to render flow

    def _normalize_xyz(self, x):
        return (x - self.xyz_min) / (self.xyz_max - self.xyz_min)

    def _denormalize_xyz(self, x):
        return x * (self.xyz_max - self.xyz_min) + self.xyz_min

    def deform(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]

        Outputs:
            x': (N, 3), xyz after deformation
        """
        x = self._normalize_xyz(x)
        dx = self.delta_xyz(x)  # sigmoid output
        dx = dx * 2. - 1.  # to [-1, 1]
        return x + dx

    def density(self, x, return_feat=False, return_xyz=False):
        """
        Inputs:
            x: (N, 3) xyz after deformation
            return_feat: whether to return intermediate feature
            return_xyz: whether to return xyz after deformation

        Outputs:
            sigmas: (N)
        """
        x = self.deform(x)
        h = self.xyz_encoder(x)
        sigmas = self.sigma_act(h[:, 0])
        ret_lst = [sigmas]
        if return_feat:
            ret_lst.append(h)
        if return_xyz:
            ret_lst.append(x)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def forward(self, x, d):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            deform_x: (N, 3), x after deformation, denormalized to original scale
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h, deform_x = self.density(
            x, return_feat=True, return_xyz=True)
        d = self.dir_encoder((d + 1.) / 2.)
        rgbs = self.rgb_net(torch.cat([d, h], 1))
        # return xyz
        if self.ret_xyz:
            # denormalize xyz to original scale
            deform_x = self._denormalize_xyz(deform_x)
            return deform_x, sigmas, rgbs
        return sigmas, rgbs

    def train(self, mode: bool = True):
        super().train(mode)
        # ngp_model should always in be in eval mode
        for kv in self.named_modules():
            if not kv[0]:  # model itself is ''
                continue
            if 'delta_xyz' in kv[0]:
                continue
            if self.ft_rgb and 'rgb_net' in kv[0]:
                continue
            kv[1].eval()
        return self
