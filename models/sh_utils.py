#  Copyright 2021 PlenOctree Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396
]
C3 = [
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    :param deg: int SH max degree. Currently, 0-4 supported
    :param sh: torch.Tensor SH coeffs (..., C, (max degree + 1) ** 2)
    :param dirs: torch.Tensor unit directions (..., 3)

    :return: (..., C)
    """
    assert deg == 2
    assert (deg + 1)**2 == sh.shape[-1]

    result = C0 * sh[..., 0]
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    result = (
        result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] -
        C1 * x * sh[..., 3])
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result + C2[0] * xy * sh[..., 4] + C2[1] * yz * sh[..., 5] + C2[2] *
        (2.0 * zz - xx - yy) * sh[..., 6] + C2[3] * xz * sh[..., 7] + C2[4] *
        (xx - yy) * sh[..., 8])

    return result
