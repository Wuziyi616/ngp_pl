import torch

from kornia.utils.grid import create_meshgrid3d

from models.voxel_networks import grid_sample, point_sample

data = torch.rand(1, 2, 4, 6, 8)
deform = torch.rand(1, 3, 4, 6, 8) / 4. - 0.125
_, C, D, H, W = data.shape

dst_coords = create_meshgrid3d(D, W, H, normalized_coordinates=True).float()
dst_coords = dst_coords.permute(0, 1, 3, 2, 4).contiguous()  # [1, ..., 3]

offsets = deform.permute(0, 2, 3, 4, 1).contiguous()  # [1, D, H, W, 3]
src_coords = torch.clamp(dst_coords + offsets, -1., 1.)
# limit deform
offsets = src_coords - dst_coords
deform = offsets.permute(0, 4, 1, 2, 3).contiguous()  # [1, 3, D, H, W]

def_data = grid_sample(data, src_coords, align_corners=True)


def pre_forward_pass(points):
    """points: [N, 3] in [-1, 1]"""
    off = point_sample(deform, points)
    def_points = points + off
    return point_sample(data, def_points)


def post_forward_pass(points):
    """points: [N, 3] in [-1, 1]"""
    return point_sample(def_data, points)


points = torch.rand(10, 3) / 4. - 0.125
pre_data = pre_forward_pass(points)
post_data = post_forward_pass(points)
