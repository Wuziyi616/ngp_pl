import torch
from einops import rearrange

import vren
from .rendering import MAX_SAMPLES, NEAR_DISTANCE
from .custom_functions import RayAABBIntersector, RayMarcher, DeformVolumeRenderer


def deform_render(model, rays, w2c, bg_flow, fx, wh, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: DeformNGP
        rays: (N_rays, 3+3), ray origins and directions
        w2c: (N_rays, 3*4), world to camera transformation matrix
        bg_flow: (N_rays, 2), default flow
        fx: focal length of the camera
        wh: (w, h) of the image

    Outputs:
        result: dictionary containing final rgb, depth, opacity and flow
    """

    rays_o, rays_d = rays[:, 0:3].contiguous(), rays[:, 3:6].contiguous()
    _, hits_t, _ = RayAABBIntersector.apply(rays_o, rays_d, model.center,
                                            model.half_size, 1)
    hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0,
           0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, w2c, fx, **kwargs)
    # TODO: normalize the flow
    if 'flow' in results:
        flow = results['flow']
        flow = torch.stack([
            flow[..., 0] / wh[0],
            flow[..., 1] / wh[1],
        ], -1)
        flow = flow + bg_flow * (1. - results['opacity']).unsqueeze(-1)
        results['flow'] = flow
    for k, v in results.items():
        results[k] = v.cpu() if kwargs.get('to_cpu', False) else v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, w2c, fx, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples
           and evaluate the properties (offsets, sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    T_threshold = kwargs.get('T_threshold', 1e-4)
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    fy = kwargs.get('fy', fx)
    device = rays_o.device
    flow = torch.zeros(N_rays, 2, device=device)
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = 0
    alive_indices = torch.arange(N_rays, device=device)

    while samples < MAX_SAMPLES:
        N_alive = len(alive_indices)
        if N_alive == 0:
            break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), 1)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs == 0, dim=1)
        if valid_mask.sum() == 0:
            break

        deform_xyzs = torch.zeros(len(xyzs), 3, device=device)
        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        _deform_xyzs, _sigmas, _rgbs = \
            model(xyzs[valid_mask], dirs[valid_mask])
        deform_xyzs[valid_mask], sigmas[valid_mask], rgbs[valid_mask] = \
            _deform_xyzs.float(), _sigmas.float(), _rgbs.float()
        xyzs = rearrange(xyzs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        deform_xyzs = rearrange(
            deform_xyzs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_deform_fw(deform_xyzs, sigmas, rgbs, deltas, ts,
                                      hits_t[:, 0], alive_indices, w2c,
                                      T_threshold, fx, fy, N_eff_samples, flow,
                                      opacity, depth, rgb)
        # remove converged rays
        alive_indices = alive_indices[alive_indices >= 0]

    rgb_bg = torch.ones(3, device=device) * model.bg_color
    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb + rgb_bg * (1. - opacity).unsqueeze(-1)
    results['flow'] = flow

    return results


@torch.cuda.amp.autocast()
def __render_rays_train(model, rays_o, rays_d, hits_t, w2c, fx, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently offsets, sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    rays_a, xyzs, dirs, deltas, ts, total_samples = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['total_samples'] = total_samples

    deform_xyzs, sigmas, rgbs = model(xyzs, dirs)

    results['flow'], results['opacity'], results['depth'], rgb = \
        DeformVolumeRenderer.apply(
            deform_xyzs, sigmas, rgbs.contiguous(), deltas, ts, rays_a, w2c,
            kwargs.get('T_threshold', 1e-4), fx, kwargs.get('fy', fx))

    rgb_bg = torch.ones(3, device=rays_o.device) * model.bg_color
    results['rgb'] = rgb + rgb_bg * (1. - results['opacity']).unsqueeze(-1)

    return results
