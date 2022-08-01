import torch


def torch_deform_render(deformed_xyzs, sigmas, rgbs, deltas, ts, rays_a, camera,
                T_threshold, fx, fy):
    """
    const int N_rays = rays_a.size(0);

    auto flow = torch::zeros({N_rays, 2}, sigmas.options());
    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());


    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // w2c transformation
    const scalar_t r1 = camera[ray_idx][0], r2 = camera[ray_idx][1], r3 = camera[ray_idx][2];
    const scalar_t r4 = camera[ray_idx][4], r5 = camera[ray_idx][5], r6 = camera[ray_idx][6];
    const scalar_t r7 = camera[ray_idx][8], r8 = camera[ray_idx][9], r9 = camera[ray_idx][10];
    const scalar_t t1 = camera[ray_idx][3], t2 = camera[ray_idx][7], t3 = camera[ray_idx][11];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        // x = f * X / Z, y = f * Y / Z
        const scalar_t x = deformed_xyzs[s][0], y = deformed_xyzs[s][1], z = deformed_xyzs[s][2];
        const scalar_t w2c_x = r1 * x + r2 * y + r3 * z + t1;
        const scalar_t w2c_y = r4 * x + r5 * y + r6 * z + t2;
        const scalar_t w2c_z = r7 * x + r8 * y + r9 * z + t3;
        flow[ray_idx][0] += w * fx * w2c_x / w2c_z;
        flow[ray_idx][1] += w * fy * w2c_y / w2c_z;

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        depth[ray_idx] += w*ts[s];
        opacity[ray_idx] += w;
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    """
    N_rays = rays_a.size(0)
    flow = torch.zeros((N_rays, 2)).type_as(sigmas)
    opacity = torch.zeros((N_rays,)).type_as(sigmas)
    depth = torch.zeros((N_rays,)).type_as(sigmas)
    rgb = torch.zeros((N_rays, 3)).type_as(sigmas)

    for n in range(N_rays):
        ray_idx, start_idx, N_samples = rays_a[n][0], rays_a[n][1], rays_a[n][2]
        # w2c transformation
        r1, r2, r3 = camera[ray_idx][0], camera[ray_idx][1], camera[ray_idx][2]
        r4, r5, r6 = camera[ray_idx][4], camera[ray_idx][5], camera[ray_idx][6]
        r7, r8, r9 = camera[ray_idx][8], camera[ray_idx][9], camera[ray_idx][10]
        t1, t2, t3 = camera[ray_idx][3], camera[ray_idx][7], camera[ray_idx][11]

        # front to back compositing
        samples, T = 0, 1.0

        flow0, flow1 = 0., 0.
        rgb0, rgb1, rgb2 = 0., 0., 0.
        depth0 = 0.
        opacity0 = 0.
        while samples < N_samples:
            s = start_idx + samples
            a = 1.0 - torch.exp(-sigmas[s]*deltas[s])
            w = a * T

            # x = f * X / Z, y = f * Y / Z
            x, y, z = deformed_xyzs[s][0], deformed_xyzs[s][1], deformed_xyzs[s][2]
            w2c_x = r1 * x + r2 * y + r3 * z + t1
            w2c_y = r4 * x + r5 * y + r6 * z + t2
            w2c_z = r7 * x + r8 * y + r9 * z + t3
            flow0 = flow0 + w * fx * w2c_x / w2c_z
            flow1 = flow1 + w * fy * w2c_y / w2c_z

            rgb0 = rgb0 + w*rgbs[s][0]
            rgb1 = rgb1 + w*rgbs[s][1]
            rgb2 = rgb2 + w*rgbs[s][2]
            depth0 = depth0 + w*ts[s]
            opacity0 = opacity0 + w
            T = T * (1.0-a)

            if T <= T_threshold:
                break
            samples += 1

        flow[ray_idx][0] = flow0
        flow[ray_idx][1] = flow1
        rgb[ray_idx][0] = rgb0
        rgb[ray_idx][1] = rgb1
        rgb[ray_idx][2] = rgb2
        depth[ray_idx] = depth0
        opacity[ray_idx] = opacity0

    return flow, opacity, depth, rgb

    """ put this under __render_rays_train()
    import sys
    sys.path.append('../')
    from test_deform_render import torch_deform_render
    with torch.cuda.amp.autocast(enabled=False):
        torch_flow, torch_opacity, torch_depth, torch_rgb = torch_deform_render(deform_xyzs.float(), sigmas.float(), rgbs.contiguous().float(), deltas.float(), ts.float(), rays_a, w2c.float(), kwargs.get('T_threshold', 1e-4), fx, kwargs.get('fy', fx))
    with torch.no_grad():
        print('rgb diff', (torch_rgb - rgb).abs().max())
        print('flow diff', (torch_flow - results['flow']).abs().max())
        print('opacity diff', (torch_opacity - results['opacity']).abs().max())
        print('depth diff', (torch_depth - results['depth']).abs().max())
    cpp_rgb_loss = (rgb ** 2).sum() * 400.
    cpp_rgb_loss.backward(retain_graph=True)
    cpp_rgb_grad = model.deform_grid.grad.data.detach().clone()
    model.zero_grad()
    torch_rgb_loss = (torch_rgb ** 2).sum() * 400.
    torch_rgb_loss.backward(retain_graph=True)
    torch_rgb_grad = model.deform_grid.grad.data.detach().clone()
    model.zero_grad()
    print('rgb loss grad diff', (torch_rgb_grad - cpp_rgb_grad).abs().max())
    torch_flow_loss = (torch_flow ** 2).sum()
    torch_flow_loss.backward(retain_graph=True)
    torch_deform_grad = model.deform_grid.grad.data.detach().clone()
    model.zero_grad()
    cpp_flow_loss = (results['flow'] ** 2).sum()
    cpp_flow_loss.backward(retain_graph=True)
    cpp_deform_grad = model.deform_grid.grad.data.detach().clone()
    model.zero_grad()
    print('flow loss grad diff', (torch_deform_grad - cpp_deform_grad).abs().max())
    breakpoint()
    """
