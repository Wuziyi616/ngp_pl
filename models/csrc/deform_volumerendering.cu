#include "utils.h"


template <typename scalar_t>
__global__ void composite_train_deform_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deformed_xyzs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> camera,
    const scalar_t T_threshold,
    const scalar_t fx,
    const scalar_t fy,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> flow,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb
){
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
}


std::vector<torch::Tensor> composite_train_deform_fw_cu(
    const torch::Tensor deformed_xyzs,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor camera,
    const float T_threshold,
    const float fx,
    const float fy
){
    const int N_rays = rays_a.size(0);

    auto flow = torch::zeros({N_rays, 2}, sigmas.options());
    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_deform_fw_cu", 
    ([&] {
        composite_train_deform_fw_kernel<scalar_t><<<blocks, threads>>>(
            deformed_xyzs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            camera.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            fx,
            fy,
            flow.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {flow, opacity, depth, rgb};
}


template <typename scalar_t>
__global__ void composite_train_deform_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dflow,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dopacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_ddepth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgb,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deformed_xyzs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> camera,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> flow,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    const scalar_t T_threshold,
    const scalar_t fx,
    const scalar_t fy,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_ddeformed_xyzs,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgbs
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // w2c transformation
    const scalar_t r1 = camera[ray_idx][0], r2 = camera[ray_idx][1], r3 = camera[ray_idx][2];
    const scalar_t r4 = camera[ray_idx][4], r5 = camera[ray_idx][5], r6 = camera[ray_idx][6];
    const scalar_t r7 = camera[ray_idx][8], r8 = camera[ray_idx][9], r9 = camera[ray_idx][10];
    const scalar_t t1 = camera[ray_idx][3], t2 = camera[ray_idx][7], t3 = camera[ray_idx][11];

    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0], G = rgb[ray_idx][1], B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx], D = depth[ray_idx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f, d = 0.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        d += w*ts[s];
        T *= 1.0f-a;

        // compute gradients by math...
        const scalar_t x = deformed_xyzs[s][0], y = deformed_xyzs[s][1], z = deformed_xyzs[s][2];
        const scalar_t w2c_x = r1 * x + r2 * y + r3 * z + t1;
        const scalar_t w2c_y = r4 * x + r5 * y + r6 * z + t2;
        const scalar_t w2c_z = r7 * x + r8 * y + r9 * z + t3;
        const scalar_t w2c_z_2 = w2c_z * w2c_z;

        dL_ddeformed_xyzs[s][0] = dL_dflow[ray_idx][0] * w * fx * (r1 / w2c_z - r7 * w2c_x / w2c_z_2) +
                                  dL_dflow[ray_idx][1] * w * fy * (r4 / w2c_z - r7 * w2c_y / w2c_z_2);
        dL_ddeformed_xyzs[s][1] = dL_dflow[ray_idx][0] * w * fx * (r2 / w2c_z - r8 * w2c_x / w2c_z_2) +
                                  dL_dflow[ray_idx][1] * w * fy * (r5 / w2c_z - r8 * w2c_y / w2c_z_2);
        dL_ddeformed_xyzs[s][2] = dL_dflow[ray_idx][0] * w * fx * (r3 / w2c_z - r9 * w2c_x / w2c_z_2) +
                                  dL_dflow[ray_idx][1] * w * fy * (r6 / w2c_z - r9 * w2c_y / w2c_z_2);

        dL_drgbs[s][0] = dL_drgb[ray_idx][0]*w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1]*w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2]*w;

        dL_dsigmas[s] = deltas[s] * (
            dL_drgb[ray_idx][0]*(rgbs[s][0]*T-(R-r)) + 
            dL_drgb[ray_idx][1]*(rgbs[s][1]*T-(G-g)) + 
            dL_drgb[ray_idx][2]*(rgbs[s][2]*T-(B-b)) + 
            dL_dopacity[ray_idx]*(1-O) + 
            dL_ddepth[ray_idx]*(ts[s]*T-(D-d))
        );
        // mathematically flow should also contribute to sigmas' gradient
        // but we apply a stop_grad here, because deformation field shouldn't
        // affect the density field (i.e. the geometry of the object)

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> composite_train_deform_bw_cu(
    const torch::Tensor dL_dflow,
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor deformed_xyzs,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor camera,
    const torch::Tensor flow,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const float T_threshold,
    const float fx,
    const float fy
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_ddeformed_xyzs = torch::zeros({N, 3}, sigmas.options());
    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_drgbs = torch::zeros({N, 3}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_deform_bw_cu", 
    ([&] {
        composite_train_deform_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dflow.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dopacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_ddepth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deformed_xyzs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            camera.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            flow.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            fx,
            fy,
            dL_ddeformed_xyzs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_ddeformed_xyzs, dL_dsigmas, dL_drgbs};
}


template <typename scalar_t>
__global__ void composite_test_deform_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> deformed_xyzs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> hits_t,
    torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> alive_indices,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> camera,
    const scalar_t T_threshold,
    const scalar_t fx,
    const scalar_t fy,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> flow,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    if (N_eff_samples[n]==0){ // no hit
        alive_indices[n] = -1;
        return;
    }

    const size_t r = alive_indices[n]; // ray index

    // w2c transformation
    const scalar_t r1 = camera[0], r2 = camera[1], r3 = camera[2];
    const scalar_t r4 = camera[4], r5 = camera[5], r6 = camera[6];
    const scalar_t r7 = camera[8], r8 = camera[9], r9 = camera[10];
    const scalar_t t1 = camera[3], t2 = camera[7], t3 = camera[11];

    // front to back compositing
    int s = 0; scalar_t T = 1-opacity[r];

    while (s < N_eff_samples[n]) {
        const scalar_t a = 1.0f - __expf(-sigmas[n][s]*deltas[n][s]);
        const scalar_t w = a * T;

        // x = f * X / Z, y = f * Y / Z
        const scalar_t x = deformed_xyzs[n][s][0], y = deformed_xyzs[n][s][1], z = deformed_xyzs[n][s][2];
        const scalar_t w2c_x = r1 * x + r2 * y + r3 * z + t1;
        const scalar_t w2c_y = r4 * x + r5 * y + r6 * z + t2;
        const scalar_t w2c_z = r7 * x + r8 * y + r9 * z + t3;
        flow[r][0] += w * fx * w2c_x / w2c_z;
        flow[r][1] += w * fy * w2c_y / w2c_z;

        rgb[r][0] += w*rgbs[n][s][0];
        rgb[r][1] += w*rgbs[n][s][1];
        rgb[r][2] += w*rgbs[n][s][2];
        depth[r] += w*ts[n][s];
        opacity[r] += w;
        T *= 1.0f-a;

        if (T <= T_threshold){ // ray has enough opacity
            alive_indices[n] = -1;
            break;
        }
        s++;
    }
}


void composite_test_deform_fw_cu(
    const torch::Tensor deformed_xyzs,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
    const torch::Tensor camera,
    const float T_threshold,
    const float fx,
    const float fy,
    const torch::Tensor N_eff_samples,
    torch::Tensor flow,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb
){
    const int N_rays = alive_indices.size(0);

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_test_deform_fw_cu", 
    ([&] {
        composite_test_deform_fw_kernel<scalar_t><<<blocks, threads>>>(
            deformed_xyzs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            sigmas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            hits_t.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            alive_indices.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
            camera.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
            T_threshold,
            fx,
            fy,
            N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            flow.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
}
