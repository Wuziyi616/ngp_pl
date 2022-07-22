#include "utils.h"


template <typename scalar_t>
__global__ void composite_train_deform_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deformed_xyzs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
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

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        // dx = f * (X + U) / (Z + W), dy = f * (Y + V) / (Z + W)
        flow[ray_idx][0] += w * fx * deformed_xyzs[s][0] / deformed_xyzs[s][2];
        flow[ray_idx][1] += w * fy * deformed_xyzs[s][1] / deformed_xyzs[s][2];

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
        // dL_doff[0] = dL_dflow[0] * dflow[0]_doff[0]
        //     dflow[0]_doff[0] = w * fx / (z + off[2])
        // dL_doff[1] = dL_dflow[1] * dflow[1]_doff[1]
        //     dflow[1]_doff[1] = w * fy / (z + off[2])
        // dL_doff[2] = dL_dflow[0] * dflow[0]_doff[2] + dL_dflow[1] * dflow[1]_doff[2]
        //     dflow[0]_doff[2] = -w * fx * (x + off[0]) / (z + off[2])^2
        //     dflow[1]_doff[2] = -w * fy * (y + off[1]) / (z + off[2])^2
        const scalar_t deform_z_2 = deformed_xyzs[s][2] * deformed_xyzs[s][2];
        dL_ddeformed_xyzs[s][0] = dL_dflow[ray_idx][0] * w * fx / deformed_xyzs[s][2];
        dL_ddeformed_xyzs[s][1] = dL_dflow[ray_idx][1] * w * fy / deformed_xyzs[s][2];
        dL_ddeformed_xyzs[s][2] = -(
            dL_dflow[ray_idx][0] * w * fx * deformed_xyzs[s][0] / deform_z_2 +
            dL_dflow[ray_idx][1] * w * fy * deformed_xyzs[s][1] / deform_z_2
        );

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

    // front to back compositing
    int s = 0; scalar_t T = 1-opacity[r];

    while (s < N_eff_samples[n]) {
        const scalar_t a = 1.0f - __expf(-sigmas[n][s]*deltas[n][s]);
        const scalar_t w = a * T;

        // dx = f * (X + U) / (Z + W), dy = f * (Y + V) / (Z + W)
        flow[r][0] += w * fx * deformed_xyzs[n][s][0] / deformed_xyzs[n][s][2];
        flow[r][1] += w * fy * deformed_xyzs[n][s][1] / deformed_xyzs[n][s][2];

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
