// Native CUDA preparation for the copied SageAttention3 Blackwell core.
// Route each (batch, head, 128-key tile) using the same V-FP4 relative L2
// criterion as E113, and prepack V to E4M3 with one FP32 scale per output
// channel / 128-key tile. The fused PV consumer is a separate follow-up.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tuple>

namespace {

__device__ __forceinline__ float e2m1_round(float x) {
    const float a = fabsf(x);
    float q = a >= 5.0f ? 6.0f :
              a > 3.5f ? 4.0f :
              a >= 2.5f ? 3.0f :
              a > 1.75f ? 2.0f :
              a >= 1.25f ? 1.5f :
              a > 0.75f ? 1.0f :
              a >= 0.25f ? 0.5f : 0.0f;
    return copysignf(q, x);
}

__global__ void route_pack_v(
    const __nv_bfloat16* __restrict__ v,
    __nv_fp8_e4m3* __restrict__ v_fp8,
    float* __restrict__ v_scale,
    uint8_t* __restrict__ use_fp8,
    float* __restrict__ relative_error_sq,
    int heads, int tokens, int dim, int tiles, float threshold_sq) {
    const int tile = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int d = threadIdx.x;
    __shared__ float err_sum[128];
    __shared__ float energy_sum[128];
    float err = 0.0f;
    float energy = 0.0f;

    if (d < dim) {
        const int base_v = ((batch * heads + head) * tokens + tile * 128) * dim + d;
        const int base_fp8 = ((batch * heads + head) * dim + d) * tokens + tile * 128;
        float tile_absmax = 0.0f;
        for (int group = 0; group < 8; ++group) {
            float values[16];
            float group_absmax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float x = __bfloat162float(v[base_v + (group * 16 + i) * dim]);
                values[i] = x;
                group_absmax = fmaxf(group_absmax, fabsf(x));
            }
            tile_absmax = fmaxf(tile_absmax, group_absmax);
            const float scale = static_cast<float>(__nv_fp8_e4m3(group_absmax / 6.0f));
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float x = values[i];
                const float reconstructed = scale == 0.0f ? 0.0f : e2m1_round(x / scale) * scale;
                const float delta = reconstructed - x;
                err += delta * delta;
                energy += x * x;
            }
        }
        const float fp8_scale = fmaxf(tile_absmax / 448.0f, 1.0e-12f);
        v_scale[((batch * heads + head) * dim + d) * tiles + tile] = fp8_scale;
        #pragma unroll 1
        for (int i = 0; i < 128; ++i) {
            const float x = __bfloat162float(v[base_v + i * dim]);
            const float normalized = fminf(448.0f, fmaxf(-448.0f, x / fp8_scale));
            v_fp8[base_fp8 + i] = __nv_fp8_e4m3(normalized);
        }
    }
    err_sum[d] = err;
    energy_sum[d] = energy;
    __syncthreads();
    for (int offset = 64; offset > 0; offset >>= 1) {
        if (d < offset) {
            err_sum[d] += err_sum[d + offset];
            energy_sum[d] += energy_sum[d + offset];
        }
        __syncthreads();
    }
    if (d == 0) {
        const int index = (batch * heads + head) * tiles + tile;
        relative_error_sq[index] = energy_sum[0] > 0.0f ? err_sum[0] / energy_sum[0] : 0.0f;
        use_fp8[index] = err_sum[0] >= threshold_sq * energy_sum[0];
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
route_and_pack_v(torch::Tensor v, double threshold) {
    TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kBFloat16, "V must be CUDA BF16");
    TORCH_CHECK(v.dim() == 4 && v.is_contiguous(), "V must be contiguous [B,H,K,D]");
    TORCH_CHECK(v.size(3) == 64 || v.size(3) == 128, "head dim must be 64 or 128");
    TORCH_CHECK(v.size(2) % 128 == 0, "K must be padded to a multiple of 128");
    TORCH_CHECK(threshold > 0.0 && threshold < 1.0, "threshold must be in (0,1)");
    c10::cuda::CUDAGuard guard(v.device());
    const auto b = v.size(0), h = v.size(1), k = v.size(2), d = v.size(3);
    const auto tiles = k / 128;
    auto fp8 = torch::empty({b, h, d, k}, v.options().dtype(at::kFloat8_e4m3fn));
    auto scale = torch::empty({b, h, d, tiles}, v.options().dtype(at::kFloat));
    auto mask = torch::empty({b, h, tiles}, v.options().dtype(at::kByte));
    auto err = torch::empty({b, h, tiles}, v.options().dtype(at::kFloat));
    route_pack_v<<<dim3(tiles, h, b), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(fp8.data_ptr()),
        scale.data_ptr<float>(), mask.data_ptr<uint8_t>(), err.data_ptr<float>(),
        h, k, d, tiles, float(threshold * threshold));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fp8, scale, mask, err};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("route_and_pack_v", &route_and_pack_v,
          "Sage3 V tile FP4-error routing and FP8-group128 preparation");
}
