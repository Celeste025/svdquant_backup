/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "params.h"
#include "launch.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t unpadded_seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor delta_s,
                      at::Tensor out,
                      const at::Tensor sfq,
                      const at::Tensor sfk,
                      const at::Tensor sfv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool per_block_mean,
                      bool is_bf16,
                      bool seqlenq_ngroups_swapped=false) {

    // Reset the parameters
    params = {};
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.delta_s_ptr = delta_s.data_ptr();
    params.sfq_ptr = sfq.data_ptr();
    params.sfk_ptr = sfk.data_ptr();
    params.sfv_ptr = sfv.data_ptr();
    
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-2) * 2;
    params.k_row_stride = k.stride(-2) * 2;
    params.v_row_stride = v.stride(-2) * 2;;
    params.q_head_stride = q.stride(-3) * 2;
    params.k_head_stride = k.stride(-3) * 2;
    params.v_head_stride = v.stride(-3) * 2; // for packed q k v

    params.ds_row_stride = delta_s.stride(-2);
    params.ds_head_stride = delta_s.stride(-3);
    
    params.sfq_row_stride = sfq.stride(-2);
    params.sfk_row_stride = sfk.stride(-2);
    params.sfv_row_stride = sfv.stride(-2);
    params.sfq_head_stride = sfq.stride(-3);
    params.sfk_head_stride = sfk.stride(-3);
    params.sfv_head_stride = sfv.stride(-3);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-2);
    params.o_head_stride = out.stride(-3);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0) * 2;
        params.k_batch_stride = k.stride(0) * 2;
        params.v_batch_stride = v.stride(0) * 2;
        params.ds_batch_stride = delta_s.stride(0);
        params.sfq_batch_stride = sfq.stride(0);
        params.sfk_batch_stride = sfk.stride(0);
        params.sfv_batch_stride = sfv.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.unpadded_seqlen_k = unpadded_seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.head_divmod = cutlass::FastDivmod(int(h));

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.per_block_mean = per_block_mean;
    if (per_block_mean) {
        params.seqlen_s = seqlen_q;
    } else {
        params.seqlen_s = 128; // size of BLOCK_M
    }
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;
    params.is_bf16 = is_bf16;
    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif
}

template<bool IsBF16>
void run_mha_fwd_dispatch_dtype(Flash_fwd_params &params, cudaStream_t stream) {
    using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
    if (params.d == 64) {
        run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 64, OType>(params, stream);
    } else if (params.d == 128) {
        run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 128, OType>(params, stream);
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
    BOOL_SWITCH(params.is_bf16, IsBF16, ([&] {
        run_mha_fwd_dispatch_dtype<IsBF16>(params, stream);
    }));
}

std::vector<at::Tensor>
mha_fwd_impl(at::Tensor &q,         // batch_size x seqlen_q x num_heads x (head_size // 2)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x (head_size // 2)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x (head_size // 2)
        const at::Tensor &sfq,
        const at::Tensor &sfk,
        const at::Tensor &sfv,
        const at::Tensor &delta_s,
        int unpadded_k,
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        bool is_causal, 
        bool per_block_mean,
        bool is_bf16,
        const at::Tensor* v8,
        const at::Tensor* v8_scale,
        const at::Tensor* v8_route
    ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm120 = dprops->major == 12 && dprops->minor == 0;
    bool is_sm121 = dprops->major == 12 && dprops->minor == 1;
    TORCH_CHECK(is_sm120 || is_sm121, "only supports Blackwell GPUs or newer.");

    auto q_dtype = q.dtype();
    auto sfq_dtype = sfq.dtype();
    TORCH_CHECK(q_dtype == torch::kUInt8, "q dtype must be uint8");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(sfq_dtype == torch::kFloat8_e4m3fn, "q dtype must be uint8");
    TORCH_CHECK(sfk.dtype() == sfq_dtype, "query and key must have the same dtype");
    TORCH_CHECK(sfv.dtype() == sfq_dtype, "query and value must have the same dtype");
    CHECK_DEVICE(sfq); CHECK_DEVICE(sfk); CHECK_DEVICE(sfv);
    
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(delta_s.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    TORCH_CHECK(q.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor must be contiguous");

    const auto sizes = q.sizes();
    auto opts = q.options();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[2];
    int num_heads = sizes[1];
    const int head_size_og = sizes[3];
    const int unpacked_head_size = head_size_og * 2;
    const int seqlen_k = k.size(2);
    const int num_heads_k = k.size(1);
    
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(unpacked_head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(num_heads == num_heads_k, "We do not support MQA/GQA yet");

    TORCH_CHECK(unpacked_head_size == 64 || unpacked_head_size == 128 || unpacked_head_size == 256, "Only support head size 64, 128, and 256 for now");

    CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
    CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
    CHECK_SHAPE(v, batch_size, num_heads_k, unpacked_head_size, seqlen_k/2);
    // CHECK_SHAPE(delta_s, batch_size, num_heads, seqlen_q / 128, seqlen_k);
    // CHECK_SHAPE(sfq, batch_size, seqlen_q, num_heads, unpacked_head_size);
    // CHECK_SHAPE(sfk, batch_size, seqlen_k, num_heads_k, unpacked_head_size);
    // CHECK_SHAPE(sfv, batch_size, unpacked_head_size, num_heads_k, seqlen_k);
    TORCH_CHECK(unpacked_head_size % 8 == 0, "head_size must be a multiple of 8");
    
    auto dtype = is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
    at::Tensor out = torch::empty({batch_size, num_heads, seqlen_q, unpacked_head_size}, opts.dtype(dtype));

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // const int head_size = round_multiple(head_size_og, 8);
    // const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    at::Tensor p8_workspace;
    if (v8 != nullptr) {
        const int num_sm = dprops->multiProcessorCount;
        p8_workspace = torch::empty({num_sm, 128, 128},
                                    opts.dtype(torch::kUInt8));
    }
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k, unpadded_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     unpacked_head_size, unpacked_head_size,
                     q, k, v, delta_s, out, 
                     sfq, sfk, sfv,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/-1,
                     /*window_size_right=*/is_causal ? 0 : -1,
                     per_block_mean,
                     is_bf16
                    );
    if (v8 != nullptr) {
        TORCH_CHECK(v8_scale != nullptr && v8_route != nullptr,
                    "mixed V requires scales and routes");
        TORCH_CHECK(v8->is_cuda() && v8_scale->is_cuda() && v8_route->is_cuda(),
                    "mixed tensors must be CUDA tensors");
        TORCH_CHECK(v8->get_device() == q.get_device() &&
                    v8_scale->get_device() == q.get_device() &&
                    v8_route->get_device() == q.get_device(),
                    "mixed tensors must be on the same device as Q");
        TORCH_CHECK(v8->is_contiguous() && v8_scale->is_contiguous() &&
                    v8_route->is_contiguous(), "mixed tensors must be contiguous");
        TORCH_CHECK(v8->dtype() == torch::kFloat8_e4m3fn &&
                    v8_scale->dtype() == torch::kFloat32 &&
                    v8_route->dtype() == torch::kUInt8,
                    "expected E4M3 V, float32 scales, uint8 routes");
        TORCH_CHECK(v8->sizes() == torch::IntArrayRef({batch_size, num_heads, unpacked_head_size, seqlen_k}),
                    "invalid mixed V shape");
        TORCH_CHECK(v8_scale->sizes() == torch::IntArrayRef({batch_size, num_heads, unpacked_head_size, seqlen_k / 128}),
                    "invalid mixed V scale shape");
        TORCH_CHECK(v8_route->sizes() == torch::IntArrayRef({batch_size, num_heads, seqlen_k / 128}),
                    "invalid mixed V route shape");
        params.v8_ptr = v8->data_ptr();
        params.v8_scale_ptr = v8_scale->data_ptr<float>();
        params.v8_route_ptr = v8_route->data_ptr<uint8_t>();
        params.p8_workspace_ptr = p8_workspace.data_ptr<uint8_t>();
    }
    // TODO: 132 sm count?
    auto tile_count_semaphore = is_causal ? torch::full({1}, 132, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // at::Tensor out_padded = out;
    // if (head_size_og % 8 != 0) {
    //     out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     if (out_.has_value()) { out_.value().copy_(out); }
    // }

    // return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
    // cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // printf("%s\n", cudaGetErrorString(err));
    if (v8 != nullptr) {
        return {out, softmax_lse, p8_workspace};
    }
    return {out, softmax_lse};
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &sfq, const at::Tensor &sfk, const at::Tensor &sfv,
    const at::Tensor &delta_s, int unpadded_k, c10::optional<at::Tensor> &out_,
    float softmax_scale, bool is_causal, bool per_block_mean, bool is_bf16) {
    return mha_fwd_impl(q, k, v, sfq, sfk, sfv, delta_s, unpadded_k,
                        out_, softmax_scale, is_causal, per_block_mean,
                        is_bf16, nullptr, nullptr, nullptr);
}

std::vector<at::Tensor> mha_fwd_mixed(
    at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &sfq, const at::Tensor &sfk, const at::Tensor &sfv,
    const at::Tensor &delta_s, int unpadded_k, c10::optional<at::Tensor> &out_,
    float softmax_scale, bool is_causal, bool per_block_mean, bool is_bf16,
    const at::Tensor &v8, const at::Tensor &v8_scale,
    const at::Tensor &v8_route) {
    return mha_fwd_impl(q, k, v, sfq, sfk, sfv, delta_s, unpadded_k,
                        out_, softmax_scale, is_causal, per_block_mean,
                        is_bf16, &v8, &v8_scale, &v8_route);
}



void print_e114_pv_layouts() {
    using Traits = Flash_fwd_kernel_traits<128, 128, 128, 3, 1, true>;
    typename Traits::TiledMmaPV mma4;
    typename Traits::TiledMmaPV8 mma8;
    printf("E114 FP4 PV thread layout: ");
    cute::print(mma4.get_thr_layout_vmnk());
    printf("\nE114 FP8 PV thread layout: ");
    cute::print(mma8.get_thr_layout_vmnk());
    printf("\nE114 FP4 output fragment: ");
    cute::print(cute::partition_fragment_C(mma4, cute::make_shape(cute::_128{}, cute::_128{})).layout());
    printf("\nE114 FP8 output fragment: ");
    cute::print(cute::partition_fragment_C(mma8, cute::make_shape(cute::_128{}, cute::_128{})).layout());
    auto identity = cute::make_identity_tensor(cute::make_shape(cute::_128{}, cute::_128{}));
    auto coordinates4 = mma4.get_thread_slice(0).partition_C(identity);
    auto coordinates8 = mma8.get_thread_slice(0).partition_C(identity);
    auto p_identity = cute::make_identity_tensor(cute::make_shape(cute::_128{}, cute::_128{}));
    auto p_coordinates4 = mma4.get_thread_slice(0).partition_A(p_identity);
    auto p_coordinates8 = mma8.get_thread_slice(0).partition_A(p_identity);
    printf("\nE114 FP4 thread-0 C coordinates: ");
    for (int i = 0; i < cute::size(coordinates4); ++i) {
        cute::print(coordinates4(i));
        printf(" ");
    }
    printf("\nE114 FP8 thread-0 C coordinates: ");
    for (int i = 0; i < cute::size(coordinates8); ++i) {
        cute::print(coordinates8(i));
        printf(" ");
    }
    printf("\nE114 FP4 thread-0 P coordinates: ");
    for (int i = 0; i < cute::size(p_coordinates4); ++i) {
        cute::print(p_coordinates4(i));
        printf(" ");
    }
    printf("\nE114 FP8 thread-0 P coordinates: ");
    for (int i = 0; i < cute::size(p_coordinates8); ++i) {
        cute::print(p_coordinates8(i));
        printf(" ");
    }
    typename Traits::TiledMmaQK mma_qk;
    for (int tid : {0, 1, 2, 3, 4, 8, 16, 32}) {
        auto qk_coords = mma_qk.get_thread_slice(tid).partition_C(identity);
        printf("\nE114 QK thread-%d first-32 C coordinates: ", tid);
        for (int i = 0; i < 32 && i < cute::size(qk_coords); ++i) {
            cute::print(qk_coords(i));
            printf(" ");
        }
    }
    printf("\n");
}

__global__ void e114_debug_fp8_gemm_kernel(
    uint8_t const* p,
    uint8_t const* v,
    float* out, float* p_seen, float* v_seen) {
    using Traits = Flash_fwd_kernel_traits<128, 128, 128, 3, 1, true>;
    typename Traits::TiledMmaPV8 tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto shape = cute::make_shape(cute::Int<128>{}, cute::Int<128>{});
    auto layout = cute::make_layout(shape,
        cute::make_stride(cute::Int<128>{}, cute::_1{}));
    auto gP = cute::make_tensor(cute::make_gmem_ptr(p), layout);
    auto gV = cute::make_tensor(cute::make_gmem_ptr(v), layout);
    auto coords_p = thread_mma.partition_A(cute::make_identity_tensor(shape));
    auto coords_v = thread_mma.partition_B(cute::make_identity_tensor(shape));
    auto part_p = thread_mma.partition_A(gP);
    auto part_v = thread_mma.partition_B(gV);
    auto frag_p = thread_mma.partition_fragment_A(gP);
    auto frag_v = thread_mma.partition_fragment_B(gV);
    cute::copy(part_p, frag_p);
    cute::copy(part_v, frag_v);
    for (int i = 0; i < cute::size(frag_p); ++i) {
        auto coord = coords_p(i);
        p_seen[int(cute::get<0>(coord)) * 128 + int(cute::get<1>(coord))] = float(frag_p(i));
    }
    for (int i = 0; i < cute::size(frag_v); ++i) {
        auto coord = coords_v(i);
        v_seen[int(cute::get<0>(coord)) * 128 + int(cute::get<1>(coord))] = float(frag_v(i));
    }
    auto frag_out = cute::partition_fragment_C(tiled_mma, shape);
    cute::clear(frag_out);
    cute::gemm(tiled_mma, frag_p, frag_v, frag_out);
    auto coords_out = thread_mma.partition_C(cute::make_identity_tensor(shape));
    for (int i = 0; i < cute::size(frag_out); ++i) {
        auto coord = coords_out(i);
        out[int(cute::get<0>(coord)) * 128 + int(cute::get<1>(coord))] = frag_out(i);
    }
}

std::vector<at::Tensor> e114_debug_fp8_gemm(const at::Tensor& p, const at::Tensor& v) {
    TORCH_CHECK(p.is_cuda() && v.is_cuda() && p.is_contiguous() && v.is_contiguous(),
                "expected contiguous CUDA inputs");
    TORCH_CHECK(p.dtype() == torch::kFloat8_e4m3fn &&
                v.dtype() == torch::kFloat8_e4m3fn,
                "expected E4M3 inputs");
    TORCH_CHECK(p.sizes() == torch::IntArrayRef({128, 128}) &&
                v.sizes() == torch::IntArrayRef({128, 128}),
                "expected 128x128 inputs");
    at::cuda::CUDAGuard device_guard{(char)p.get_device()};
    auto out = torch::empty({128, 128}, p.options().dtype(torch::kFloat32));
    auto p_seen = torch::empty_like(out);
    auto v_seen = torch::empty_like(out);
    e114_debug_fp8_gemm_kernel<<<1, 256, 0,
        at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<uint8_t const*>(p.data_ptr()),
        reinterpret_cast<uint8_t const*>(v.data_ptr()),
        out.data_ptr<float>(), p_seen.data_ptr<float>(), v_seen.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, p_seen, v_seen};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("fwd_mixed", &mha_fwd_mixed, "Forward pass with FP8 V-tile routing");
    m.def("print_e114_pv_layouts", &print_e114_pv_layouts,
          "Print native FP4/FP8 PV MMA fragment layouts");
    m.def("e114_debug_fp8_gemm", &e114_debug_fp8_gemm,
          "Debug native FP8 128x128 GEMM against PyTorch");
}
