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

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "utils.h"
#include "named_barrier.h"
namespace flash {

using namespace cute;

#ifdef SAGE3_MIXED_NOINLINE_FP8_PV
// Keep the uncommon FP8 PV path's register lifetime out of the FP4 loop.
// Route checks stay in the caller, so the all-FP4 path never makes a call.
template <typename Fn, typename Scores, typename Output>
__device__ __noinline__ bool call_fp8_pv_noinline(
    Fn const& fn, Scores& scores, Output& output, int key_tile, bool first_tile) {
    return fn(scores, output, key_tile, first_tile);
}
#endif

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    // using TMAElement = Element;
    // using TMAElementSF = typename Ktraits::ElementSF;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int BlockMean = Ktraits::BlockMean;
    using GmemTiledCopy = typename Ktraits::GmemTiledCopy;
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
    using SmemLayoutDS = typename Ktraits::SmemLayoutDS;
    using SmemLayoutAtomDS = typename Ktraits::SmemLayoutAtomDS;
    using LayoutDS = decltype(
        blocked_product(
            SmemLayoutAtomDS{}, 
            make_layout(
            make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
            make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))
        )
        );
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeSF = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d // 16, head, batch)
    using LayoutSF = typename Ktraits::LayoutSF;
    using LayoutP = typename Ktraits::LayoutP;
    using LayoutSFP = typename Ktraits::LayoutSFP;
    using SfAtom = typename Ktraits::SfAtom;
    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));

    using TMA_KV = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        _1{})); 
    
    using TMA_Vt = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutVt{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{})); 
    
    using TMA_DS = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), LayoutDS{}),
        take<0, 2>(SmemLayoutDS{}),
        make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));
    
    using BlkScaledConfig = typename Ktraits::BlkScaledConfig;
    using GmemTiledCopySF = typename Ktraits::GmemTiledCopySF;
    using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
    using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
    using SmemLayoutSFV = typename Ktraits::SmemLayoutSFV;
    using SmemLayoutSFVt = typename Ktraits::SmemLayoutSFVt;

    using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySF{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
        SmemLayoutSFQ{},
        make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
        _1{}));  // No programmatic multicast


    using TMA_SFKV = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySF{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
        SmemLayoutSFK{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
        _1{}));

    using TMA_SFVt = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySF{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
        SmemLayoutSFVt{}(_,_,cute::Int<0>{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));

    using SmemCopyAtomQ = typename Ktraits::SmemCopyAtomQ;
    using SmemCopyAtomKV = typename Ktraits::SmemCopyAtomKV;
    using SmemCopyAtomSF = typename Ktraits::SmemCopyAtomSF;
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;
    using TiledMmaPV8 = typename Ktraits::TiledMmaPV8;
    static constexpr int NumMmaThreads = size(TiledMmaQK{});
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize((SmemLayoutSFQ{})) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(size((SmemLayoutQ{})) * sizeof_bits<Element>::value));
    
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFK{})) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutDS{})) * cute::sizeof_bits_v<float>) +
        cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutK{})) * sizeof_bits<Element>::value));
    
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFVt{})) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutVt{})) * sizeof_bits<Element>::value));

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        ShapeQKV const unpadded_shape_K;
        Element const* ptr_Vt;
        ShapeQKV const shape_Vt;
        StrideQKV const stride_Vt;
        ElementSF const* ptr_SFQ{nullptr};
        ShapeSF const shape_SFQ{};
        ElementSF const* ptr_SFK{nullptr};
        ShapeSF const shape_SFK{};
        ElementSF const* ptr_SFVt{nullptr};
        ShapeSF const shape_SFVt{};
        float const* ptr_ds;
        ShapeQKV const shape_ds;
        StrideQKV const stride_ds;
        float const softmax_scale_log2;
        uint8_t const* ptr_v8;
        float const* ptr_v8_scale;
        uint8_t const* ptr_v8_route;
        uint8_t* ptr_p8_workspace;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        LayoutSF const layout_SFQ;
        ShapeQKV const shape_K;
        ShapeQKV const unpadded_shape_K;
        LayoutSF const layout_SFK;
        ShapeQKV const shape_Vt;
        LayoutSF const layout_SFVt;
        LayoutDS const layout_DS;
        TMA_Q tma_load_Q;
        TMA_SFQ tma_load_SFQ;
        TMA_KV tma_load_K;
        TMA_SFKV tma_load_SFK;
        TMA_Vt tma_load_Vt;
        TMA_SFVt tma_load_SFVt;
        TMA_DS tma_load_DS;
        float const softmax_scale_log2;
        uint8_t const* ptr_v8;
        float const* ptr_v8_scale;
        uint8_t const* ptr_v8_route;
        uint8_t* ptr_p8_workspace;
    };


    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopy{},
            mQ,
            SmemLayoutQ{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_KV tma_load_K = make_tma_copy(
            GmemTiledCopy{},
            mK,
            SmemLayoutK{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            _1{}); // mcast along M mode for this N load, if any
        Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_Vt), args.shape_Vt, args.stride_Vt);
        TMA_Vt tma_load_Vt = make_tma_copy(
            GmemTiledCopy{},
            mVt,
            SmemLayoutVt{}(_, _, _0{}),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{}); // mcast along M mode for this N load, if any
        auto [Seqlen_Q, Seqlen_K, HeadNum, Batch] = args.shape_ds;
        LayoutDS layout_ds = tile_to_shape(SmemLayoutAtomDS{}, make_shape(Seqlen_Q, Seqlen_K, HeadNum, Batch), Step<_2,_1,_3,_4>{});
        Tensor mDS = make_tensor(make_gmem_ptr(args.ptr_ds), layout_ds);
        TMA_DS tma_load_ds = make_tma_copy (
            GmemTiledCopy{},
            mDS,
            SmemLayoutDS{}(_, _, _0{}),
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{});
        LayoutSF layout_sfq = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFQ);
        Tensor mSFQ = make_tensor(make_gmem_ptr(args.ptr_SFQ), layout_sfq);
        TMA_SFQ tma_load_sfq = make_tma_copy<uint16_t>(
            GmemTiledCopySF{},
            mSFQ,
            SmemLayoutSFQ{},
            make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{});
        LayoutSF layout_sfk = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFK);
        Tensor mSFK = make_tensor(make_gmem_ptr(args.ptr_SFK), layout_sfk);
        TMA_SFKV tma_load_sfk = make_tma_copy<uint16_t>(
            GmemTiledCopySF{},
            mSFK,
            SmemLayoutSFK{}(_, _, _0{}),
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{});
        LayoutSF layout_sfvt = BlkScaledConfig::tile_atom_to_shape_SFVt(args.shape_SFVt);
        Tensor mSFVt = make_tensor(make_gmem_ptr(args.ptr_SFVt), layout_sfvt);
        TMA_SFVt tma_load_sfvt = make_tma_copy<uint16_t>(
            GmemTiledCopySF{},
            mSFVt,
            SmemLayoutSFVt{}(_, _, _0{}),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{});
        return {args.shape_Q, layout_sfq,
                args.shape_K, args.unpadded_shape_K, layout_sfk,
                args.shape_Vt, layout_sfvt,
                layout_ds,
                tma_load_Q, tma_load_sfq,
                tma_load_K, tma_load_sfk,
                tma_load_Vt, tma_load_sfvt,
                tma_load_ds, 
                args.softmax_scale_log2,
                args.ptr_v8, args.ptr_v8_scale, args.ptr_v8_route,
                args.ptr_p8_workspace};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Vt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFVt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_DS.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_n_block_max(Params const& mainloop_params, int m_block) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        }
        return n_block_max;
    }

    template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE constexpr
    auto
    thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
      CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
  
      using AtomShape_MNK  = typename Atom::Shape_MNK;
      using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
  
      auto permutation_mnk = TiledPerm{};
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  
      // Reorder the tensor for the TiledAtom
      auto t_tile = make_tile(get<0>(permutation_mnk),
                              get<2>(permutation_mnk));
      auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)
  
      // Tile the tensor for the Atom
      auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                              make_layout(size<2>(AtomShape_MNK{})));
      auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))
  
      // Transform the Atom mode from (M,K) to (Thr,Val)
      auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))
  
      // Tile the tensor for the Thread
      auto thr_tile = make_tile(_,
                                make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                          make_layout(size<3>(thr_layout_vmnk))));
      auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
  
      return thr_tensor;
    }
  
    template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE constexpr
    auto
    thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
      CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
  
      using AtomShape_MNK  = typename Atom::Shape_MNK;
      using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
  
      auto permutation_mnk = TiledPerm{};
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  
      // Reorder the tensor for the TiledAtom
      auto t_tile = make_tile(get<1>(permutation_mnk),
                              get<2>(permutation_mnk));
      auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)
  
      // Tile the tensor for the Atom
      auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                              make_layout(size<2>(AtomShape_MNK{})));
      auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))
  
      // Transform the Atom mode from (M,K) to (Thr,Val)
      auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))
  
      // Tile the tensor for the Thread
      auto thr_tile = make_tile(_,
                                make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                          make_layout(size<3>(thr_layout_vmnk))));
      auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
      return thr_tensor;
    }

    template <class SFATensor, class ThrMma>
    CUTE_HOST_DEVICE constexpr
    auto
    partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
    {
      using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
      auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
      auto thr_vmnk = thread_mma.thr_vmnk_;
      auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
      auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
      return make_fragment_like<ValTypeSF>(partition_SFA);
    }
  
    template <class SFBTensor, class ThrMma>
    CUTE_HOST_DEVICE constexpr
    auto
    partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
    {
      using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
      auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
      auto thr_vmnk = thread_mma.thr_vmnk_;
      auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
      auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
      return make_fragment_like<ValTypeSF>(partition_SFB);
    }

    template<class TiledMma>
    CUTE_HOST_DEVICE constexpr
    auto
    get_layoutSFA_TV(TiledMma& mma)
    {
      // (M,K) -> (M,K)
      auto tile_shape_mnk = tile_shape(mma);
      auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  
      // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
      auto atile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                  make_stride(               Int<1>{} ,                Int<0>{} )),
                                      _));
  
      // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
      auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
      // (thr_idx,val) -> (M,K)
      return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
    }
  
    template<class TiledMma>
    CUTE_HOST_DEVICE constexpr
    auto
    get_layoutSFB_TV(TiledMma& mma)
    {
      // (N,K) -> (N,K)
      auto tile_shape_mnk = tile_shape(mma);
      auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  
      // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
      auto btile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                  make_stride(               Int<0>{} ,                Int<1>{} )),
                                      _));
  
      // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
      auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
      // (thr_idx,val) -> (M,K)
      return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
    }

    template <typename SchedulerParams, typename SharedStorage, typename WorkTileInfo>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         SchedulerParams const& scheduler_params,
         MainloopPipelineQ pipeline_q,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineStateQ& smem_pipe_write_q,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage &shared_storage,
         WorkTileInfo work_tile_info,
         int& work_idx,
         int& tile_count_semaphore
         ) {

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);

        int n_block_max = get_n_block_max(mainloop_params, m_block);

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
        Tensor mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);
        Tensor mDS = mainloop_params.tma_load_DS.get_tma_tensor(shape(mainloop_params.layout_DS));
        Tensor mSFQ = mainloop_params.tma_load_SFQ.get_tma_tensor(shape(mainloop_params.layout_SFQ));
        Tensor mSFK = mainloop_params.tma_load_SFK.get_tma_tensor(shape(mainloop_params.layout_SFK));
        Tensor mSFVt = mainloop_params.tma_load_SFVt.get_tma_tensor(shape(mainloop_params.layout_SFVt));
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVt = local_tile(mVt(_, _, bidh, bidb), make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));  // (N, K, _)
        Tensor gDS = [&] {
                        if constexpr (BlockMean) {
                            return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}), make_coord(m_block, _));
                        } else {
                            return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}), make_coord(_0{}, _));
                        }
                    }();
        Tensor gSFQ = local_tile(mSFQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gSFK = local_tile(mSFK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gSFVt = local_tile(mSFVt(_, _, bidh, bidb), make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));
        auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = block_tma_q.partition_S(gQ);
        Tensor tQsQ = block_tma_q.partition_D(sQ);
        auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});
        Tensor tQgSFQ = block_tma_sfq.partition_S(gSFQ);
        Tensor tQsSFQ = block_tma_sfq.partition_D(sSFQ);
        auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));
        auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
        Tensor tKgSFK = group_modes<0, 3>(block_tma_sfk.partition_S(gSFK));
        Tensor tKsSFK = group_modes<0, 3>(block_tma_sfk.partition_D(sSFK));
        auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
        Tensor tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));
        Tensor tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));
        auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
        Tensor tVgSFVt = group_modes<0, 3>(block_tma_sfvt.partition_S(gSFVt));
        Tensor tVsSFVt = group_modes<0, 3>(block_tma_sfvt.partition_D(sSFVt));
        auto block_tma_ds = mainloop_params.tma_load_DS.get_slice(cluster_local_block_id.x);
        Tensor tDSgDS = group_modes<0, 3>(block_tma_ds.partition_S(gDS));
        Tensor tDSsDS = group_modes<0, 3>(block_tma_ds.partition_D(sDS));
        uint16_t mcast_mask_kv = 0;

        int n_block = n_block_max - 1;
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
        pipeline_q.producer_acquire(smem_pipe_write_q);
        copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgQ, tQsQ);
        copy(mainloop_params.tma_load_SFQ.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgSFQ, tQsSFQ);
        ++smem_pipe_write_q;
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
            tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
            tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
            tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
            tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
        copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
            tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
        }

        n_block--;
        if (lane_predicate) {
            // CUTLASS_PRAGMA_NO_UNROLL
            #pragma unroll 2
            for (; n_block >= 0; --n_block) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
                copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
                copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
                copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
        ++work_idx;
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineQ pipeline_q,
              MainloopPipeline pipeline_k, 
              MainloopPipeline pipeline_v,
              PipelineStateQ& smem_pipe_write_q,
              PipelineState& smem_pipe_write_k, 
              PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
          pipeline_q.producer_tail(smem_pipe_write_q);
          pipeline_k.producer_tail(smem_pipe_write_k);
          pipeline_v.producer_tail(smem_pipe_write_v);
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipelineQ pipeline_q,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineStateQ& smem_pipe_read_q,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO_store,
        SoftmaxFused& softmax_fused,
        int n_block_count,
        int thread_idx,
        int work_idx,
        int m_block,
        int head_id,
        int batch_id,
        SharedStorage& shared_storage
        ) {

        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockK = get<2>(TileShape_MNK{});
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
        Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

        Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
        Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_,_,Int<0>{}));
        Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_,_,Int<0>{}));
        Tensor tOrP = make_tensor_like<Element>(LayoutP{});
        Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);
        Tensor tSrSFK = partition_fragment_SFB(sSFK(_,_,Int<0>{}), thread_mma_qk);
        Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_,_,Int<0>{}), thread_mma_pv);
        Tensor tOrSFP = make_tensor<ElementSF>(LayoutSFP{});
        Tensor tOrSFP_flt = filter_zeros(tOrSFP);
        Tensor tSrDS = make_tensor<float>(make_shape(_8{}, _4{}), make_stride(_1{}, _8{}));
        // copy qk and sf from smem to rmem
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        Tensor tSsK = smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
        Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_pv);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
        Tensor tOsVt = smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sVt));
        Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt); 

        auto tile_shape_mnk = tile_shape(tiled_mma_qk);
        auto smem_tiled_copy_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{}, 
                                                        get_layoutSFA_TV(tiled_mma_qk),
                                                        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                        );
        auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(thread_idx);
        Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ));
        Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

        auto smem_tiled_copy_SFK = make_tiled_copy_impl(SmemCopyAtomSF{}, 
                                                        get_layoutSFB_TV(tiled_mma_qk),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                        );
        auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(thread_idx);
        Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
        Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

        auto smem_tiled_copy_SFV = make_tiled_copy_impl(SmemCopyAtomSF{}, 
                                                        get_layoutSFB_TV(tiled_mma_pv),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                        );
        auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(thread_idx);
        Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(as_position_independent_swizzle_tensor(sSFVt));
        Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int const unpadded_seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
        int n_block = n_block_count - 1;

        auto copy_k_block = [&](auto block_id) {
            auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
            auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());
            copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
        };
        
        auto copy_v_block = [&](auto block_id) {
            auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
            auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());
            copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
        };
        // auto gemm_qk = [&](auto block_id) {
        //     cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, block_id), tSrSFQ(_, _, block_id)), make_zip_tensor(tSrK(_, _, block_id), tSrSFK(_, _, block_id)), tSrS);
        // };
        // auto gemm_pv = [&](auto block_id) {
        //     cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, block_id), tOrSFP(_, _, block_id)), make_zip_tensor(tOrVt(_, _, block_id), tOrSFVt(_, _, block_id)), tOrO);
        // };
        auto add_delta_s = [&](auto& acc) {
            auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
            auto acc_float4 = recast<float4>(acc);
            int quad_id = (threadIdx.x % 4) * 2;
            for (int i = 0; i < 4; i++) {
                auto num = quad_id + i * 8;
                float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
                float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
                acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
                acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
                acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
                acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
            }
        };
        consumer_wait(pipeline_q, smem_pipe_read_q);
        copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);
        pipeline_q.consumer_release(smem_pipe_read_q);
        ++smem_pipe_read_q;

        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
        Tensor tSrS_converion_view = make_tensor(tSrS.data(), flash::convert_to_conversion_layout(tSrS.layout()));
        Tensor AbsMaxP = make_tensor_like<float>(
            make_layout(shape(group<1, 4>(flatten(tSrS_converion_view.layout()(make_coord(_0{}, _), _, _)))))
        );
        consumer_wait(pipeline_k, smem_pipe_read_k);
        copy_k_block(_0{});
        add_delta_s(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
            cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)), 
                                    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS);
            if (k_block < size<2>(tSrQ) - 1) {
                copy_k_block(k_block + 1);
            } else {
                pipeline_k.consumer_release(smem_pipe_read_k);
                ++smem_pipe_read_k;
            }
        }
        
         
        auto col_limit_causal = [&](int row, int n_block) {
            return row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };
        {
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = thread_mma_qk.partition_C(cS);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    // K's FP4 pack permutes token rows within each 32-key group.
                    // MMA columns address that physical order, while the unpadded
                    // length is a logical-token boundary.
                    int physical_col = int(get<1>(tScS(i)));
                    int local_col = physical_col % 32;
                    int logical_col = (physical_col / 32) * 32 +
                                      (local_col / 8) * 2 +
                                      ((local_col % 8) / 2) * 8 + local_col % 2;
                    if (logical_col >= int(unpadded_seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                } else { 
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN,
                                                        col_limit_causal(int(get<0>(tScS(i))), n_block))) {
                        tSrS(i) = -INFINITY;
                    }
                }
            }
        }
        auto quantize = [&](auto mma_k, auto acc_conversion_view) {
            Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
            Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);
            Tensor SFP = make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
            Tensor SFP_uint32_view = recast<uint32_t>(SFP);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
                uint32_t& tmp = SFP_uint32_view(i / 4);
                flash::packed_float_to_ue4m3(
                    AbsMaxP_stagek(i), 
                    AbsMaxP_stagek(i + 1), 
                    AbsMaxP_stagek(i + 2), 
                    AbsMaxP_stagek(i + 3), 
                    tmp
                );
            }
            int const quad_id = threadIdx.x & 3;
            uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);
            Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
            Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));
        
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        flash::packed_float_to_e2m1(
                            acc_conversion_stagek(make_coord(_0{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_1{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_2{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_3{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_4{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_5{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_6{}, i), mma_m),
                            acc_conversion_stagek(make_coord(_7{}, i), mma_m),
                            tOrP_uint32_view(i, mma_m)
                        );
                    }
                    uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
                    uint32_t peer_sfp  = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
                    if ((quad_id & 1) == 0) {
                        uint32_t sfp = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
                        tOrSFP_uint32_view(_0{}, mma_m) = sfp;
                    } else {
                        uint32_t sfp = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
                        tOrSFP_uint32_view(_0{}, mma_m) = sfp;
                    }
            }
        };

        // E114 first native mixed-PV path. QK and the default PV route keep
        // Sage3's original NVFP4 MMAs. The selected route builds one E4M3 P
        // tile in CTA-private global scratch, then performs
        // an SM120 E4M3 MMA against the prepacked E4M3 V tile. The explicit
        // register permutation below maps the FP8 MMA C fragment back to the
        // original Sage3 FP4 C fragment before online-softmax accumulation.
        auto run_fp8_pv = [&](auto& scores, auto& output, int key_tile, bool first_tile) {
            if (mainloop_params.ptr_v8_route == nullptr) { return false; }
            int const head_count = get<2>(mainloop_params.shape_Q);
            int const key_tiles = seqlen_k / kBlockN;
            int const route_index = (batch_id * head_count + head_id) * key_tiles + key_tile;
            if (mainloop_params.ptr_v8_route[route_index] == 0) { return false; }

            Tensor score_reduction = make_tensor(scores.data(), flash::convert_to_reduction_layout(scores.layout()));
            Tensor tile_max = make_fragment_like(softmax_fused.row_max);
            flash::reduce_max(score_reduction, tile_max);
            // QK scores for one physical row span a four-lane quad. Match
            // SoftmaxFused's warp reduction before choosing a P8 row scale;
            // a thread-local max would assign inconsistent scales to that row.
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size(tile_max); ++mi) {
                tile_max(mi) = fmaxf(tile_max(mi),
                    __shfl_xor_sync(0xffffffffu, tile_max(mi), 1));
                tile_max(mi) = fmaxf(tile_max(mi),
                    __shfl_xor_sync(0xffffffffu, tile_max(mi), 2));
            }
            Tensor p8_scale = make_fragment_like(softmax_fused.row_max);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size(tile_max); ++mi) {
                float global_max = first_tile ? tile_max(mi) : fmaxf(tile_max(mi), softmax_fused.row_max(mi));
                float local_peak = tile_max(mi) == -INFINITY ? 0.0f :
                    flash::ptx_exp2((tile_max(mi) - global_max) * mainloop_params.softmax_scale_log2);
                // Sage3's online denominator is scaled by 2688. Keep the
                // FP8 PV numerator in that same domain before finalization.
                p8_scale(mi) = fmaxf(local_peak * 6.0f, 1.0e-12f);
            }

            auto* p8_storage = reinterpret_cast<cutlass::float_e4m3_t*>(
                mainloop_params.ptr_p8_workspace +
                blockIdx.x * kBlockM * kBlockN);
            Tensor score_coords = thread_mma_qk.partition_C(
                make_identity_tensor(select<0, 1>(TileShape_MNK{})));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(scores); ++i) {
                auto coord = score_coords(i);
                int row = int(get<0>(coord));
                int col = int(get<1>(coord));
                // partition_fragment_C stores the QK accumulator in a
                // different linear order from partition_C(identity)'s
                // coordinate view. Undo the 4x4 pair-interleave before
                // materializing the row-major P8 tile consumed by FP8 PV.
                int const true_col = (col / 32) * 32 +
                    ((col % 8) / 2) * 8 + ((col % 32) / 8) * 2 + col % 2;
                int mi = (row / 8) & 1;
                float global_max = first_tile ? tile_max(mi) : fmaxf(tile_max(mi), softmax_fused.row_max(mi));
                float p = scores(i) == -INFINITY ? 0.0f :
                    flash::ptx_exp2((scores(i) - global_max) * mainloop_params.softmax_scale_log2);
                p8_storage[row * kBlockN + true_col] =
                    cutlass::float_e4m3_t(fminf(p * 2688.0f / p8_scale(mi), 448.0f));
            }
            cutlass::arch::NamedBarrier::sync(256, 8);

            if (first_tile) {
                softmax_fused.template online_softmax_with_quant</*Is_first=*/true>(
                    scores, AbsMaxP, mainloop_params.softmax_scale_log2);
            } else {
                softmax_fused.template online_softmax_with_quant</*Is_first=*/false>(
                    scores, AbsMaxP, mainloop_params.softmax_scale_log2);
            }
            consumer_wait(pipeline_v, smem_pipe_read_v);

            TiledMmaPV8 tiled_mma_pv8;
            auto thread_mma_pv8 = tiled_mma_pv8.get_thread_slice(thread_idx);
            // CuTe's SM120 F8F6F4 trait declares uint8_t register operands:
            // copy the E4M3 *encoding bytes*, not numerically convert E4M3
            // values to uint8_t (which would turn 1.0 into 0x01).
            Tensor sP8 = make_tensor(make_gmem_ptr(
                reinterpret_cast<uint8_t const*>(p8_storage)),
                make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                            make_stride(Int<kBlockN>{}, _1{})));
            Tensor tPgP8 = thread_mma_pv8.partition_A(sP8);
            Tensor p8_frag = thread_mma_pv8.partition_fragment_A(sP8);
            cute::copy(tPgP8, p8_frag);
            cutlass::arch::NamedBarrier::sync(256, 8);

            auto const* v8_ptr = mainloop_params.ptr_v8;
            int const v8_head_offset = (batch_id * head_count + head_id) * kBlockK * seqlen_k;
            Tensor gV8 = make_tensor(make_gmem_ptr(v8_ptr + v8_head_offset + key_tile * kBlockN),
                make_layout(make_shape(Int<kBlockK>{}, Int<kBlockN>{}),
                            make_stride(seqlen_k, _1{})));
            Tensor tVgV8 = thread_mma_pv8.partition_B(gV8);
            Tensor v8_frag = thread_mma_pv8.partition_fragment_B(gV8);
            cute::copy(tVgV8, v8_frag);

            auto fp8_acc_layout = partition_fragment_C(
                tiled_mma_pv8, select<0, 2>(TileShape_MNK{})).layout();
#ifdef SAGE3_MIXED_INPLACE_FP8_ACC
            // FP8 and FP4 C fragments each hold 64 floats per thread. Reuse
            // the output fragment, then permute the FP8 register order in
            // place instead of keeping both fragments alive across the MMA.
            Tensor fp8_acc = make_tensor(output.data(), fp8_acc_layout);
#else
            Tensor fp8_acc = partition_fragment_C(tiled_mma_pv8, select<0, 2>(TileShape_MNK{}));
#endif
            clear(fp8_acc);
            cute::gemm(tiled_mma_pv8, p8_frag, v8_frag, fp8_acc);
            Tensor out_coords = thread_mma_pv.partition_C(
                make_identity_tensor(select<0, 2>(TileShape_MNK{})));
            auto const* v8_scale = mainloop_params.ptr_v8_scale;
#ifdef SAGE3_MIXED_INPLACE_FP8_ACC
            // old FP8 indices for each 16-value FP4 block:
            // [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15].
            // This is four disjoint 3-cycles plus four fixed points.
            CUTLASS_PRAGMA_UNROLL
            for (int block = 0; block < size(output); block += 16) {
                float tmp = output(block + 2);
                output(block + 2) = output(block + 4);
                output(block + 4) = output(block + 8);
                output(block + 8) = tmp;
                tmp = output(block + 3);
                output(block + 3) = output(block + 5);
                output(block + 5) = output(block + 9);
                output(block + 9) = tmp;
                tmp = output(block + 6);
                output(block + 6) = output(block + 12);
                output(block + 12) = output(block + 10);
                output(block + 10) = tmp;
                tmp = output(block + 7);
                output(block + 7) = output(block + 13);
                output(block + 13) = output(block + 11);
                output(block + 11) = tmp;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(output); ++i) {
                int const row = int(get<0>(out_coords(i)));
                int const col = int(get<1>(out_coords(i)));
                int const mi = (row / 8) & 1;
                float const vs = v8_scale[((batch_id * head_count + head_id) * kBlockK + col) * key_tiles + key_tile];
                output(i) *= p8_scale(mi) * vs;
            }
#else
            clear(output);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(output); ++i) {
                int const block = i / 16;
                int const within = i % 16;
                int const fp8_i = (block * 4 + (within % 8) / 2) * 4 +
                                  (within >= 8 ? 2 : 0) + within % 2;
                int const row = int(get<0>(out_coords(i)));
                int const col = int(get<1>(out_coords(i)));
                int const mi = (row / 8) & 1;
                float const vs = v8_scale[((batch_id * head_count + head_id) * kBlockK + col) * key_tiles + key_tile];
                output(i) = fp8_acc(fp8_i) * p8_scale(mi) * vs;
            }
#endif
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;
            return true;
        };

#ifdef SAGE3_MIXED_NOINLINE_FP8_PV
        auto try_fp8_pv = [&](auto& scores, auto& output, int key_tile, bool first_tile) {
            if (mainloop_params.ptr_v8_route == nullptr) { return false; }
            int const head_count = get<2>(mainloop_params.shape_Q);
            int const key_tiles = seqlen_k / kBlockN;
            int const route_index = (batch_id * head_count + head_id) * key_tiles + key_tile;
            if (mainloop_params.ptr_v8_route[route_index] == 0) { return false; }
            return call_fp8_pv_noinline(run_fp8_pv, scores, output, key_tile, first_tile);
        };
#else
        auto& try_fp8_pv = run_fp8_pv;
#endif

        if (!try_fp8_pv(tSrS, tOrO_store, n_block, true)) {
            softmax_fused.template online_softmax_with_quant</*Is_first=*/true>(tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);

            consumer_wait(pipeline_v, smem_pipe_read_v);
            copy_v_block(_0{});
            quantize(_0{}, tSrS_converion_view);
            CUTLASS_PRAGMA_UNROLL
            for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)), 
                                        make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
                if (v_block < size<2>(tOrP) - 1) {
                    copy_v_block(v_block + 1);
                    quantize(v_block + 1, tSrS_converion_view);
                } else {
                    pipeline_v.consumer_release(smem_pipe_read_v);
                    ++smem_pipe_read_v;
                }
            }
        }
        
        n_block--;
        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
        // // Only go through these if Is_causal, since n_masking_steps = 1 when !Is_causal
        CUTLASS_PRAGMA_UNROLL
        for (int masking_step = 0; masking_step < n_masking_steps - 1 && n_block >= 0; ++masking_step, --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
            Tensor tSrS_converion_view = make_tensor(tSrS.data(), flash::convert_to_conversion_layout(tSrS.layout()));
            consumer_wait(pipeline_k, smem_pipe_read_k);
            copy_k_block(_0{});
            add_delta_s(tSrS);
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)), 
                                    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS);
                if (k_block < size<2>(tSrQ) - 1) {
                    copy_k_block(k_block + 1);
                }
            }
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K
            ++smem_pipe_read_k;
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = thread_mma_qk.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if (int(get<1>(tScS(i))) >= col_limit_causal(int(get<0>(tScS(i))), n_block - 1)) {
                    tSrS(i) = -INFINITY;
                }
            }
            Tensor tOrO = make_fragment_like(tOrO_store);
            if (!try_fp8_pv(tSrS, tOrO, n_block, false)) {
                softmax_fused.template online_softmax_with_quant</*Is_first=*/false>(tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);
                consumer_wait(pipeline_v, smem_pipe_read_v);
                copy_v_block(_0{});
                quantize(_0{}, tSrS_converion_view);
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)), 
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_converion_view);
                    }
                }
                pipeline_v.consumer_release(smem_pipe_read_v);
                ++smem_pipe_read_v;
            }
            if (masking_step > 0) { softmax_fused.rescale_o(tOrO_store, tOrO); }
        }

        #pragma unroll 1
        for (; n_block >= 0; --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
            Tensor tSrS_converion_view = make_tensor(tSrS.data(), flash::convert_to_conversion_layout(tSrS.layout()));
            consumer_wait(pipeline_k, smem_pipe_read_k);
            copy_k_block(_0{});
            add_delta_s(tSrS);
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)), 
                                    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS);
                if (k_block < size<2>(tSrQ) - 1) {
                    copy_k_block(k_block + 1);
                } else {
                    pipeline_k.consumer_release(smem_pipe_read_k);
                    ++smem_pipe_read_k;
                }
            }
            Tensor tOrO = make_fragment_like(tOrO_store);
            if (!try_fp8_pv(tSrS, tOrO, n_block, false)) {
                softmax_fused.template online_softmax_with_quant</*Is_first=*/false>(tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);
                consumer_wait(pipeline_v, smem_pipe_read_v);
                copy_v_block(_0{});
                quantize(_0{}, tSrS_converion_view);
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)), 
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_converion_view);
                    } else {
                        pipeline_v.consumer_release(smem_pipe_read_v);
                        ++smem_pipe_read_v;
                    }
                }
            }
            softmax_fused.rescale_o(tOrO_store, tOrO);
        }
        softmax_fused.finalize(tOrO_store);
        return;
    }

};

} // namespace flash

