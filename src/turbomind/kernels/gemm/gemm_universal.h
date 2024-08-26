// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/math.h"

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

template<class PtrA, class PtrU, class PtrB, class PtrV, class Tc>
struct GemmParams {
    int m;
    int n;
    int k;

    PtrA A;
    int  lda;
    PtrU U;
    int  ldu;
    PtrB B;
    int  ldb;
    PtrV V;
    int  ldv;

    int  log_tile;
    int3 tiled_shape;

    int chunk_per_split;
    int chunk_offset;  // splits - chunk_cnt % splits

    EpilogueParam<Tc> epilogue;
};

template<class Arch_, class Mainloop, class Epilogue_, class CtaMap_>
struct GemmUniversal {

    // using Impl = typename Mainloop::Impl;
    using Impl = Mainloop;

    using Ta = typename Impl::Ta;
    using Tb = typename Impl::Tb;
    using Tu = typename Impl::Tu;
    using Tv = typename Impl::Tv;

    using Epilogue = Epilogue_;

    using Tc = typename Epilogue::Tc;

    using Arch   = Arch_;
    using CtaMap = CtaMap_;

    // col major == M-major (A)
    // row major == N-major (B)
    static constexpr Order kOrderC = Epilogue::kOrder;

    static constexpr int CTA_M = Impl::CTA_M;
    static constexpr int CTA_N = Impl::CTA_N;
    static constexpr int CTA_K = Impl::CTA_K;

    static constexpr bool SplitK = Epilogue::SplitK;

    using FragC = typename Impl::FragC;

    static constexpr int WARP_CNT = Impl::WARPS;

    using OperandA = typename Mainloop::OperandA;
    using OperandB = typename Mainloop::OperandB;
    using OperandU = typename Mainloop::OperandU;
    using OperandV = typename Mainloop::OperandV;

    static constexpr int kChunkSizeK = std::max(CTA_K, std::max(OperandU::kGroupSize, OperandV::kGroupSize));

    static constexpr int kGroupSizeU = OperandU::kGroupSize;
    static constexpr int kGroupSizeV = OperandV::kGroupSize;

    union SharedStorage {
        typename Mainloop::SharedStorage mainloop;
        typename Epilogue::SharedStorage epilogue;
    };

    static constexpr Order kOrderA = OperandA::kOrder;
    static constexpr Order kOrderB = OperandB::kOrder;
    static constexpr Order kOrderU = OperandU::kOrder;
    static constexpr Order kOrderV = OperandV::kOrder;

    static constexpr Pack kPackA = OperandA::kPack;
    static constexpr Pack kPackB = OperandB::kPack;

    using PtrA = get_pointer_type<Ta>;
    using PtrB = get_pointer_type<Tb>;
    using PtrU = get_pointer_type<Tu>;
    using PtrV = get_pointer_type<Tv>;

    using Param = GemmParams<PtrA, PtrU, PtrB, PtrV, Tc>;

    __device__ void operator()(const Param& param, const CtaMap& cta_map, char* smem_buf)
    {
        const auto tile_offset = CtaMap::get_tile_offset(param.log_tile);

        const auto& tiled_shape = param.tiled_shape;

        // Sub-optimal when the split is uneven
        //   e.g. ceil_div(10, 3) = 4 -> [4, 4, 2], however [3, 3, 4] is better in every aspect
        //   const int chunk_cnt = (param.k + kChunkSizeK - 1) / kChunkSizeK;
        // const int chunk_per_split = (chunk_cnt + tiled_shape.z - 1) / tiled_shape.z;
        // const int offset_k        = chunk_per_split * kChunkSizeK * tile_offset.z;
        // const int gemm_k_size     = std::min(offset_k + chunk_per_split * kChunkSizeK, param.k) - offset_k;

        int chunk_id    = tile_offset.z * param.chunk_per_split + max(tile_offset.z - param.chunk_offset, 0);
        int offset_k    = chunk_id * kChunkSizeK;
        int gemm_k_size = (param.chunk_per_split + int(tile_offset.z >= param.chunk_offset)) * kChunkSizeK;
        gemm_k_size     = min(offset_k + gemm_k_size, param.k) - offset_k;

        const int offset_m = tile_offset.x * CTA_M;
        const int offset_n = tile_offset.y * CTA_N;

        if (offset_m >= param.m || offset_n >= param.n || offset_k >= param.k) {  // empty tile
            return;
        }

        const int end_m = min(CTA_M, param.m - offset_m);
        const int end_n = min(CTA_N, param.n - offset_n);

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // Is 8 enough?
        __align__(8) FragC frag_C{};

        int tile_iter = (gemm_k_size + CTA_K - 1) / CTA_K;

        typename OperandA::GmemIter gmem_A{param.A, param.lda, {offset_m, offset_k}, {end_m, CTA_K}};
        typename OperandB::GmemIter gmem_B{param.B, param.ldb, {offset_n, offset_k}, {end_n, CTA_K}};

        /// TODO: move `ceil_div` into `GmemIter`
        typename OperandU::GmemIter gmem_U{
            param.U, param.ldu, {offset_m, ceil_div(offset_k, kGroupSizeU)}, {end_m, ceil_div(CTA_K, kGroupSizeU)}};
        typename OperandV::GmemIter gmem_V{
            param.V, param.ldv, {offset_n, ceil_div(offset_k, kGroupSizeV)}, {end_n, ceil_div(CTA_K, kGroupSizeV)}};

        Mainloop mainloop{};

        mainloop(gmem_A, gmem_B, gmem_U, gmem_V, frag_C, tile_iter, storage.mainloop);

        Epilogue epilogue{};

        const bool is_primary = offset_k + gemm_k_size == param.k;

        epilogue(frag_C, tile_offset, tiled_shape, end_m, end_n, is_primary, param.epilogue, storage.epilogue);
    }
};

extern __shared__ char smem_buf[];

template<class Kernel, class Params, class CtaMap>
__global__ void gemm_kernel(Params params, CtaMap cta_map)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel kernel;
        kernel(params, cta_map, smem_buf);
    }
#endif
}

}  // namespace turbomind::gemm
