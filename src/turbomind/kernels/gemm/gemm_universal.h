// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/math.h"

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

struct GemmParam {
    MatrixParam a;
    MatrixParam b;
    MatrixParam u;
    MatrixParam v;
};

template<class Op>
__inline__ __device__ MatrixData resolve_op(const MatrixParam& param, int gemm_id)
{
    return resolve<typename Op::Dtype, Op::GmemIter::kMode>(param, gemm_id);
}

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

    static constexpr bool kDynamicSched = is_dynamic_scheduler<CtaMap>::value;
    static constexpr bool kSplitK       = Epilogue::SplitK;

    using FragC = typename Impl::FragC;

    static constexpr int WARP_CNT = Impl::WARPS;

    using OperandA = typename Mainloop::OperandA;
    using OperandB = typename Mainloop::OperandB;
    using OperandU = typename Mainloop::OperandU;
    using OperandV = typename Mainloop::OperandV;

    static constexpr int kChunkSizeK = std::max(CTA_K, std::max(OperandU::kGroupSize, OperandV::kGroupSize));

    static constexpr int kGSizeU = OperandU::kGroupSize;
    static constexpr int kGSizeV = OperandV::kGroupSize;

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

    using Param = GemmParam;

    __device__ void operator()(const Param& param, const EpilogueParam& epi_param, CtaMap& cta_map, char* smem_buf)
    {
        if (!cta_map.init()) {
            return;
        }

        const auto [M, N, K, L] = cta_map.gemm_shape();
        const auto tile_offset  = cta_map.tile_offset();

        const auto [iter_k_beg, iter_k_end] = cta_map.iter_k_range();

        const int offset_m = tile_offset.x * CTA_M;
        const int offset_n = tile_offset.y * CTA_N;
        const int offset_k = iter_k_beg * CTA_K;

        if (offset_m >= M || offset_n >= N || offset_k >= K) {  // empty tile
            return;
        }

        const int extent_m = min(CTA_M, M - offset_m);
        const int extent_n = min(CTA_N, N - offset_n);

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // Is 8 enough?
        __align__(8) FragC frag_C{};

        // int tile_iter = (gemm_k_size + CTA_K - 1) / CTA_K;
        int tile_iter = iter_k_end - iter_k_beg;

        const int g = tile_offset.w;

        const auto mat_A = resolve_op<OperandA>(param.a, g);
        const auto mat_B = resolve_op<OperandB>(param.b, g);
        const auto mat_U = resolve_op<OperandU>(param.u, g);
        const auto mat_V = resolve_op<OperandV>(param.v, g);

        typename OperandA::GmemIter gmem_A{mat_A, {offset_m, offset_k}, {extent_m, CTA_K}};
        typename OperandB::GmemIter gmem_B{mat_B, {offset_n, offset_k}, {extent_n, CTA_K}};

        const int2 offset_U{offset_m, cdiv(offset_k, kGSizeU)}, extent_U{extent_m, cdiv(CTA_K, kGSizeU)};
        typename OperandU::GmemIter gmem_U{mat_U, offset_U, extent_U};

        const int2 offset_V{offset_n, cdiv(offset_k, kGSizeV)}, extent_V{extent_n, cdiv(CTA_K, kGSizeV)};
        typename OperandV::GmemIter gmem_V{mat_V, offset_V, extent_V};

        Mainloop mainloop{};
        mainloop(gmem_A, gmem_B, gmem_U, gmem_V, frag_C, tile_iter, storage.mainloop);

        {
            cta_map.init();

            const auto [M, N, K, L] = cta_map.gemm_shape();

            const auto tiled_shape = cta_map.tiled_shape();
            const auto tile_offset = cta_map.tile_offset();

            const int2 extents = {min(CTA_M, M - tile_offset.x * CTA_M), min(CTA_N, N - tile_offset.y * CTA_N)};

            const bool is_last = cta_map.iter_k_range().y * CTA_K == K;

            Epilogue epilogue{};
            epilogue(frag_C,  //
                     tile_offset,
                     tiled_shape,
                     extents,
                     cta_map.tile_id(),
                     is_last,
                     epi_param,
                     storage.epilogue);
        }
    }
};

extern __shared__ char smem_buf[];

template<class Kernel, class Param, class EpilogueParam, class CtaMap>
__global__ void gemm_kernel(Param param, EpilogueParam epi_param, CtaMap cta_map)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel kernel;
        kernel(param, epi_param, cta_map, smem_buf);
    }
#endif
}

}  // namespace turbomind::gemm
