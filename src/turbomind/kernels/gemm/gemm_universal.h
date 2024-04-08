// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/sync.h"

namespace turbomind::gemm {

template<class Arch_, class Mainloop, class DataIter, class CtaMap_>
struct GemmUniversal {

    using T    = typename Mainloop::T;
    using Impl = typename Mainloop::Impl;

    using Arch   = Arch_;
    using CtaMap = CtaMap_;

    static constexpr int CTA_M = Impl::CTA_M;
    static constexpr int CTA_N = Impl::CTA_N;
    static constexpr int CTA_K = Impl::CTA_K;

    using FragC = typename Impl::FragC;

    static constexpr int WARP_CNT = Impl::WARP_CNT;

    using SharedStorage = typename Mainloop::SharedStorage;

    // row.col.row
    struct Param {
        const T* A;  // x (m,k)
        const T* B;  // W (n,k)
        T*       C;  //   (m,n)
        int      m;
        int      n;
        int      k;
        int      log_tile;
    };

    __device__ void operator()(const Param& param, const CtaMap& cta_map, char* smem_buf)
    {
        const auto tile_offset = CtaMap::get_tile_offset(param.log_tile);

        const int m_idx = tile_offset.x * CTA_M;
        const int n_idx = tile_offset.y * CTA_N;
        const int k_idx = tile_offset.z * CTA_K;

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        FragC frag_C{};

        int      tile_iter = (param.k + CTA_K - 1) / CTA_K;
        DataIter data_iter{param, m_idx, n_idx, k_idx, param.k};

        typename Mainloop::GmemIterA gmem_A{param.k};
        typename Mainloop::GmemIterB gmem_B{param.k};

        Mainloop mainloop{};

        mainloop(gmem_A, gmem_B, frag_C, data_iter, tile_iter, storage);

        StoreC(frag_C, m_idx, n_idx, param);
    }

    __device__ void StoreC(FragC& frag_C, int offset_m, int offset_n, const Param& param)
    {
        Impl::StoreC(frag_C, [&](int mi, int ni, const auto& vec) {
            Store(&param.C[(offset_m + mi) * param.n + offset_n + ni], cast<half>(vec));  //
        });
    }
};

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void gemm_kernel(typename Kernel::Param params, typename Kernel::CtaMap cta_map)
{
    Kernel kernel;
    kernel(params, cta_map, smem_buf);
}

}  // namespace turbomind::gemm