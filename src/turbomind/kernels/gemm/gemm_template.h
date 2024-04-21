// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl_81616.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80.h"
#include "src/turbomind/kernels/gemm/tile_iterator.h"

namespace turbomind::gemm {

namespace detail {

template<class T>
auto cast(T* p)
{
    if constexpr (bitsof<T> % 8 == 0) {
        return p;
    }
    else {
        return (char*)p;
    }
}

}  // namespace detail

template<class T, class Tb>
void invoke(T* C, const T* A, const Tb* B, const T* Q, int m, int n, int k, cudaStream_t st)
{
    constexpr int CTA_M  = 128;
    constexpr int CTA_N  = 128;
    constexpr int CTA_K  = 32;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 32;

    // constexpr int CTA_M  = 64;
    // constexpr int CTA_N  = 64;
    // constexpr int CTA_K  = 32;
    // constexpr int WARP_M = 64;
    // constexpr int WARP_N = 64;
    // constexpr int WARP_K = 32;

    // constexpr int CTA_M  = 128;
    // constexpr int CTA_N  = 256;
    // constexpr int CTA_K  = 64;
    // constexpr int WARP_M = 64;
    // constexpr int WARP_N = 64;
    // constexpr int WARP_K = 64;

    using Tq = half2;

    using Impl   = Impl<MMA_81616, T, Tb, Tq, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 4, 1>;
    using Kernel = GemmUniversal<void, Mainloop_sm80<Impl>, CtaSwizzleMap<8>>;

    using Map = typename Kernel::CtaMap;

    auto tiles = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, 1);

    auto log_tile = Map::get_log_tile(tiles);

    auto grid = Map::get_grid_shape(tiles);

    // printf("grid = [%d %d %d]\n", (int)grid.x, (int)grid.y, (int)grid.z);
    auto block = Impl::WARP_CNT * WARP_SIZE;

    [[maybe_unused]] static const int _ = [] {
        std::cout << "A:\n";
        Print(typename Impl::ThreadMapA{});
        std::cout << "\nB:\n";
        Print(typename Impl::ThreadMapB{});
        std::cout << "\nQ:\n";
        Print(typename Impl::ThreadMapQ{});
        printf("warp count: %d\n", Impl::WARP_CNT);
        return 0;
    }();

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);
    if constexpr (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(gemm_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    gemm_kernel<Kernel><<<grid, block, kSmemSize, st>>>(
        typename Kernel::Param{(T*)A, detail::cast((Tb*)B), (Tq*)Q, C, m, n, k, log_tile}, typename Kernel::CtaMap{});
}

}  // namespace turbomind::gemm