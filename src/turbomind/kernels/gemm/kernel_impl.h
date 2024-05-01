// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/kernel.h"

namespace turbomind::gemm {

template<class Gemm>
class KernelImpl: public Kernel {
public:
    // import frequently used constants
    static constexpr int CTA_M = Gemm::CTA_M;
    static constexpr int CTA_N = Gemm::CTA_N;
    static constexpr int CTA_K = Gemm::CTA_K;

    using Impl = typename Gemm::Impl;

    KernelImpl()
    {
        layout_A_ = LayoutType::kRowMajor;
        layout_B_ = LayoutType::kFragment_81616;
        layout_C_ = LayoutType::kRowMajor;

        type_A_ = get_data_type_v<typename Gemm::T>;
        type_B_ = get_data_type_v<typename Gemm::Tb>;
        type_C_ = get_data_type_v<typename Gemm::T>;

        quant_type_ = QuantType::kAsym_FMA;

        cta_tile_size_  = {Gemm::CTA_M, Gemm::CTA_N, Gemm::CTA_K};
        warp_tile_size_ = {Impl::WARP_M, Impl::WARP_N, Impl::WARP_K};
        chunk_size_k_   = Gemm::kChunkSizeK;

        align_m_ = Gemm::AlignedM;
        align_n_ = Gemm::AlignedN;

        smem_size_ = sizeof(typename Gemm::SharedStorage);

        stages_  = Impl::Stages;
        split_k_ = Gemm::SplitK;
        swizzle_ = Gemm::CtaMap::N;

        arch_ = 80;

        if (smem_size_ > (48 << 10)) {
            cudaFuncSetAttribute(gemm_kernel<Gemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_ctas_, gemm_kernel<Gemm>, Impl::WARP_CNT * WARP_SIZE, smem_size_);

        name_ = GetName();
    }

    int Launch(int          m,
               int          n,
               int          k,
               const void*  A,
               int          lda,
               const void*  B,
               int          ldb,
               const void*  Q,
               int          ldq,
               float        beta,
               void*        C,
               int          ldc,
               int          splits,
               EpilogueType epilogue,
               int*         barriers,
               size_t&      barriers_size,
               void*        workspace,
               size_t&      workspace_size,
               cudaStream_t st) override
    {
        using Map = typename Gemm::CtaMap;

        const auto tiles = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, splits);

        if (splits > 1 && barriers == nullptr && workspace == nullptr) {
            GetWorkspaceSizes(m, n, tiles.x, tiles.y, splits, barriers_size, workspace_size);
            return -1;
        }

        const auto log_tile = Map::get_log_tile(tiles);

        const auto grid  = Map::get_grid_shape(tiles);
        const auto block = Gemm::Impl::WARP_CNT * WARP_SIZE;

        using Ta = typename Gemm::T;
        using Tb = typename Gemm::Tb;
        using Tq = typename Gemm::Tq;
        using Tc = typename Gemm::T;

        typename Gemm::Param param{(Ta*)A, _cast((Tb*)B), (Tq*)Q, (Tc*)C, m, n, k, log_tile, tiles, (float*)workspace, barriers};

        gemm_kernel<Gemm><<<grid, block, smem_size_, st>>>(param, Map{});

        return 0;
    }

    template<class T>
    static auto _cast(T* p)
    {
        if constexpr (bitsof<T> % 8 == 0) {
            return p;
        }
        else {
            return (char*)p;
        }
    }

    // ! This assumes n results is 16 byte aligned partials
    void
    GetWorkspaceSizes(int m, int n, int tiled_m, int tiled_n, int splits, size_t& barriers_size, size_t& workspace_size)
    {
        workspace_size = sizeof(float) * m * n * splits;
        barriers_size  = sizeof(int) * tiled_m * tiled_n * splits;
    }

    int GetMaxSplits(int m, int n, size_t barrier_size, size_t workspace_size) override
    {
        if (!Gemm::SplitK) {
            return 1;
        }
        const int tiled_m = ceil_div(m, CTA_M);
        const int tiled_n = ceil_div(m, CTA_N);
        size_t    barriers_per_split{};
        size_t    workspace_per_split{};
        GetWorkspaceSizes(m, n, tiled_m, tiled_n, 1, barriers_per_split, workspace_per_split);
        return std::min<int>(barrier_size / barriers_per_split, workspace_size / workspace_per_split);
    }
};

}  // namespace turbomind::gemm