// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace turbomind::gemm {

class Kernel {
public:
    virtual ~Kernel() = default;

    virtual int Launch(int          m,
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
                       cudaStream_t st) = 0;

    // virtual because different implemntation may have different workspace requeirements
    virtual int GetMaxSplits(int m, int n, size_t barrier_size, size_t workspace_size) = 0;

    // true if this kernel can be used to compute the gemm
    bool is_feasible(const GemmDesc& desc) const noexcept;

    std::pair<int, int64_t> FindSplitCount(int m, int n, int k, int max_split_k, int sm_count, int max_wave_count = 16);

    int3 cta_tile_size() const noexcept
    {
        return cta_tile_size_;
    }

    int3 warp_tile_size() const noexcept
    {
        return warp_tile_size_;
    }

    int chunk_size_k() const noexcept
    {
        return chunk_size_k_;
    }

    bool align_m() const noexcept
    {
        return align_m_;
    }

    bool align_n() const noexcept
    {
        return align_n_;
    }

    int stages() const noexcept
    {
        return stages_;
    }

    bool split_k() const noexcept
    {
        return split_k_;
    }

    int arch() const noexcept
    {
        return arch_;
    }

    int smem_size() const noexcept
    {
        return smem_size_;
    }

    std::string name() const
    {
        return name_;
    }

protected:
    std::string GetName() const;

    LayoutType layout_A_;
    LayoutType layout_B_;
    LayoutType layout_C_;

    DataType type_A_;
    DataType type_B_;
    DataType type_C_;

    QuantType quant_type_;

    int3 cta_tile_size_;
    int3 warp_tile_size_;
    int  chunk_size_k_;

    bool align_m_;
    bool align_n_;

    int smem_size_;

    int  stages_;
    int  swizzle_;
    bool split_k_;  // has split-k support?

    int arch_;
    int max_active_ctas_;

    std::string name_;
};

}  // namespace turbomind::gemm