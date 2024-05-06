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

    virtual int Launch(const Operation&    operation,
                       const void*         alpha,
                       const void*         A,
                       const MatrixLayout& Adesc,
                       const void*         B,
                       const MatrixLayout& Bdesc,
                       const void*         Q,
                       const MatrixLayout& Qdesc,
                       const void*         beta,
                       const void*         C,
                       const MatrixLayout& Cdesc,
                       void*               D,
                       const MatrixLayout& Ddesc,
                       int                 splits,
                       void*               barriers,
                       size_t&             barriers_size,
                       void*               workspace,
                       size_t&             workspace_size,
                       cudaStream_t        stream) = 0;

    // virtual because different implemntation may have different workspace requeirements
    virtual int GetMaxSplits(int m, int n, size_t barrier_size, size_t workspace_size) = 0;

    // true if this kernel can be used to compute the gemm
    bool is_feasible(const GemmDesc& desc) const noexcept;

    std::pair<int, int64_t> FindSplitCount(int m, int n, int k, int max_split_k, int sm_count, int max_wave_count = 16);

    int3 cta_tile_size() const noexcept
    {
        return desc_.cta_tile;
    }

    int3 warp_tile_size() const noexcept
    {
        return desc_.warp_tile;
    }

    int chunk_size_k() const noexcept
    {
        return chunk_size_k_;
    }

    bool align_m() const noexcept
    {
        return desc_.align_m;
    }

    bool align_n() const noexcept
    {
        return desc_.align_n;
    }

    int stages() const noexcept
    {
        return desc_.stages;
    }

    bool split_k() const noexcept
    {
        return desc_.split_k;
    }

    int arch() const noexcept
    {
        return desc_.arch;
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

    KernelDesc desc_;

    int chunk_size_k_;
    int smem_size_;
    int max_active_ctas_;

    std::string name_;
};

}  // namespace turbomind::gemm