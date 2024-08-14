// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <array>
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>

namespace turbomind::gemm {

struct KernelMetric {
    int64_t mio_cost;
    int64_t mma_cost;
};

class Kernel {
public:
    virtual ~Kernel() = default;

    virtual int Launch(const Operation&    operation,
                       float               alpha,
                       const void*         A,
                       const MatrixLayout& Adesc,
                       const void*         U,
                       const MatrixLayout& Udesc,
                       const void*         B,
                       const MatrixLayout& Bdesc,
                       const void*         V,
                       const MatrixLayout& Vdesc,
                       float               beta,
                       const void*         C,
                       const MatrixLayout& Cdesc,
                       void*               D,
                       const MatrixLayout& Ddesc,
                       int                 swizzle,
                       int                 splits,
                       Workspace&          workspace,
                       cudaStream_t        stream) = 0;

    // virtual because different implementation may have different workspace requeirements
    virtual int GetMaxSplits(int m, int n, int k, size_t barrier_size, size_t partials_size) = 0;

    // true if this kernel can be used to compute the gemm
    bool is_feasible(const GemmDesc& desc) const noexcept;

    std::vector<std::pair<int, KernelMetric>>
    Estimate_v2(std::array<int, 3> size, int max_splits, int max_waves, int sm_count) const;

    virtual int GetSwizzle(int m, int n, int k, int splits, int swizzle) = 0;

    const KernelDesc& desc() const noexcept
    {
        return desc_;
    }

    int3 cta_tile_size() const noexcept
    {
        return desc_.cta_tile;
    }

    int3 warp_tile_size() const noexcept
    {
        return desc_.mma_tile;
    }

    int chunk_size_k() const noexcept
    {
        return chunk_size_k_;
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

    std::string name_;
};

struct ClusteringParam {
    bool cache_policy;
    bool max_active_ctas;
};

std::vector<std::vector<LaunchSpec>> Cluster(const std::vector<LaunchSpec>& specs, const ClusteringParam& param);

}  // namespace turbomind::gemm
