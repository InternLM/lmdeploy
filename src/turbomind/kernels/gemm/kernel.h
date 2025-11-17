// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct KernelMetric {
    int64_t mio_cost;
    int64_t mma_cost;
};

class Kernel {
public:
    Kernel(): desc_{}, info_{} {}

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

    // true if this kernel can be used to compute the gemm
    virtual bool is_feasible(const GemmDesc& desc) const noexcept;

    virtual int GetMaxSwizzle(const int4& shape) const = 0;

    virtual int GetMaxSplits(const int4& shape, int swizzle, size_t bsize, size_t psize) const = 0;

    const KernelDesc& desc() const noexcept
    {
        return desc_;
    }

    const KernelInfo& info() const noexcept
    {
        return info_;
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
        return info_.chunk_size_k;
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
        return info_.attr.sharedSizeBytes + info_.dynamic_smem_size;
    }

    std::string name() const
    {
        return info_.name;
    }

protected:
    std::string GetName() const;

    KernelDesc desc_;
    KernelInfo info_;
};

struct ClusteringParam {
    bool cache_policy;
    bool max_active_ctas;
};

std::vector<std::vector<LaunchSpec>> Cluster(const std::vector<LaunchSpec>& specs, const ClusteringParam& param);

std::unique_ptr<Kernel> transpose(Kernel& kernel);

}  // namespace turbomind::gemm
