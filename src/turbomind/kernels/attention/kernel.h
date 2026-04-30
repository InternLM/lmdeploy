// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/desc.h"

namespace turbomind::attention {

class Kernel {
public:
    Kernel(): desc_{}, info_{} {}

    virtual ~Kernel() = default;

    [[nodiscard]] virtual cudaError_t Launch(const void* params, int sm_count) const = 0;

    const KernelDesc& desc() const noexcept
    {
        return desc_;
    }

    const KernelInfo& info() const noexcept
    {
        return info_;
    }

    int arch() const noexcept
    {
        return desc_.arch;
    }

    int smem_size() const noexcept
    {
        return info_.attr.sharedSizeBytes + info_.dynamic_smem_size;
    }

    const std::string& name() const
    {
        return info_.name;
    }

protected:
    KernelDesc desc_;
    KernelInfo info_;
};

}  // namespace turbomind::attention
