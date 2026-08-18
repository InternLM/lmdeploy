// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/kernels/gemm/kernel.h"

namespace turbomind::gemm {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop);

    [[nodiscard]] const std::vector<Kernel*>& kernels() const
    {
        return ptrs_;
    }

private:
    bool Add(std::unique_ptr<Kernel> kernel);

    std::shared_ptr<cudaDeviceProp>      device_prop_;
    int                                  arch_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
    std::vector<Kernel*>                 ptrs_;
};

}  // namespace turbomind::gemm
