// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include <iostream>

namespace turbomind::gemm {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop);

    template<class Config>
    [[maybe_unused]] bool Add()
    {
        return Add(std::make_unique<KernelImpl<typename Config::Kernel>>());
    }

    [[nodiscard]] const std::vector<std::unique_ptr<Kernel>>& kernels() const
    {
        return kernels_;
    }

private:
    bool Add(std::unique_ptr<Kernel> kernel);

    void f16_u4g128_f16_tnt_sm70_s884();
    void f16_u4g128_f16_tnt_sm75_simt();
    void f16_u4g128_f16_tnt_sm75_s16816();
    void f16_u4g128_f16_tnt_sm80_s16816();

private:
    std::shared_ptr<cudaDeviceProp>      device_prop_;
    int                                  arch_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

}  // namespace turbomind::gemm
