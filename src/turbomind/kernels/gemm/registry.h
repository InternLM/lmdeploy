// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include <iostream>

namespace turbomind::gemm {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop):
        device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
    {
        f16_u4g128_f16_tnt_sm70_s884();
        f16_u4g128_f16_tnt_sm75_simt();
        f16_u4g128_f16_tnt_sm75_s16816();
        f16_u4g128_f16_tnt_sm80_s16816();
    }

    [[maybe_unused]] bool Add(std::unique_ptr<Kernel> kernel)
    {
        if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
            return false;
        }
        if (arch_ < kernel->arch()) {
            return false;
        }
        std::cout << "register: " << kernel->name() << "\n";
        kernels_.push_back(std::move(kernel));
        return true;
    }

    template<class Config>
    [[maybe_unused]] bool Add()
    {
        auto kernel = std::make_unique<KernelImpl<typename Config::Kernel>>();

        if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
            return false;
        }
        if (arch_ < kernel->arch()) {
            return false;
        }
        std::cout << "register: " << kernel->name() << "\n";
        kernels_.push_back(std::move(kernel));
        return true;
    }

    [[nodiscard]] const std::vector<std::unique_ptr<Kernel>>& kernels() const
    {
        return kernels_;
    }

private:
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
