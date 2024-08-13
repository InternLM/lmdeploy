// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include <memory>

namespace turbomind::gemm {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop);

    template<class Config>
    [[maybe_unused]] bool Add()
    {
        return Add(std::make_unique<KernelImpl<typename Config::Kernel>>());
    }

    [[nodiscard]] const std::vector<Kernel*>& kernels() const
    {
        return ptrs_;
    }

private:
    bool Add(std::unique_ptr<Kernel> kernel);

    void f16_u4g128_f16_tnt_sm70_s884();
    void f16_u4g128_f16_tnt_sm75_simt();
    void f16_u4g128_f16_tnt_sm75_s16816();
    void f16_u4g128_f16_tnt_sm80_s16816();
    void f16_u4g128_f16_tnt_sm90_s16816();

    void u4g128_f16_f16_nnn_sm80_s16816();

private:
    std::shared_ptr<cudaDeviceProp>      device_prop_;
    int                                  arch_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
    std::vector<Kernel*>                 ptrs_;
};

}  // namespace turbomind::gemm
