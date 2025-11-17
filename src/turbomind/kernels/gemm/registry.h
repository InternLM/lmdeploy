// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include <memory>

namespace turbomind::gemm {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop);

    /// TODO: remove this
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

    void sm90_16816_4();
    void sm90_16816_8();
    void sm90_16816_16();

    void sm80_16816_4();
    void sm80_16816_8();
    void sm80_16816_16();

    void sm75_16816_4();
    void sm75_16816_8();
    void sm75_16816_16();

    void sm70_884_4();
    void sm70_884_8();
    void sm70_884_16();

    void sm90_64n32_8();

    void cublas_float();

private:
    std::shared_ptr<cudaDeviceProp>      device_prop_;
    int                                  arch_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
    std::vector<Kernel*>                 ptrs_;
};

}  // namespace turbomind::gemm
