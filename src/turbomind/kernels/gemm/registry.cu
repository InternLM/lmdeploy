// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdlib>
#include <iostream>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/registry.h"

namespace turbomind::gemm {

Registry::Registry(std::shared_ptr<cudaDeviceProp> device_prop):
    device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
{
    for (auto& register_fn : gKernelFactories()) {
        Collector collector;
        register_fn(collector, arch_);
        for (auto& k : collector.release()) {
            Add(std::move(k));
        }
    }
}

bool Registry::Add(std::unique_ptr<Kernel> kernel)
{
    bool is_valid = true;

    if (!is_arch_compatible(kernel->arch(), arch_)) {
        is_valid = false;
    }

    if (is_valid && std::getenv("TM_GEMM_DEBUG_OCCUPANCY")) {
        std::cout << "register: " << kernel->name()                                        //
                  << ", shared: " << (kernel->smem_size() >> 10) << " KB"                  //
                  << ", regs: " << kernel->info().attr.numRegs                             //
                  << ", local: " << (float)kernel->info().attr.localSizeBytes << " bytes"  //
                  << ", max_active_ctas: " << kernel->info().max_active_ctas << " \n";
    }

    if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
        is_valid = false;
    }

    if (is_valid) {
        ptrs_.push_back(kernels_.emplace_back(transpose(*kernel)).get());
        ptrs_.push_back(kernels_.emplace_back(std::move(kernel)).get());
    }

    return true;
}

}  // namespace turbomind::gemm
