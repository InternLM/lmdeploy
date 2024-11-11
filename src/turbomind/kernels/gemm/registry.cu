// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/registry.h"

namespace turbomind::gemm {

Registry::Registry(std::shared_ptr<cudaDeviceProp> device_prop):
    device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
{
    f16_u4g128_f16_tnt_sm70_s884();
    f16_u4g128_f16_tnt_sm75_simt();
    f16_u4g128_f16_tnt_sm75_s16816();
    f16_u4g128_f16_tnt_sm80_s16816();
    f16_u4g128_f16_tnt_sm90_s16816();

    sm70_s884_dynamic();
    sm75_s16816_dynamic();
    sm80_s16816_dynamic<half>();
    sm90_s16816_dynamic<half>();
    sm80_s16816_dynamic<nv_bfloat16>();
    sm90_s16816_dynamic<nv_bfloat16>();

    // u4g128_f16_f16_nnn_sm80_s16816();
}

bool Registry::Add(std::unique_ptr<Kernel> kernel)
{
    bool is_valid = true;
    if (!is_arch_compatible(kernel->arch(), arch_)) {
        is_valid = false;
    }
    if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
        is_valid = false;
    }
    // if (is_valid) {
    //     std::cout << "register: " << kernel->name()                                        //
    //               << ", shared: " << (kernel->smem_size() >> 10) << " KB"                  //
    //               << ", regs: " << kernel->desc().attr.numRegs                             //
    //               << ", local: " << (float)kernel->desc().attr.localSizeBytes << " bytes"  //
    //               << ", max_active_ctas: " << kernel->desc().max_active_ctas * is_valid << " \n";
    // }
    if (is_valid) {
        kernels_.push_back(std::move(kernel));
        ptrs_.push_back(kernels_.back().get());
    }
    return true;
}

}  // namespace turbomind::gemm
