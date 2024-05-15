#include "src/turbomind/kernels/gemm/kernel.h"
#include <iostream>

namespace turbomind::gemm {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop):
        device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
    {
        // register_sm80_f16_s4_asym_g128_basic();
        // register_sm80_f16_s4_asym_g128_extra();

        // register_sm80_f16_f16();
        // reigster_sm80_s16816gemm_f16_f16();

        register_sm70_s884gemm_f16_f16();
    }

    [[maybe_unused]] bool Add(std::unique_ptr<Kernel> kernel)
    {
        if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
            return false;
        }
        if (arch_ < kernel->arch()) {
            return false;
        }
        kernels_.push_back(std::move(kernel));
        return true;
    }

    [[nodiscard]] const std::vector<std::unique_ptr<Kernel>>& kernels() const
    {
        return kernels_;
    }

private:
    void register_sm80_f16_s4_asym_g128_basic();
    void register_sm80_f16_s4_asym_g128_extra();
    void register_sm80_f16_s4_asym_g64();
    void register_sm80_f16_s4_asym_g32();

    void register_sm80_f16_f16();

    void reigster_sm80_s16816gemm_f16_f16();
    void register_sm70_s884gemm_f16_f16();

private:
    std::shared_ptr<cudaDeviceProp>      device_prop_;
    int                                  arch_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

}  // namespace turbomind::gemm