#pragma once

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <optional>

namespace turbomind::gemm {

struct PopulateParam {
    int    max_splits;
    int    max_waves;
    int    swizzle;
    size_t barriers_size;
    size_t partials_size;
};

class Context {
public:
    explicit Context(const cudaDeviceProp& prop);

    bool Init(const Operation&    operation,
              const MatrixLayout& Adesc,
              const MatrixLayout& Udesc,
              const MatrixLayout& Bdesc,
              const MatrixLayout& Vdesc,
              const MatrixLayout& Cdesc,
              const MatrixLayout& Ddesc);

    std::vector<Kernel*> Filter(const std::vector<Kernel*>& kernels) const;

    std::vector<LaunchSpec> Populate(const Kernel& kernel, const PopulateParam& param) const;

    std::vector<LaunchSpec> Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const;

    const GemmDesc& desc() const
    {
        return desc_;
    }

    const GemmDesc& get_desc(const Kernel& kernel) const
    {
        return kernel.desc().transpose ? desc_trans_ : desc_;
    }

    // Alignment
    // (align_m, align_n, align_k) -> is_aligned
    //  gcd_mnk need to be part of gemm desc

    // Max splits
    // (max_mn_tiles, max_k_tiles) -> max_splits

    // CTA Swizzling
    // - GemmScheduler: return get_log_tile
    // - DynamicScheduler: bypass

    // Cost estimation
    //

protected:
    int arch_{};
    int sm_count_{};

    GemmDesc desc_{};
    GemmDesc desc_trans_{};
};

}  // namespace turbomind::gemm
