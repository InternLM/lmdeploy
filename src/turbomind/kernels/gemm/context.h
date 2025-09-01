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
    virtual ~Context() = default;

    explicit Context(const cudaDeviceProp& prop);

    virtual std::optional<GemmDesc> Init(const Operation&    operation,
                                         const MatrixLayout& Adesc,
                                         const MatrixLayout& Udesc,
                                         const MatrixLayout& Bdesc,
                                         const MatrixLayout& Vdesc,
                                         const MatrixLayout& Cdesc,
                                         const MatrixLayout& Ddesc) = 0;

    virtual std::vector<Kernel*> Filter(const std::vector<Kernel*>& kernels) const = 0;

    virtual std::vector<LaunchSpec> Populate(const Kernel& kernel, const PopulateParam& param) const = 0;

    virtual std::vector<LaunchSpec> Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const = 0;

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

    std::optional<GemmDesc> desc_{};
};

class StaticGemmContext: public Context {
public:
    explicit StaticGemmContext(const cudaDeviceProp& prop);

    std::optional<GemmDesc> Init(const Operation&    operation,
                                 const MatrixLayout& Adesc,
                                 const MatrixLayout& Udesc,
                                 const MatrixLayout& Bdesc,
                                 const MatrixLayout& Vdesc,
                                 const MatrixLayout& Cdesc,
                                 const MatrixLayout& Ddesc) override;

    std::vector<Kernel*> Filter(const std::vector<Kernel*>& kernels) const override;

    std::vector<LaunchSpec> Populate(const Kernel& kernel, const PopulateParam& param) const override;

    std::vector<LaunchSpec> Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const override;
};

}  // namespace turbomind::gemm
