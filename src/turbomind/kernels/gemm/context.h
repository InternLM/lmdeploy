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

    virtual Tape Schedule(const LaunchSpec& spec) = 0;

    virtual bool is_dynamic_sched() const noexcept = 0;

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

    Tape Schedule(const LaunchSpec&) override
    {
        return {};
    }

    bool is_dynamic_sched() const noexcept override
    {
        return false;
    }

protected:
};

class DynamicGemmContext: public StaticGemmContext {
public:
    DynamicGemmContext(const cudaDeviceProp& prop, cudaStream_t stream);

    ~DynamicGemmContext() override;

    Tape Schedule(const LaunchSpec&) override;

    bool is_dynamic_sched() const noexcept override
    {
        return true;
    }

protected:
    cudaStream_t stream_;
    Tape         tape_;
    int4         last_shape_{};
    LaunchSpec   last_spec_{};
};

class MoeGemmContext: public Context {
public:
    MoeGemmContext(int experts, int experts_per_token, const cudaDeviceProp& prop, cudaStream_t stream);

    ~MoeGemmContext() override;

    std::optional<GemmDesc> Init(const Operation&    operation,
                                 const MatrixLayout& Adesc,
                                 const MatrixLayout& Udesc,
                                 const MatrixLayout& Bdesc,
                                 const MatrixLayout& Vdesc,
                                 const MatrixLayout& Cdesc,
                                 const MatrixLayout& Ddesc) override;

    std::vector<Kernel*> Filter(const std::vector<Kernel*>& kernels) const override;

    // batch size
    // m: cdiv(exp_per_tok * tokens, experts)

    // FMA_all:
    // m: exp_per_tok * tokens
    // n: output_dims
    // k:  input_dims

    // MIO:
    // A: exp_per_tok * tokens * input_dims
    // C: exp_per_tok * tokens * output_dims
    // B: experts * output_dims * input_dims

    std::vector<LaunchSpec> Populate(const Kernel& kernel, const PopulateParam& param) const override;

    std::vector<LaunchSpec> Swizzle(const LaunchSpec& spec, const std::vector<int>& swizzle) const override;

    bool is_dynamic_sched() const noexcept override
    {
        return true;
    }

    Tape Schedule(const LaunchSpec&) override;

    void update(int expert_num, int experts_per_token, const int* offsets)
    {
        expert_num_        = expert_num;
        experts_per_token_ = experts_per_token;
        offsets_           = offsets;
    }

protected:
    int expert_num_;
    int experts_per_token_;

    cudaStream_t stream_;

    int        output_dim_;
    int        input_dim_;
    int        tokens_;
    const int* offsets_;
    Tape       tape_;
};

}  // namespace turbomind::gemm
