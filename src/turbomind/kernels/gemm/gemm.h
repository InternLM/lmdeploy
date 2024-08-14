// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace turbomind::gemm {

class Gemm {
public:
    static constexpr size_t kBarriersSize = 1 << 20;
    static constexpr size_t kPartialsSize = 32 << 20;

    Gemm();

    ~Gemm();

    [[nodiscard]] int Run(const Operation&    operation,
                          float               alpha,
                          const void*         A,
                          const MatrixLayout& Adesc,
                          const void*         U,
                          const MatrixLayout& Udesc,
                          const void*         B,
                          const MatrixLayout& Bdesc,
                          const void*         V,
                          const MatrixLayout& Vdesc,
                          float               beta,
                          const void*         C,
                          const MatrixLayout& Cdesc,
                          void*               D,
                          const MatrixLayout& Ddesc,
                          const Workspace&    workspace,
                          cudaStream_t        stream);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    [[nodiscard]] std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] int
Convert(const void* S, const MatrixLayout& Sdesc, void* D, const MatrixLayout& Ddesc, cudaStream_t stream);

std::tuple<Order, Pack, Order, Pack> get_weight_and_scales_layout(int sm, bool force_simt);

}  // namespace turbomind::gemm
