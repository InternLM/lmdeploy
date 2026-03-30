// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/gemm/types.h"

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

}  // namespace turbomind::gemm
