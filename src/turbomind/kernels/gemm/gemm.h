// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <memory>

namespace turbomind::gemm {

class DisptchCache {};

class Gemm {
public:
    Gemm();

    ~Gemm();

    [[nodiscard]] int Run(const Operation&    operation,
                          const void*         alpha,
                          const void*         A,
                          const MatrixLayout& Adesc,
                          const void*         B,
                          const MatrixLayout& Bdesc,
                          const void*         Q,
                          const MatrixLayout& Qdesc,
                          const void*         beta,
                          const void*         C,
                          const MatrixLayout& Cdesc,
                          void*               D,
                          const MatrixLayout& Ddesc,
                          const Workspace&    workspace,
                          cudaStream_t        stream);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int Convert(const void*         S,  //
            const MatrixLayout& Sdesc,
            void*               D,
            const MatrixLayout& Ddesc,
            cudaStream_t        stream);

}  // namespace turbomind::gemm