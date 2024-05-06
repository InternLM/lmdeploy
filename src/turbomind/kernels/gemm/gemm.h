// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <memory>

namespace turbomind::gemm {

template<class T, class Tb>
void invoke(
    T* C, const T* A, const Tb* B, const T* Q, int m, int n, int k, int splits, void* workspace, cudaStream_t st);

class DisptchCache {};

class Gemm {
public:
    Gemm();

    ~Gemm();

    int Run(const Operation&    operation,
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
            void*               barriers,
            size_t              barriers_size,
            void*               workspace,
            size_t              workspace_size,
            cudaStream_t        stream);

    // map<GemmDesc, LaunchSpec> -> (LaunchSpec -> [KernelDesc]) -> [(GemmDesc, KernelDesc)]
    [[maybe_unused]] int Export(std::ostream& os);

    // [(GemmDesc, KernelDesc)] -> (KernelDesc -> Kernel*) -> map<GemmDesc, LaunchSpec>
    [[maybe_unused]] int Import(std::istream& is);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind::gemm