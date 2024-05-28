// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"

#include <cublas_v2.h>

namespace turbomind::gemm {

class Reference {
public:
    Reference();
    ~Reference();

    void set_stream(cudaStream_t stream);

    void gemm(const void* A, MatrixLayout Adesc, const void* B, MatrixLayout Bdesc, void* C, MatrixLayout Cdesc);

private:
    cublasHandle_t handle_;
};

}  // namespace turbomind::gemm