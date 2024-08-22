// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/test/reference.h"
#include <cstdio>

namespace turbomind::gemm {

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "*** Check failed: (%s) @ %s:%d\n", #cond, __FILE__, __LINE__);                            \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

MatrixLayout transpose(MatrixLayout x)
{
    std::swap(x.rows, x.cols);
    x.order = x.order == Order::kColMajor ? Order::kRowMajor : Order::kColMajor;
    return x;
}

cudaDataType to_cuda_dtype(DataType dtype)
{
    switch (dtype) {
        case DataType::F16:
            return CUDA_R_16F;
        case DataType::BF16:
            return CUDA_R_16BF;
        default:
            CHECK("unsupported data type" && 0);
    }
    return {};
}

}  // namespace

Reference::Reference()
{
    cublasCreate(&handle_);
}

Reference::~Reference()
{
    if (handle_) {
        cublasDestroy(handle_);
        handle_ = {};
    }
}

void Reference::set_stream(cudaStream_t stream)
{
    cublasSetStream(handle_, stream);
}

void Reference::gemm(const void* A, MatrixLayout Adesc, const void* B, MatrixLayout Bdesc, void* C, MatrixLayout Cdesc)
{

    // Transpose the problem for C to be column major
    if (Cdesc.order == Order::kRowMajor) {
        std::swap(A, B);
        std::swap(Adesc, Bdesc);
        Adesc = transpose(Adesc);
        Bdesc = transpose(Bdesc);
        Cdesc = transpose(Cdesc);
        // (n, k) (k, m)
    }

    CHECK(Adesc.cols == Bdesc.rows);

    // (m, k) (k, n)
    int m = Cdesc.rows;
    int n = Cdesc.cols;
    int k = Adesc.cols;
    CHECK(Adesc.rows == m);
    CHECK(Bdesc.cols == n);
    CHECK(Bdesc.rows == k);

    float alpha = 1.f;
    float beta  = 0.f;

    auto to_cublas_op = [](Order o) { return o == Order::kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T; };

    auto status = cublasGemmEx(handle_,
                               to_cublas_op(Adesc.order),
                               to_cublas_op(Bdesc.order),
                               m,
                               n,
                               k,
                               &alpha,
                               A,
                               to_cuda_dtype(Adesc.type),
                               Adesc.ld,
                               B,
                               to_cuda_dtype(Bdesc.type),
                               Bdesc.ld,
                               &beta,
                               C,
                               to_cuda_dtype(Cdesc.type),
                               Cdesc.ld,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    CHECK(status == CUBLAS_STATUS_SUCCESS);
}

}  // namespace turbomind::gemm
