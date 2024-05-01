// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <memory>

namespace turbomind::gemm {

template<class T, class Tb>
void invoke(
    T* C, const T* A, const Tb* B, const T* Q, int m, int n, int k, int splits, void* workspace, cudaStream_t st);

struct QuantDesc {
    QuantType type;
    int       group_size;
};

struct MatmulDesc {
    QuantDesc    quant_desc;
    EpilogueType epilogue;
};

struct WorksapceSpec {
    void*  barriers;
    size_t barriers_size;
    void*  workspace;
    size_t workspace_size;
};

struct MatrixDesc {
    DataType   type;
    LayoutType order;
    int        rows;
    int        cols;
    int        ld;
};

// cublasLtMatmulDesc_t   computeDesc;
// cublasLtMatrixLayout_t Adesc;
// cublasLtMatmulAlgo_t   algo;

class DisptchCache {};

class Gemm {
public:
    Gemm();

    ~Gemm();

    int Run(LayoutType   layout_A,  // row-major
            LayoutType   layout_B,  // col-major or fragment type
            LayoutType   layout_C,  // row-major
            EpilogueType epilogue,
            int          m,
            int          n,
            int          k,
            const void*  A,
            DataType     type_A,  // f16
            int          lda,
            const void*  B,
            DataType     type_B,  // u4
            int          ldb,
            const void*  Q,
            QuantType    quant_type,  // asym
            int          ldq,
            const float* beta,
            void*        C,
            DataType     type_C,  // f16
            int          ldc,     // return by converter
            int*         barriers,
            size_t       barriers_size,
            void*        workspace,
            size_t       workspace_size,
            cudaStream_t stream);

    int Run_v2(const MatmulDesc& compute_desc,
               const void*       alpha,
               const void*       A,
               const MatrixDesc& Adesc,
               const void*       B,
               const MatrixDesc& Bdesc,
               const void*       Q,
               const MatrixDesc& Qdesc,
               const void*       beta,
               const void*       C,
               const MatrixDesc& Cdesc,
               void*             D,
               const MatrixDesc& Ddesc,
               void*             barriers,
               size_t            barriers_size,
               void*             workspace,
               size_t            workspace_size,
               cudaStream_t      stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind::gemm