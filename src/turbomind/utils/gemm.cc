/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/turbomind/utils/gemm.h"

namespace turbomind {

/* ***************************** GEMM Impl ******************************** */

Gemm::Gemm(IAllocator* allocator, cudaStream_t stream, std::string config_file)
{
    allocator_ = allocator;
    stream_    = stream;
    mutex_     = new std::mutex();  // mutex per process
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    check_cuda_error(cublasSetStream(cublas_handle_, stream));

    if (allocator_ != nullptr) {
        workspace_ = allocator_->reMalloc(workspace_, WORKSPACE_SIZE);
    }
    loadGemmConfig(config_file);
}

Gemm::~Gemm()
{
    if (allocator_ != nullptr) {
        allocator_->free((void**)(&workspace_));
        allocator_ = nullptr;
    }
    cublasLtDestroy(cublaslt_handle_);
    cublasDestroy(cublas_handle_);
    delete cublas_algo_map_;
    delete mutex_;
}

std::string Gemm::toString()
{
    const char* a_type_str       = a_type_ == TYPE_FP16 ? "FP16" : "FP32";
    const char* b_type_str       = b_type_ == TYPE_FP16 ? "FP16" : "FP32";
    const char* c_type_str       = c_type_ == TYPE_FP16 ? "FP16" : "FP32";
    const char* compute_type_str = compute_type_ == TYPE_FP16 ? "FP16" : "FP32";
    return fmtstr(
        "Gemm[a_type=%s, b_type=%s, c_type=%s, compute_type=%s]", a_type_str, b_type_str, c_type_str, compute_type_str);
}

void Gemm::setAllocator(IAllocator* allocator)
{
    if (allocator_ != nullptr && workspace_ != nullptr) {
        allocator_->free((void**)(&workspace_));
    }
    allocator_ = allocator;
    if (allocator_ != nullptr) {
        workspace_ = allocator_->reMalloc(workspace_, WORKSPACE_SIZE);
    }
}

void Gemm::setCudaStream(cudaStream_t& stream)
{
    stream_ = stream;
    cublasSetStream(cublas_handle_, stream);
}

void Gemm::setComputeType(DataType compute_type)
{
    checkDataTypeValidity(compute_type);
    compute_type_ = compute_type;
}

void Gemm::setTypes(DataType a_type, DataType b_type, DataType c_type, DataType compute_type)
{
    checkDataTypeValidity(a_type);
    checkDataTypeValidity(b_type);
    checkDataTypeValidity(c_type);
    a_type_ = a_type;
    b_type_ = b_type;
    c_type_ = c_type;
    setComputeType(compute_type);
}

template<typename T>
void Gemm::setDefaultTypes()
{
    if (std::is_same<T, float>::value) {
        setTypes(TYPE_FP32, TYPE_FP32, TYPE_FP32, TYPE_FP32);
    }
    else if (std::is_same<T, half>::value) {
        setTypes(TYPE_FP16, TYPE_FP16, TYPE_FP16, TYPE_FP16);
    }
    else {
        throw GemmNotSupportedException("Gemm supports float or half type.");
    }
}

void Gemm::loadGemmConfig(std::string config_file)
{
    if (cublas_algo_map_ != nullptr) {
        delete cublas_algo_map_;  // unload the previous cublas map.
    }
    cublas_algo_map_ = new cublasAlgoMap(config_file);
}

void Gemm::gemm(const GemmOp              transa,
                const GemmOp              transb,
                const size_t              m,
                const size_t              n,
                const size_t              k,
                const void*               input,
                const DenseWeight<float>& weight,
                void*                     output,
                const float               alpha,
                const float               beta)
{
    gemm(transa,
         transb,
         m,
         n,
         k,
         input,
         a_type_,
         (transa == GEMM_OP_N) ? k : m,
         (const void*)weight.kernel,
         b_type_,
         (transb == GEMM_OP_N) ? n : k,
         output,
         c_type_,
         n,
         alpha,
         beta);
}

void Gemm::gemm(const GemmOp             transa,
                const GemmOp             transb,
                const size_t             m,
                const size_t             n,
                const size_t             k,
                const void*              input,
                const DenseWeight<half>& weight,
                void*                    output,
                const float              alpha,
                const float              beta)
{
    gemm(transa,
         transb,
         m,
         n,
         k,
         input,
         a_type_,
         (transa == GEMM_OP_N) ? k : m,
         (const void*)weight.kernel,
         b_type_,
         (transb == GEMM_OP_N) ? n : k,
         output,
         c_type_,
         n,
         alpha,
         beta);
}

void Gemm::gemm(const GemmOp transa,
                const GemmOp transb,
                const size_t m,
                const size_t n,
                const size_t k,
                const void*  A,
                const void*  B,
                void*        C,
                const float  alpha,
                const float  beta)
{
    size_t lda = (transa == GEMM_OP_N) ? k : m;
    size_t ldb = (transb == GEMM_OP_N) ? n : k;
    size_t ldc = n;
    gemm(transa, transb, m, n, k, A, a_type_, lda, B, b_type_, ldb, C, c_type_, ldc, alpha, beta);
}

void Gemm::gemm(const GemmOp transa,
                const GemmOp transb,
                const size_t m,
                const size_t n,
                const size_t k,
                const void*  A,
                const size_t lda,
                const void*  B,
                const size_t ldb,
                void*        C,
                const size_t ldc,
                const float  alpha,
                const float  beta)
{
    gemm(transa, transb, m, n, k, A, a_type_, lda, B, b_type_, ldb, C, c_type_, ldc, alpha, beta);
}

void Gemm::gemm(const GemmOp   transa,
                const GemmOp   transb,
                const size_t   m,
                const size_t   n,
                const size_t   k,
                const void*    A,
                const DataType Atype,
                const size_t   lda,
                const void*    B,
                const DataType Btype,
                const size_t   ldb,
                void*          C,
                const DataType Ctype,
                const size_t   ldc,
                const float    alpha,
                const float    beta)
{
    FT_LOG_TRACE("Gemm::gemm [m=%ld, n=%ld, k=%ld, lda=%ld, ldb=%ld, ldc=%ld]", m, n, k, lda, ldb, ldc);

    // Implementation copied from cublasMMWrapper::Gemm
    // Switch A and B since both cublas and cublasLt assume a column major layout,
    // while A and B are both row major layout.
    const void* a_data_ptr = B;
    const void* b_data_ptr = A;

    cublasOperation_t a_op = getCublasOperation(transb);
    cublasOperation_t b_op = getCublasOperation(transa);

    cudaDataType_t a_type = getCublasDataType(Btype);
    cudaDataType_t b_type = getCublasDataType(Atype);
    cudaDataType_t c_type = getCublasDataType(Ctype);

    // swap m and n
    const size_t _m = n;
    const size_t _n = m;

    // swap lda and ldb;
    const size_t _lda = ldb;
    const size_t _ldb = lda;

    mutex_->lock();
    // Use cublas as default in FP32 and cublasLt as default in FP16
    bool is_fp16_compute_type = compute_type_ == TYPE_FP16;
    bool using_cublasLt       = Atype == TYPE_FP16;
    int  batch_count          = 1;

    half        h_alpha = (half)alpha;
    half        h_beta  = (half)beta;
    const void* alpha_ptr =
        is_fp16_compute_type ? reinterpret_cast<const void*>(&h_alpha) : reinterpret_cast<const void*>(&alpha);
    const void* beta_ptr =
        is_fp16_compute_type ? reinterpret_cast<const void*>(&h_beta) : reinterpret_cast<const void*>(&beta);

    // TODO: unify CUBLAS_DATA_TYPE and DataType.
    int findAlgo =
        cublas_algo_map_->isExist(batch_count, _m, _n, k, (a_type == CUDA_R_16F) ? HALF_DATATYPE : FLOAT_DATATYPE);
    cublasLtMatmulAlgo_info info =
        cublas_algo_map_->getAlgo(batch_count, _m, _n, k, (a_type == CUDA_R_16F) ? HALF_DATATYPE : FLOAT_DATATYPE);
    if (findAlgo) {
        using_cublasLt = (info.stages != -1);
    }

    if (using_cublasLt) {
        const size_t a_rows = (a_op == getCublasOperation(GEMM_OP_N)) ? _m : k;
        const size_t a_cols = (a_op == getCublasOperation(GEMM_OP_N)) ? k : _m;
        const size_t b_rows = (b_op == getCublasOperation(GEMM_OP_N)) ? k : _n;
        const size_t b_cols = (b_op == getCublasOperation(GEMM_OP_N)) ? _n : k;

        cublasLtMatmulDesc_t   matmul_desc = NULL;
        cublasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL;
        cudaDataType_t         scale_type   = getCublasDataType(compute_type_);
        auto                   compute_type = getCublasComputeType(compute_type_);

        // --------------------------------------
        // Create descriptors for the original matrices
        cublasLtMatrixLayoutCreate(&a_desc, a_type, a_rows, a_cols, _lda);
        cublasLtMatrixLayoutCreate(&b_desc, b_type, b_rows, b_cols, _ldb);
        cublasLtMatrixLayoutCreate(&c_desc, c_type, _m, _n, ldc);
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type);
#else
        cublasLtMatmulDescCreate(&matmul_desc, compute_type);
#endif

        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &a_op, sizeof(cublasOperation_t));
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &b_op, sizeof(cublasOperation_t));

        cublasLtMatmulAlgo_t algo;
        void*                workspace      = workspace_;
        int                  workspace_size = workspace_ == nullptr ? 0 : CUBLAS_WORKSPACE_SIZE;
        if (findAlgo) {
            if (info.workspaceSize > workspace_size) {
                findAlgo = 0;
            }
            else {
                cublasLtMatmulAlgoInit(
                    cublaslt_handle_, compute_type, scale_type, a_type, b_type, c_type, c_type, info.algoId, &algo);
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));
#endif
            }
        }

        cublasLtMatmul(cublaslt_handle_,
                       matmul_desc,
                       alpha_ptr,
                       a_data_ptr,
                       a_desc,
                       b_data_ptr,
                       b_desc,
                       beta_ptr,
                       C,
                       c_desc,
                       C,
                       c_desc,
                       (findAlgo == 1 ? (&algo) : NULL),
                       workspace,
                       workspace_size,
                       stream_);

        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        sync_check_cuda_error();
    }
    else {
        cudaDataType_t compute_type = getCublasDataType(compute_type_);
        int            cublas_algo  = info.algoId;
        check_cuda_error(cublasGemmEx(cublas_handle_,
                                      a_op,
                                      b_op,
                                      _m,
                                      _n,
                                      k,
                                      alpha_ptr,
                                      a_data_ptr,
                                      a_type,
                                      _lda,
                                      b_data_ptr,
                                      b_type,
                                      _ldb,
                                      beta_ptr,
                                      C,
                                      c_type,
                                      ldc,
                                      compute_type,
                                      static_cast<cublasGemmAlgo_t>(cublas_algo)));
        sync_check_cuda_error();
    }
    mutex_->unlock();
}

void Gemm::batchedGemm(const GemmOp       transa,
                       const GemmOp       transb,
                       const size_t       m,
                       const size_t       n,
                       const size_t       k,
                       const void* const* A,
                       const void* const* B,
                       void* const*       C,
                       const size_t       batch_size,
                       const float        alpha,
                       const float        beta)
{
    size_t lda = (transa == GEMM_OP_N) ? k : m;
    size_t ldb = (transb == GEMM_OP_N) ? n : k;
    size_t ldc = n;
    batchedGemm(transa, transb, m, n, k, A, a_type_, lda, B, b_type_, ldb, C, c_type_, ldc, batch_size, alpha, beta);
}

void Gemm::batchedGemm(const GemmOp       transa,
                       const GemmOp       transb,
                       const size_t       m,
                       const size_t       n,
                       const size_t       k,
                       const void* const* A,
                       const size_t       lda,
                       const void* const* B,
                       const size_t       ldb,
                       void* const*       C,
                       const size_t       ldc,
                       const size_t       batch_size,
                       const float        alpha,
                       const float        beta)
{
    batchedGemm(transa, transb, m, n, k, A, a_type_, lda, B, b_type_, ldb, C, c_type_, ldc, batch_size, alpha, beta);
}

void Gemm::batchedGemm(const GemmOp       transa,
                       const GemmOp       transb,
                       const size_t       m,
                       const size_t       n,
                       const size_t       k,
                       const void* const* A,
                       const DataType     Atype,
                       const size_t       lda,
                       const void* const* B,
                       const DataType     Btype,
                       const size_t       ldb,
                       void* const*       C,
                       const DataType     Ctype,
                       const size_t       ldc,
                       const size_t       batch_size,
                       const float        alpha,
                       const float        beta)
{
    FT_LOG_TRACE(
        "Gemm::batchedGemm [b=%ld m=%ld, n=%ld, k=%ld, lda=%ld, ldb=%ld, ldc=%ld]", batch_size, m, n, k, lda, ldb, ldc);

    // Switch A and B.
    const void* const* a_data_ptr = B;
    const void* const* b_data_ptr = A;

    cublasOperation_t a_op = getCublasOperation(transb);
    cublasOperation_t b_op = getCublasOperation(transa);

    cudaDataType_t a_type = getCublasDataType(Btype);
    cudaDataType_t b_type = getCublasDataType(Atype);
    cudaDataType_t c_type = getCublasDataType(Ctype);

    // swap m and n, lda and ldb
    const size_t _m   = n;
    const size_t _n   = m;
    const size_t _lda = ldb;
    const size_t _ldb = lda;

    half h_alpha = (half)alpha;
    half h_beta  = (half)beta;

    mutex_->lock();
    bool        is_fp16_compute_type = compute_type_ == TYPE_FP16;
    const void* alpha_ptr =
        is_fp16_compute_type ? reinterpret_cast<const void*>(&h_alpha) : reinterpret_cast<const void*>(&alpha);
    const void* beta_ptr =
        is_fp16_compute_type ? reinterpret_cast<const void*>(&h_beta) : reinterpret_cast<const void*>(&beta);
    cublasLtMatmulAlgo_info info =
        cublas_algo_map_->getAlgo(batch_size, m, n, k, (a_type == CUDA_R_16F) ? HALF_DATATYPE : FLOAT_DATATYPE);

    check_cuda_error(cublasGemmBatchedEx(cublas_handle_,
                                         a_op,
                                         b_op,
                                         _m,
                                         _n,
                                         k,
                                         alpha_ptr,
                                         a_data_ptr,
                                         a_type,
                                         _lda,
                                         b_data_ptr,
                                         b_type,
                                         _ldb,
                                         beta_ptr,
                                         C,
                                         c_type,
                                         ldc,
                                         batch_size,
                                         getCublasComputeType(compute_type_),
                                         static_cast<cublasGemmAlgo_t>(info.algoId)));
    mutex_->unlock();
}

void Gemm::stridedBatchedGemm(GemmOp       transa,
                              GemmOp       transb,
                              const size_t m,
                              const size_t n,
                              const size_t k,
                              const void*  A,
                              const void*  B,
                              void*        C,
                              const size_t batch_size,
                              const float  alpha,
                              const float  beta)
{
    size_t  lda     = (transa == GEMM_OP_N) ? k : m;
    size_t  ldb     = (transb == GEMM_OP_N) ? n : k;
    size_t  ldc     = n;
    int64_t stridea = m * k;
    int64_t strideb = k * n;
    int64_t stridec = m * n;

    stridedBatchedGemm(transa,
                       transb,
                       m,
                       n,
                       k,
                       A,
                       a_type_,
                       lda,
                       stridea,
                       B,
                       b_type_,
                       ldb,
                       strideb,
                       C,
                       c_type_,
                       ldc,
                       stridec,
                       batch_size,
                       compute_type_,
                       alpha,
                       beta);
}

void Gemm::stridedBatchedGemm(GemmOp        transa,
                              GemmOp        transb,
                              const size_t  m,
                              const size_t  n,
                              const size_t  k,
                              const void*   A,
                              const int64_t strideA,
                              const void*   B,
                              const int64_t strideB,
                              void*         C,
                              const int64_t strideC,
                              const size_t  batch_size,
                              const float   alpha,
                              const float   beta)
{
    size_t lda = (transa == GEMM_OP_N) ? k : m;
    size_t ldb = (transb == GEMM_OP_N) ? n : k;
    size_t ldc = n;
    stridedBatchedGemm(transa,
                       transb,
                       m,
                       n,
                       k,
                       A,
                       a_type_,
                       lda,
                       strideA,
                       B,
                       b_type_,
                       ldb,
                       strideB,
                       C,
                       c_type_,
                       ldc,
                       strideC,
                       batch_size,
                       compute_type_,
                       alpha,
                       beta);
}

void Gemm::stridedBatchedGemm(GemmOp        transa,
                              GemmOp        transb,
                              const size_t  m,
                              const size_t  n,
                              const size_t  k,
                              const void*   A,
                              const size_t  lda,
                              const int64_t strideA,
                              const void*   B,
                              const size_t  ldb,
                              const int64_t strideB,
                              void*         C,
                              const size_t  ldc,
                              const int64_t strideC,
                              const size_t  batch_size,
                              const float   alpha,
                              const float   beta)
{
    stridedBatchedGemm(transa,
                       transb,
                       m,
                       n,
                       k,
                       A,
                       a_type_,
                       lda,
                       strideA,
                       B,
                       b_type_,
                       ldb,
                       strideB,
                       C,
                       c_type_,
                       ldc,
                       strideC,
                       batch_size,
                       compute_type_,
                       alpha,
                       beta);
}

void Gemm::stridedBatchedGemm(GemmOp        transa,
                              GemmOp        transb,
                              const size_t  m,
                              const size_t  n,
                              const size_t  k,
                              const void*   A,
                              DataType      Atype,
                              const size_t  lda,
                              const int64_t strideA,
                              const void*   B,
                              DataType      Btype,
                              const size_t  ldb,
                              const int64_t strideB,
                              void*         C,
                              DataType      Ctype,
                              const size_t  ldc,
                              const int64_t strideC,
                              const size_t  batch_size,
                              DataType      compute_type,
                              const float   alpha,
                              const float   beta)
{
    FT_LOG_TRACE("Gemm::stridedBatchedGemm [b=%ld, m=%ld, n=%ld, k=%ld, lda=%ld, ldb=%ld, ldc=%ld]",
                 batch_size,
                 m,
                 n,
                 k,
                 lda,
                 ldb,
                 ldc);

    // Switch A and B.
    const void* a_data_ptr = B;
    const void* b_data_ptr = A;

    cublasOperation_t a_op = getCublasOperation(transb);
    cublasOperation_t b_op = getCublasOperation(transa);

    cudaDataType_t a_type = getCublasDataType(Btype);
    cudaDataType_t b_type = getCublasDataType(Atype);
    cudaDataType_t c_type = getCublasDataType(Ctype);

    // swap m and n, lda and ldb, stride A and B
    const size_t  _m       = n;
    const size_t  _n       = m;
    const size_t  _lda     = ldb;
    const size_t  _ldb     = lda;
    const int64_t _stridea = strideB;
    const int64_t _strideb = strideA;

    half h_alpha = (half)alpha;
    half h_beta  = (half)beta;

    mutex_->lock();
    bool        is_fp16_compute_type = compute_type_ == TYPE_FP16;
    const void* alpha_ptr =
        is_fp16_compute_type ? reinterpret_cast<const void*>(&h_alpha) : reinterpret_cast<const void*>(&alpha);
    const void* beta_ptr =
        is_fp16_compute_type ? reinterpret_cast<const void*>(&h_beta) : reinterpret_cast<const void*>(&beta);
    cublasLtMatmulAlgo_info info =
        cublas_algo_map_->getAlgo(batch_size, m, n, k, (a_type == CUDA_R_16F) ? HALF_DATATYPE : FLOAT_DATATYPE);

    check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle_,
                                                a_op,
                                                b_op,
                                                _m,
                                                _n,
                                                k,
                                                alpha_ptr,
                                                a_data_ptr,
                                                a_type,
                                                _lda,
                                                _stridea,
                                                b_data_ptr,
                                                b_type,
                                                _ldb,
                                                _strideb,
                                                beta_ptr,
                                                C,
                                                c_type,
                                                ldc,
                                                strideC,
                                                batch_size,
                                                getCublasComputeType(compute_type),
                                                static_cast<cublasGemmAlgo_t>(info.algoId)));
    mutex_->unlock();
}

void Gemm::checkDataTypeValidity(const DataType& type)
{
    if (type != TYPE_FP32 && type != TYPE_FP16) {
        throw GemmNotSupportedException("Gemm supports TYPE_FP16 or TYPE_FP32");
    }
}

/* ************************* End of GEMM Impl **************************** */

// void Int8Gemm::gemm(Tensor& C,
//                     const GemmOp transa,
//                     const GemmOp transb,
//                     const Tensor& A,
//                     const Tensor& B,
//                     const float alpha,
//                     const float beta)
// {

// }

/* ************************* SpGEMM Impl *********************************** */
#ifdef SPARSITY_ENABLED
SpGemm::SpGemm(IAllocator* allocator, cudaStream_t stream, std::string config_file, std::string spconfig_file):
    Gemm(allocator, stream, config_file)
{
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle_));
    // TODO(jaedeokk):
    //   Let's make cublasAlgoMap load gemm/spgemm config separtely,
    //   allowing us to inherit Gemm's constructor.
    // cublas_algo_map_.loadSpGemmConfig(spconfig_file);  // enable this line later.

    a_type_       = TYPE_FP16;
    b_type_       = TYPE_FP16;
    c_type_       = TYPE_FP16;
    compute_type_ = TYPE_FP16;
}

SpGemm::~SpGemm()
{
    cusparseLtDestroy(&cusparselt_handle_);
    // Need to destroy matmul description cache.
    for (auto& kv : a_desc_map_) {  // kv = (mark, a_desc)
        cusparseLtMatDescriptorDestroy(&a_desc_map_[kv.first]);
    }
    for (auto& kv : b_desc_map_) {  // kv = (mark, b_desc)
        cusparseLtMatDescriptorDestroy(&b_desc_map_[kv.first]);
    }
    for (auto& kv : c_desc_map_) {  // kv = (mark, c_desc)
        cusparseLtMatDescriptorDestroy(&c_desc_map_[kv.first]);
    }
}

std::string SpGemm::toString()
{
    const char* a_type_str       = a_type_ == TYPE_FP16 ? "FP16" : "FP32";
    const char* b_type_str       = b_type_ == TYPE_FP16 ? "FP16" : "FP32";
    const char* c_type_str       = c_type_ == TYPE_FP16 ? "FP16" : "FP32";
    const char* compute_type_str = compute_type_ == TYPE_FP16 ? "FP16" : "FP32";
    return fmtstr("SpGemm[a_type=%s, b_type=%s, c_type=%s, compute_type=%s]",
                  a_type_str,
                  b_type_str,
                  c_type_str,
                  compute_type_str);
}

void SpGemm::loadGemmConfig(std::string config_file, std::string spconfig_file)
{
    if (cublas_algo_map_ != nullptr) {
        delete cublas_algo_map_;  // unload algo map.
    }
    cublas_algo_map_ = new cublasAlgoMap(config_file, spconfig_file);
}

void SpGemm::checkDataTypeValidity(const DataType& type)
{
    if (type != TYPE_FP16) {
        throw GemmNotSupportedException("Sparse GEMM only supports FP16 data type now.");
    }
}

bool SpGemm::useBaseGemm(size_t batch_size, size_t m, size_t n, size_t k)
{
    return !cublas_algo_map_->isUseSparse(batch_size, m, n, k);
}

// Temporal gemm helper mtehod to use template T.
template<typename T>
void SpGemm::weightGemmHelper(const GemmOp          transa,
                              const GemmOp          transb,
                              const size_t          m,
                              const size_t          n,
                              const size_t          k,
                              const void*           input,
                              const DenseWeight<T>& weight,
                              void*                 output,
                              const float           alpha,
                              const float           beta)
{
    size_t lda = (transa == GEMM_OP_N) ? k : m;
    size_t ldb = (transb == GEMM_OP_N) ? n : k;
    size_t ldc = n;
    if (useBaseGemm(1, m, n, k) || weight.sp_kernel == nullptr) {
        Gemm::gemm(transa,
                   transb,
                   m,
                   n,
                   k,
                   input,
                   a_type_,
                   lda,
                   (const void*)weight.kernel,
                   b_type_,
                   ldb,
                   output,
                   c_type_,
                   ldc,
                   alpha,
                   beta);
    }
    else {
        gemm(transa,
             transb,
             m,
             n,
             k,
             input,
             a_type_,
             lda,
             (const void*)weight.sp_kernel,
             b_type_,
             ldb,
             output,
             c_type_,
             ldc,
             alpha,
             beta);
    }
}

void SpGemm::gemm(const GemmOp              transa,
                  const GemmOp              transb,
                  const size_t              m,
                  const size_t              n,
                  const size_t              k,
                  const void*               input,
                  const DenseWeight<float>& weight,
                  void*                     output,
                  const float               alpha,
                  const float               beta)
{
    weightGemmHelper<float>(transa, transb, m, n, k, input, weight, output, alpha, beta);
}
void SpGemm::gemm(const GemmOp             transa,
                  const GemmOp             transb,
                  const size_t             m,
                  const size_t             n,
                  const size_t             k,
                  const void*              input,
                  const DenseWeight<half>& weight,
                  void*                    output,
                  const float              alpha,
                  const float              beta)
{
    weightGemmHelper<half>(transa, transb, m, n, k, input, weight, output, alpha, beta);
}

void SpGemm::gemm(const GemmOp   transa,
                  const GemmOp   transb,
                  const size_t   m,
                  const size_t   n,
                  const size_t   k,
                  const void*    A,
                  const DataType Atype,
                  const size_t   lda,
                  const void*    B,
                  const DataType Btype,
                  const size_t   ldb,
                  void*          C,
                  const DataType Ctype,
                  const size_t   ldc,
                  const float    alpha,
                  const float    beta)
{
    FT_LOG_TRACE("SpGemm::gemm [m=%ld, n=%ld, k=%ld, lda=%ld, ldb=%ld, ldc=%ld]", m, n, k, lda, ldb, ldc);
    checkDataTypeValidity(Atype);
    checkDataTypeValidity(Btype);
    checkDataTypeValidity(Ctype);
    checkDataTypeValidity(compute_type_);

    if (useBaseGemm(1, m, n, k)) {
        // Compute by the base GEMM.
        Gemm::gemm(transa, transb, m, n, k, A, Atype, lda, B, Btype, ldb, C, Ctype, ldc, alpha, beta);
        return;
    }

    // Switch A/B due to column major layout in computation.
    //  Typical usecase of Gemm family is to compute Y = X * W where X is an
    //  input tensor and W is a kernel weight. Compression takes a lot time
    //  so only the kernel weight (which is fixed in inference time) can be
    //  sparse. Using B as sparse seems not stable, unfortunately.
    //  (e.g. caching matrix descriptions is not correctly working.)
    //  Thus, SpGemm considers a column major layout in computation to make
    //  C^T = B^T * A^T, where a kernel weight "B" is located at the front.
    const void* a_data = B;
    const void* b_data = A;

    cusparseOrder_t order = CUSPARSE_ORDER_COL;

    cusparseOperation_t opA = getCusparseOperation(transb);
    cusparseOperation_t opB = getCusparseOperation(transa);

    cudaDataType_t a_type = getCublasDataType(Btype);
    cudaDataType_t b_type = getCublasDataType(Atype);
    cudaDataType_t c_type = getCublasDataType(Ctype);

    const size_t _m   = n;
    const size_t _n   = m;
    const size_t _lda = ldb;
    const size_t _ldb = lda;

    const size_t a_rows = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? _m : k;
    const size_t a_cols = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? k : _m;
    const size_t b_rows = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? k : _n;
    const size_t b_cols = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? _n : k;
    const size_t c_rows = _m;
    const size_t c_cols = _n;

    const unsigned      alignment    = 16;
    cusparseComputeType compute_type = getCusparseComputeType(compute_type_);

    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;

    char mark[256];
    sprintf(mark, "%d_%ld_%ld_%ld_%s_%s", 1, m, n, k, getGemmOpString(transb).c_str(), getGemmOpString(transa).c_str());
    if (a_desc_map_.find(mark) != a_desc_map_.end()) {
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&cusparselt_handle_,
                                                      &matmul,
                                                      opA,
                                                      opB,
                                                      &a_desc_map_[mark],
                                                      &b_desc_map_[mark],
                                                      &c_desc_map_[mark],
                                                      &c_desc_map_[mark],
                                                      compute_type));
    }
    else {
        // initializing MatDesc takes a lot of time
        cusparseLtMatDescriptor_t a_desc, b_desc, c_desc;
        a_desc_map_[mark] = a_desc;
        b_desc_map_[mark] = b_desc;
        c_desc_map_[mark] = c_desc;
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&cusparselt_handle_,
                                                          &a_desc_map_[mark],
                                                          a_rows,
                                                          a_cols,
                                                          _lda,
                                                          alignment,
                                                          a_type,
                                                          order,
                                                          CUSPARSELT_SPARSITY_50_PERCENT));
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &cusparselt_handle_, &b_desc_map_[mark], b_rows, b_cols, _ldb, alignment, b_type, order));
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &cusparselt_handle_, &c_desc_map_[mark], c_rows, c_cols, ldc, alignment, c_type, order));
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&cusparselt_handle_,
                                                      &matmul,
                                                      opA,
                                                      opB,
                                                      &a_desc_map_[mark],
                                                      &b_desc_map_[mark],
                                                      &c_desc_map_[mark],
                                                      &c_desc_map_[mark],
                                                      compute_type));
    }

    mutex_->lock();
    CHECK_CUSPARSE(
        cusparseLtMatmulAlgSelectionInit(&cusparselt_handle_, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    int alg = cublas_algo_map_->getSpAlgo(1, a_rows, b_cols, a_cols);
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cusparselt_handle_, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)));
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&cusparselt_handle_, &alg_sel, &workspace_size));
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&cusparselt_handle_, &plan, &matmul, &alg_sel, workspace_size));

    void*        d_workspace = nullptr;  // Can we use the workspace of the class?
    int          num_streams = 1;
    cudaStream_t streams[1]  = {stream_};
    CHECK_CUSPARSE(cusparseLtMatmul(
        &cusparselt_handle_, &plan, &alpha, a_data, b_data, &beta, C, C, d_workspace, streams, num_streams))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    mutex_->unlock();
    sync_check_cuda_error();
}
#endif

/* ************************* End of SpGEMM Impl ************************** */

/* ***************************** GEMM utils ****************************** */

std::shared_ptr<Gemm> createGemm(IAllocator* allocator, cudaStream_t stream, bool sparse, bool quantized)
{
    FT_LOG_TRACE(
        "Create Gemm instance [sparse=%s, quantized=%s]", sparse ? "true" : "false", quantized ? "true" : "false");
    std::shared_ptr<Gemm> gemm;
    if (!sparse) {
        if (!quantized) {
            gemm = std::make_shared<Gemm>(allocator, stream);
        }
        else {
            throw GemmNotSupportedException("Int8 Gemm is not supported yet");
        }
    }
    else {
#ifdef SPARSITY_ENABLED
        if (sparse && !quantized) {
            gemm = std::make_shared<SpGemm>(allocator, stream);
        }
        else {
            throw GemmNotSupportedException("Int8 Sparse Gemm is not supported yet");
        }
#else
        throw GemmNotSupportedException("Sparsity support is not enabled. To enabled sparisty, "
                                        "please provide `-DSPARSITY_SUPPORT` flag for compilation.");
#endif
    }
    return gemm;
}

cudaDataType_t getCublasDataType(DataType dtype)
{
    switch (dtype) {
        case TYPE_FP16:
            return CUDA_R_16F;
        case TYPE_FP32:
            return CUDA_R_32F;
        default:
            throw GemmNotSupportedException("Not supported data type.");
    }
}

#if (CUDART_VERSION >= 11000)
cublasComputeType_t getCublasComputeType(DataType ctype)
{
    switch (ctype) {
        case TYPE_FP16:
            return CUBLAS_COMPUTE_16F;
        case TYPE_FP32:
            return CUBLAS_COMPUTE_32F;
        default:
            throw GemmNotSupportedException("Not supported cublas compute type.");
    }
}
#else
cudaDataType_t getCublasComputeType(DataType ctype)
{
    switch (ctype) {
        case TYPE_FP16:
            return CUDA_R_16F;
        case TYPE_FP32:
            return CUDA_R_32F;
        default:
            throw GemmNotSupportedException("Not supported cublas compute type.");
    }
}
#endif

cublasOperation_t getCublasOperation(GemmOp op)
{
    switch (op) {
        case GEMM_OP_N:
            return CUBLAS_OP_N;
        case GEMM_OP_T:
            return CUBLAS_OP_T;
        default:
            throw GemmNotSupportedException("Unknown GemmOp provided.");
    }
}

std::string getGemmOpString(const GemmOp& op)
{
    switch (op) {
        case GEMM_OP_T:
            return "T";
        case GEMM_OP_N:
            return "N";
    }
    throw GemmNotSupportedException("Unknown GemmOp provided.");
}

#ifdef SPARSITY_ENABLED
cusparseOperation_t getCusparseOperation(GemmOp op)
{
    switch (op) {
        case GEMM_OP_N:
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        case GEMM_OP_T:
            return CUSPARSE_OPERATION_TRANSPOSE;
        default:
            throw GemmNotSupportedException("Unknown GemmOp provided.");
    }
}

cusparseComputeType getCusparseComputeType(DataType ctype)
{
    if (ctype != TYPE_FP16) {
        throw GemmNotSupportedException("Sparse GEMM supports TYPE_FP16 compute type only.");
    }
    return CUSPARSE_COMPUTE_16F;
}

void pruneMatrixB(void* data, const cudaStream_t& stream, const size_t k, const size_t n, const GemmOp trans)
{
    FT_LOG_TRACE("Prune matrix B [k=%ld, n=%ld, op=%s]", k, n, getGemmOpString(trans).c_str());

    // Due to A/B switching, the matrix B will be used as a matrix A.
    const cusparseOrder_t order     = CUSPARSE_ORDER_COL;
    const size_t          rows      = (trans == GEMM_OP_N) ? n : k;
    const size_t          cols      = (trans == GEMM_OP_N) ? k : n;
    const size_t          ld        = rows;
    const unsigned        alignment = 16;

    const cusparseLtPruneAlg_t prune_alg = CUSPARSELT_PRUNE_SPMMA_STRIP;
    const cusparseOperation_t  op        = getCusparseOperation(trans);
    const cudaDataType_t       dtype     = CUDA_R_16F;  // fixed under cusparselt == 0.2.0.

    // 0: B is sparse,  1: A is sparse
    // B matrix will be used as A matrix at the SpGemm::gemm.
    const int is_sparse_a = 1;

    // TODO: Let the resource manager handle GPU-related resources later.
    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));
    cusparseLtMatDescriptor_t mat_desc;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &handle, &mat_desc, rows, cols, ld, alignment, dtype, order, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(cusparseLtSpMMAPrune2(&handle, &mat_desc, is_sparse_a, op, data, data, prune_alg, stream));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&mat_desc));
    CHECK_CUSPARSE(cusparseLtDestroy(&handle));
}

size_t compressMatrixB(void**              output,
                       IAllocator&         allocator,
                       const cudaStream_t& stream,
                       const void*         input,
                       const size_t        k,
                       const size_t        n,
                       const GemmOp        trans)
{
    FT_LOG_TRACE("compressMatrix [k=%ld, n=%ld, dtype=FP16]", k, n);

    // swap A/B due to column/row major layout mismatch.
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    const size_t    rows  = (trans == GEMM_OP_N) ? n : k;
    const size_t    cols  = (trans == GEMM_OP_N) ? k : n;
    const size_t    ld    = rows;

    cudaDataType_t            dtype    = CUDA_R_16F;  // fixed under cusparselt == 0.2.0.
    cusparseLtSparsity_t      sparsity = CUSPARSELT_SPARSITY_50_PERCENT;
    cusparseOperation_t       op       = getCusparseOperation(trans);
    cusparseLtMatDescriptor_t mat_desc;
    const unsigned            alignment   = 16;
    const int                 is_sparse_a = 1;  // 0: B is sparse,  1: A is sparse

    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));

    CHECK_CUSPARSE(
        cusparseLtStructuredDescriptorInit(&handle, &mat_desc, rows, cols, ld, alignment, dtype, order, sparsity))

    size_t compressed_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, &mat_desc, &compressed_size));
    if (compressed_size == 0) {
        throw GemmInvalidException("Fail to compute correct compressed_size, got 0. This error may be "
                                   "caused by a too small input matrix.");
    }

    *output = allocator.malloc(compressed_size, false);
    CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, &mat_desc, is_sparse_a, op, input, *output, stream))

    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&mat_desc));
    CHECK_CUSPARSE(cusparseLtDestroy(&handle));
    return compressed_size;
}

#endif

/* ************************* End of GEMM utils **************************** */

}  // end of namespace turbomind
