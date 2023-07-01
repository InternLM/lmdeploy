/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cublasINT8MMWrapper.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace turbomind {
cublasINT8MMWrapper::cublasINT8MMWrapper(cublasLtHandle_t cublaslt_handle,
                                         cudaStream_t     stream,
                                         cublasAlgoMap*   cublas_algo_map,
                                         std::mutex*      mu,
                                         bool             use_ORDER_COL32_2R_4R4):
    cublasMMWrapper(nullptr, cublaslt_handle, stream, cublas_algo_map, mu, nullptr),
    use_ORDER_COL32_2R_4R4_(use_ORDER_COL32_2R_4R4)
{
}

cublasINT8MMWrapper::cublasINT8MMWrapper(cublasHandle_t   cublas_handle,
                                         cublasLtHandle_t cublaslt_handle,
                                         cudaStream_t     stream,
                                         cublasAlgoMap*   cublas_algo_map,
                                         std::mutex*      mu,
                                         bool             use_ORDER_COL32_2R_4R4):
    cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, mu, nullptr),
    use_ORDER_COL32_2R_4R4_(use_ORDER_COL32_2R_4R4)
{
}

#ifdef SPARSITY_ENABLED
cublasINT8MMWrapper::cublasINT8MMWrapper(cublasLtHandle_t   cublaslt_handle,
                                         cusparseLtHandle_t cusparselt_handle,
                                         cudaStream_t       stream,
                                         cublasAlgoMap*     cublas_algo_map,
                                         std::mutex*        mu,
                                         bool               use_ORDER_COL32_2R_4R4):
    cublasMMWrapper(nullptr, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, mu, nullptr),
    use_ORDER_COL32_2R_4R4_(use_ORDER_COL32_2R_4R4)
{
}
#endif

cublasINT8MMWrapper::~cublasINT8MMWrapper()
{
    mu_ = nullptr;
}

cublasINT8MMWrapper::cublasINT8MMWrapper(const cublasINT8MMWrapper& wrapper):
#ifdef SPARSITY_ENABLED
    cublasMMWrapper(nullptr,
                    wrapper.cublaslt_handle_,
                    wrapper.cusparselt_handle_,
                    wrapper.stream_,
                    wrapper.cublas_algo_map_,
                    wrapper.mu_,
                    wrapper.allocator_),
#else
    cublasMMWrapper(
        nullptr, wrapper.cublaslt_handle_, wrapper.stream_, wrapper.cublas_algo_map_, wrapper.mu_, wrapper.allocator_),
#endif
    use_ORDER_COL32_2R_4R4_(wrapper.use_ORDER_COL32_2R_4R4_)
{
}

// for int8 cublasLtMM with algo
// ATransform should be m*n, CUBLASLT_ORDER_COL32
// kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
// res is m*n, CUBLASLT_ORDER_COL32
void cublasINT8MMWrapper::Gemm(int*          res,
                               int           batchCount,
                               int           m,
                               int           n,
                               int           k,
                               int64_t       stridea,
                               int64_t       strideb,
                               int64_t       stridec,
                               const int8_t* ATransform,
                               const int8_t* kernel)
{
    mu_->lock();
    cublasOperation_t opTranspose = CUBLAS_OP_T;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t computeType = CUDA_R_32I;
#endif
    cublasLtMatmulDesc_t   matmulDesc;
    cublasLtMatrixLayout_t AtransformDesc = NULL;
    cublasLtMatrixLayout_t BtransformDesc = NULL;
    cublasLtMatrixLayout_t CtransformDesc = NULL;
    cublasLtOrder_t        order_COL32    = CUBLASLT_ORDER_COL32;

    cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
    if (use_ORDER_COL32_2R_4R4_) {
        order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
    }
    else {
        order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
    }
#else
    order_matrixB              = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

    int ldaTransform = 32 * m;
    int ldbTransform;
    if (use_ORDER_COL32_2R_4R4_) {
        ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
    }
    else {
        ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
    }
    int ldcTransform = 32 * m;

    // create matmulDesc
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
    cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));
    cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
    cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform);
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    if (batchCount > 1) {
        cublasLtMatrixLayoutSetAttribute(
            AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(
            AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
        cublasLtMatrixLayoutSetAttribute(
            BtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(
            BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
        cublasLtMatrixLayoutSetAttribute(
            CtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(
            CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
    }

    int alphaI = 1;
    int betaI  = 0;

    // get algo
    cublasLtMatmulAlgo_t algo;
    int                  findAlgo = 0;
    if (cublas_algo_map_->isExist(batchCount, m, n, k, INT8_DATATYPE)) {
        // printf("find algo %s\n", markStr.c_str());
        findAlgo = 1;

        cublasLtMatmulAlgo_info tmp_info = cublas_algo_map_->getAlgo(batchCount, m, n, k, INT8_DATATYPE);

        cublasLtMatmulAlgoInit(cublaslt_handle_,
                               computeType,
                               CUDA_R_32I,
                               CUDA_R_8I,
                               CUDA_R_8I,
                               CUDA_R_32I,
                               CUDA_R_32I,
                               tmp_info.algoId,
                               &algo);
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(tmp_info.customOption), sizeof(tmp_info.customOption));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tmp_info.tile), sizeof(tmp_info.tile));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(tmp_info.splitK_val), sizeof(tmp_info.splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(tmp_info.swizzle), sizeof(tmp_info.swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(tmp_info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(tmp_info.stages), sizeof(tmp_info.stages));
#endif
    }
    else {
        findAlgo = 1;
        int algoId;
        if (use_ORDER_COL32_2R_4R4_) {
            algoId = 7;
        }
        else {
            algoId = 6;
        }
        int swizzle         = 0;
        int customOption    = 0;
        int tile            = 20;
        int splitK_val      = 0;
        int reductionScheme = 0;
        cublasLtMatmulAlgoInit(
            cublaslt_handle_, computeType, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
        int stages;
        if (use_ORDER_COL32_2R_4R4_) {
            stages = 15;
        }
        else {
            stages = 13;
        }
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif
    }

    cublasLtMatmul(cublaslt_handle_,
                   matmulDesc,
                   &alphaI,
                   ATransform,
                   AtransformDesc,
                   kernel,
                   BtransformDesc,
                   &betaI,
                   res,
                   CtransformDesc,
                   res,
                   CtransformDesc,
                   (findAlgo == 1 ? (&algo) : NULL),
                   NULL,
                   0,
                   stream_);

    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(AtransformDesc);
    cublasLtMatrixLayoutDestroy(BtransformDesc);
    cublasLtMatrixLayoutDestroy(CtransformDesc);
    sync_check_cuda_error();
    mu_->unlock();
}

// for int8 IO cublasLtMM with algo
// ATransform should be m*k CUBLASLT_ORDER_COL32
// kernel should be n*k CUBLASLT_ORDER_COL4_4R2_8C
// res is m*n CUBLASLT_ORDER_COL32
void cublasINT8MMWrapper::Gemm(int8_t*       res,
                               int           batchCount,
                               int           m,
                               int           n,
                               int           k,
                               int64_t       stridea,
                               int64_t       strideb,
                               int64_t       stridec,
                               const float   alpha,
                               const int8_t* ATransform,
                               const int8_t* kernel)
{
    mu_->lock();
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    // int8 gemm does not support CUBLAS_POINTER_MODE_DEVICE
    // cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    cudaDataType_t scaleType = CUDA_R_32F;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t computeType = CUDA_R_32I;
#endif
    cublasLtMatmulDesc_t   matmulDesc;
    cublasLtMatrixLayout_t AtransformDesc = NULL;
    cublasLtMatrixLayout_t BtransformDesc = NULL;
    cublasLtMatrixLayout_t CtransformDesc = NULL;
    cublasLtOrder_t        order_COL32    = CUBLASLT_ORDER_COL32;

    cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
    if (use_ORDER_COL32_2R_4R4_) {
        order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
    }
    else {
        order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
    }
#else
    order_matrixB              = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

    int ldaTransform = 32 * m;

    int ldbTransform;
    if (use_ORDER_COL32_2R_4R4_) {
        ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
    }
    else {
        ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
    }

    int ldcTransform = 32 * m;

    // create matmulDesc
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scaleType, sizeof(scaleType));
    // cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode,
    // sizeof(cublasLtPointerMode_t));
    cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
    cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    if (batchCount > 1) {
        cublasLtMatrixLayoutSetAttribute(
            AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(
            AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
        cublasLtMatrixLayoutSetAttribute(
            BtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(
            BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
        cublasLtMatrixLayoutSetAttribute(
            CtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
        cublasLtMatrixLayoutSetAttribute(
            CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
    }

    // get algo
    cublasLtMatmulAlgo_t algo;
    int                  findAlgo = 0;
    if (cublas_algo_map_->isExist(batchCount, m, n, k, INT8_DATATYPE)) {
        findAlgo = 1;

        cublasLtMatmulAlgo_info tmp_info = cublas_algo_map_->getAlgo(batchCount, m, n, k, INT8_DATATYPE);

        cublasLtMatmulAlgoInit(cublaslt_handle_,
                               computeType,
                               CUDA_R_32F,
                               CUDA_R_8I,
                               CUDA_R_8I,
                               CUDA_R_8I,
                               CUDA_R_8I,
                               tmp_info.algoId,
                               &algo);
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(tmp_info.customOption), sizeof(tmp_info.customOption));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tmp_info.tile), sizeof(tmp_info.tile));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(tmp_info.splitK_val), sizeof(tmp_info.splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(tmp_info.swizzle), sizeof(tmp_info.swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(tmp_info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(tmp_info.stages), sizeof(tmp_info.stages));
#endif
    }
    else {
        findAlgo = 1;
        int algoId;
        if (use_ORDER_COL32_2R_4R4_) {
            algoId = 7;
        }
        else {
            algoId = 6;
        }
        int swizzle         = 0;
        int customOption    = 0;
        int tile            = 20;
        int splitK_val      = 0;
        int reductionScheme = 0;
        cublasLtMatmulAlgoInit(
            cublaslt_handle_, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
        int stages;
        if (use_ORDER_COL32_2R_4R4_) {
            stages = 15;
        }
        else {
            stages = 13;
        }
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif
    }

    float beta = 0.0f;
    cublasLtMatmul(cublaslt_handle_,
                   matmulDesc,
                   &alpha,
                   ATransform,
                   AtransformDesc,
                   kernel,
                   BtransformDesc,
                   &beta,
                   res,
                   CtransformDesc,
                   res,
                   CtransformDesc,
                   (findAlgo == 1 ? (&algo) : NULL),
                   NULL,
                   0,
                   stream_);

    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(AtransformDesc);
    cublasLtMatrixLayoutDestroy(BtransformDesc);
    cublasLtMatrixLayoutDestroy(CtransformDesc);
    sync_check_cuda_error();
    mu_->unlock();
}

template<typename T>
int cublasINT8MMWrapper::getFusedINT8QKVType(const int k, const int n, const AttentionWeight<T>* attention_weights)
{

    int           fusedINT8QKV_type = 0;
    const int8_t* Q_weight          = (const int8_t*)(attention_weights->query_weight.kernel);
    const int8_t* K_weight          = (const int8_t*)(attention_weights->key_weight.kernel);
    const int8_t* V_weight          = (const int8_t*)(attention_weights->value_weight.kernel);
    // for QKV weight are DataType_ & continue
    if ((attention_weights->query_weight.kernel + n * k == attention_weights->key_weight.kernel)
        && (attention_weights->key_weight.kernel + n * k == attention_weights->value_weight.kernel)) {
        fusedINT8QKV_type = 1;
    }
    // for QVK weight are int8 & continue
    else if ((Q_weight + n * k == K_weight) && (K_weight + n * k == V_weight)) {
        fusedINT8QKV_type = 2;
    }
    return fusedINT8QKV_type;
}

bool cublasINT8MMWrapper::getUseOrderCol322R4R4()
{
    return use_ORDER_COL32_2R_4R4_;
}

template int
cublasINT8MMWrapper::getFusedINT8QKVType(const int k, const int n, const AttentionWeight<float>* attention_weights);

template int
cublasINT8MMWrapper::getFusedINT8QKVType(const int k, const int n, const AttentionWeight<half>* attention_weights);

#ifdef SPARSITY_ENABLED
// A is sparse weight [m,k], non transposed row major
// B is activation input [k, n], non transposed col major
void cublasINT8MMWrapper::SpGemm(
    const int m, const int n, const int k, const float alpha, const void* A, const void* B, void* C)
{
    cudaDataType_t                 Atype        = CUDA_R_8I;
    cudaDataType_t                 Btype        = CUDA_R_8I;
    cudaDataType_t                 Ctype        = CUDA_R_8I;
    cusparseComputeType            compute_type = CUSPARSE_COMPUTE_32I;
    cusparseOrder_t                col_order    = CUSPARSE_ORDER_COL;
    cusparseOrder_t                row_order    = CUSPARSE_ORDER_ROW;
    cusparseOperation_t            opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t            opB          = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;

    auto     num_A_rows = m;
    auto     num_A_cols = k;
    auto     num_B_rows = k;
    auto     num_B_cols = n;
    auto     num_C_rows = m;
    auto     num_C_cols = n;
    unsigned alignment  = 16;
    auto     lda        = num_A_cols;
    auto     ldb        = num_B_rows;
    auto     ldc        = num_C_rows;
    float    _beta(0.0f);

    char mark[256];
    sprintf(mark, "%d_%d_%d_%d", 1, m, n, k);
    if (sp_mat_A_desc_map_.find(mark) != sp_mat_A_desc_map_.end()) {
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&cusparselt_handle_,
                                                      &matmul,
                                                      opA,
                                                      opB,
                                                      &sp_mat_A_desc_map_[mark],
                                                      &sp_mat_B_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      compute_type))
    }
    else {
        // initializing MatDesc takes a lot of time
        cusparseLtMatDescriptor_t mat_A, mat_B, mat_C;
        sp_mat_A_desc_map_[mark] = mat_A;
        sp_mat_B_desc_map_[mark] = mat_B;
        sp_mat_C_desc_map_[mark] = mat_C;
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&cusparselt_handle_,
                                                          &sp_mat_A_desc_map_[mark],
                                                          num_A_rows,
                                                          num_A_cols,
                                                          lda,
                                                          alignment,
                                                          Atype,
                                                          row_order,
                                                          CUSPARSELT_SPARSITY_50_PERCENT))
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &cusparselt_handle_, &sp_mat_B_desc_map_[mark], num_B_rows, num_B_cols, ldb, alignment, Btype, col_order))
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &cusparselt_handle_, &sp_mat_C_desc_map_[mark], num_C_rows, num_C_cols, ldc, alignment, Ctype, col_order))
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&cusparselt_handle_,
                                                      &matmul,
                                                      opA,
                                                      opB,
                                                      &sp_mat_A_desc_map_[mark],
                                                      &sp_mat_B_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      &sp_mat_C_desc_map_[mark],
                                                      compute_type))
    }
    mu_->lock();
    CHECK_CUSPARSE(
        cusparseLtMatmulAlgSelectionInit(&cusparselt_handle_, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    int alg = cublas_algo_map_->getSpAlgo(1, num_A_rows, num_B_cols, num_A_cols);
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cusparselt_handle_, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&cusparselt_handle_, &alg_sel, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&cusparselt_handle_, &plan, &matmul, &alg_sel, workspace_size))

    void*        d_workspace = nullptr;
    int          num_streams = 1;
    cudaStream_t streams[1]  = {stream_};
    CHECK_CUSPARSE(
        cusparseLtMatmul(&cusparselt_handle_, &plan, &alpha, A, B, &_beta, C, C, d_workspace, streams, num_streams))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    sync_check_cuda_error();
    mu_->unlock();
}
#endif
}  // namespace turbomind
