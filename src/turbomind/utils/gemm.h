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

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

// TODO: Need to remove the dependency of the layer module.
//   e.g. refactor Weight class to some base module.
#include "src/turbomind/layers/DenseWeight.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasAlgoMap.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

// cublas default workspace size: 32MB. Let me make this as a Gemm property.
#define WORKSPACE_SIZE 33554432

namespace turbomind {

// A wrapper of cublas or cusparse matrix operator.
//  - GEMM_OP_N = CUBLAS_OP_N or CUSPARSE_OP_N
//  - GEMM_OP_T = CUBLAS_OP_T or CUSPARSE_OP_T
enum GemmOp {
    GEMM_OP_N,
    GEMM_OP_T
};

// A base class of the GEMM family.
// In the current version Gemm is as a base class as well as an interface.
class Gemm {

public:
    Gemm() = delete;  // Disable a default constructor
    /**
     * A Gemm class.
     *
     * NOTE:
     *   A, B, C are assumed to have a row major layout, while a backend cuda libraries
     *   assumes a column major layout. However, a family of Gemm has already handled
     *   such discrepancy internally. Please use naively without a trick like switching
     *   inputs A and B that aligns the matrix layout.
     *
     * Restriction: Supported in/out data or compute types: TYPE_FP16, TYPE_FP32.
     *
     * TODO:
     *   Unify resource allocation/release from a singleton GPU resource managers.
     *   Thus, allocator, stream can be replaced by a resource handler later.
     *   E.g. Gemm(std::shared_ptr<ResourceManager> resource_manager), and
     *        stream_ = resource_manager.getCudaStream();
     *        buffer = resource_manager.malloc(...);
     *
     * @param allocator   Resource allocator.
     * @param stream      A CUDA stream.
     * @param config_file A file path of a GEMM configuration.
     */
    Gemm(IAllocator* allocator, cudaStream_t stream, std::string config_file = GEMM_CONFIG);
    Gemm(Gemm const& other) = delete;
    virtual ~Gemm();

    virtual std::string toString();

    /**
     * @brief Set GEMM compute type.
     *
     * @param compute_type The data type of accumulation type inside GEMM computation.
     *                     (Choices: TYPE_FP16, TYPE_FP32)
     *
     * @throw GemmNotSupportedException if a type is not TYPE_FP16 or TYPE_FP32.
     * @throw std::runtime_error  if any exception inside CUDA.
     */
    void setComputeType(DataType compute_type);

    /**
     * @brief Set matrix data types and compute precision.
     *
     * Supported data or compute types: TYPE_FP16, TYPE_FP32
     *
     * @param a_type  The data type of a matrix A.
     * @param b_type  The data type of a matrix B.
     * @param c_type  The data type of a matrix C.
     * @param compute_type  The data type of accumulation type inside GEMM computation.
     *
     * @throw GemmNotSupportedException if a type is not TYPE_FP16 or TYPE_FP32.
     * @throw std::runtime_error  if any exception inside CUDA.
     */
    void setTypes(DataType a_type, DataType b_type, DataType c_type, DataType compute_type);

    /**
     * @brief Set matrix data and compute types by default values.
     *
     * Default configs:
     *  - T=float : data type=TYPE_FP32, compute type=TYPE_FP32
     *  - T=half  : data type=TYPE_FP16, compute type=TYPE_FP32
     */
    template<typename T>
    void setDefaultTypes();

    void loadGemmConfig(std::string config_file);

    void setAllocator(IAllocator* allocator);
    void setCudaStream(cudaStream_t& stream);

    // Th APIs below are to see how the interface will change
    // if it cooperates with Tensor. To enable it, we need to
    // update the Tensor class. For instance, data is need to
    // be of type (void*) rather than (const void*) to pass it
    // as the output C of gemm.
    // virtual void gemm(Tensor& C,
    //                   const GemmOp transa,
    //                   const GemmOp transb,
    //                   const Tensor& A,
    //                   const Tensor& B,
    //                   const float alpha = 1.0f,
    //                   const float beta = 0.0f);
    //
    // virtual void batchedMatmul(std::vector<Tensor> Carray,
    //                            const GemmOp transa,
    //                            const GemmOp transb,
    //                            const std::vector<Tensor> Aarray,
    //                            const std::vector<Tensor> Barray,
    //                            const float alpha = 1.0f,
    //                            const float beta = 0.0f);
    //
    // virtual void stridedBatchedGemm(Tensor& C,
    //                                 const GemmOp transa,
    //                                 const GemmOp transb,
    //                                 const Tensor& A,
    //                                 const Tensor& B,
    //                                 const float alpha = 1.0f,
    //                                 const float beta = 0.0f);

    // TODO:
    // This function cooperates with a Weight object to simply Gemm calls
    // inside layers, computing the following formula
    //     output(C) = input(A) * weight_kernel(B)
    // where weight_kernel can be changed according to Gemm functions.
    // DenseWeight is of a template struct, not allowing override the method.
    // We temperally add an interface here for two cases float/half,
    // but to finialze this function, we need an interface of a weight class
    // which is not a template class.
    virtual void gemm(const GemmOp              transa,
                      const GemmOp              transb,
                      const size_t              m,
                      const size_t              n,
                      const size_t              k,
                      const void*               input,
                      const DenseWeight<float>& weight,
                      void*                     output,
                      const float               alpha = 1.0f,
                      const float               beta  = 0.0f);
    virtual void gemm(const GemmOp             transa,
                      const GemmOp             transb,
                      const size_t             m,
                      const size_t             n,
                      const size_t             k,
                      const void*              input,
                      const DenseWeight<half>& weight,
                      void*                    output,
                      const float              alpha = 1.0f,
                      const float              beta  = 0.0f);

    virtual void gemm(const GemmOp transa,
                      const GemmOp transb,
                      const size_t m,
                      const size_t n,
                      const size_t k,
                      const void*  A,
                      const void*  B,
                      void*        C,
                      const float  alpha = 1.0f,
                      const float  beta  = 0.0f);

    virtual void gemm(const GemmOp transa,
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
                      const float  alpha = 1.0f,
                      const float  beta  = 0.0f);
    /**
     * @brief Compute the matrix multiplication `C = \alpha * op(A) * op(B) + \beta * C`.
     *
     * @param transa A transpose operation of a matrix A (GEMM_OP_N or GEMM_OP_T).
     * @param transb A transpose operation of a matrix B (GEMM_OP_N or GEMM_OP_T).
     * @param m      A number of rows of a matrix op(A) and C.
     * @param n      A number of columns of a matrix op(B) or C.
     * @param k      A number of columns of op(A) and rows of op(B).
     * @param A      A device pointer of a matrix A of dimension (m x lda).
     * @param Atype  A data type of A (TYPE_FP16 or TYPE_FP32)
     * @param lda    A leading dimension of the matrix A.
     * @param B      A device pointer of a matrix B of dimension (k x ldb).
     * @param Btype  A data type of B (TYPE_FP16 or TYPE_FP32)
     * @param ldb    A leading dimension of the matrix B.
     * @param C      (Output) A device pointer of a matrix C of dimension (m x ldc).
     * @param Ctype  A data type of C (TYPE_FP16 or TYPE_FP32)
     * @param ldc    A leading dimension of the matrix C.
     * @param alpha  A scale factor for A*B (default: 1.0f).
     * @param beta   A scale factor for C (default: 0.0f).
     *
     * @throw GemmNotSupportedException if a type is not TYPE_FP16 or TYPE_FP32.
     * @throw std::runtime_error  if any exception inside CUDA.
     */
    virtual void gemm(const GemmOp   transa,
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
                      const float    alpha = 1.0f,
                      const float    beta  = 0.0f);

    virtual void batchedGemm(const GemmOp       transa,
                             const GemmOp       transb,
                             const size_t       m,
                             const size_t       n,
                             const size_t       k,
                             const void* const* A,
                             const void* const* B,
                             void* const*       C,
                             const size_t       batch_size,
                             const float        alpha = 1.0f,
                             const float        beta  = 0.0f);

    virtual void batchedGemm(const GemmOp       transa,
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
                             const float        alpha = 1.0f,
                             const float        beta  = 0.0f);

    /**
     * @brief Compute the matrix multiplication of batch of matrices As and Bs
     *
     * For input batch A[i]/B[i] and output batch C[i], i = 0, ..., batch_size - 1,
     *  `C[i] = \alpha * op(A[i]) * op(B[i]) + \beta * C[i]`.
     *
     * @param transa A transpose operation of a matrix A (GEMM_OP_N or GEMM_OP_T).
     * @param transb A transpose operation of a matrix B (GEMM_OP_N or GEMM_OP_T).
     * @param m      A number of rows of a matrix op(A) and C.
     * @param n      A number of columns of a matrix op(B) or C.
     * @param k      A number of columns of op(A) and rows of op(B).
     * @param A      An array of device pointers of batch of input A matrices.
     * @param Atype  A data type of A (TYPE_FP16 or TYPE_FP32)
     * @param lda    A leading dimension of the matrix A.
     * @param B      An array of device pointers of batch of input B matrices.
     * @param Btype  A data type of B (TYPE_FP16 or TYPE_FP32)
     * @param ldb    A leading dimension of the matrix B.
     * @param C      (Output) An array of device pointers of batch of output C matrices.
     * @param Ctype  A data type of C (TYPE_FP16 or TYPE_FP32)
     * @param ldc    A leading dimension of the matrix C.
     * @param alpha  A scale factor for A*B (default: 1.0f).
     * @param beta   A scale factor for C (default: 0.0f).
     *
     * @throw GemmNotSupportedException if a type is not TYPE_FP16 or TYPE_FP32.
     * @throw std::runtime_error  if any exception inside CUDA.
     */
    virtual void batchedGemm(const GemmOp       transa,
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
                             const float        alpha = 1.0f,
                             const float        beta  = 0.0f);

    virtual void stridedBatchedGemm(GemmOp       transa,
                                    GemmOp       transb,
                                    const size_t m,
                                    const size_t n,
                                    const size_t k,
                                    const void*  A,
                                    const void*  B,
                                    void*        C,
                                    const size_t batch_size,
                                    const float  alpha = 1.0f,
                                    const float  beta  = 0.0f);

    virtual void stridedBatchedGemm(GemmOp        transa,
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
                                    const float   alpha = 1.0f,
                                    const float   beta  = 0.0f);

    virtual void stridedBatchedGemm(GemmOp        transa,
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
                                    const float   alpha = 1.0f,
                                    const float   beta  = 0.0f);
    /**
     * @brief Compute the strided matrix multiplication of batch of matrices As and Bs
     *
     * For input batch A[i]/B[i] and output batch C[i], i = 0, ..., batch_size - 1,
     *  `C[i] = \alpha * op(A[i]) * op(B[i]) + \beta * C[i]`.
     *
     * @param transa   A transpose operation of a matrix A (GEMM_OP_N or GEMM_OP_T).
     * @param transb   A transpose operation of a matrix B (GEMM_OP_N or GEMM_OP_T).
     * @param m        A number of rows of a matrix op(A) and C.
     * @param n        A number of columns of a matrix op(B) or C.
     * @param k        A number of columns of op(A) and rows of op(B).
     * @param A        An array of device pointers of batch of input A matrices.
     * @param Atype    A data type of A (TYPE_FP16 or TYPE_FP32)
     * @param lda      A leading dimension of the matrix A.
     * @param strideA  An offset in number of elements between matrix A[i] and A[i+1].
     * @param B        An array of device pointers of batch of input B matrices.
     * @param Btype    A data type of B (TYPE_FP16 or TYPE_FP32)
     * @param ldb      A leading dimension of the matrix B.
     * @param strideB  An offset in number of elements between matrix B[i] and B[i+1].
     * @param C        (Output) An array of device pointers of batch of output C matrices.
     * @param Ctype    A data type of C (TYPE_FP16 or TYPE_FP32)
     * @param ldc      A leading dimension of the matrix C.
     * @param strideC  An offset in number of elements between matrix C[i] and C[i+1].
     * @param compute_type  An accumulation type of GEMM.
     * @param alpha    A scale factor for A*B (default: 1.0f).
     * @param beta     A scale factor for C (default: 0.0f).
     *
     * @throw GemmNotSupportedException if a type is not TYPE_FP16 or TYPE_FP32.
     * @throw std::runtime_error  if any exception inside CUDA.
     */
    virtual void stridedBatchedGemm(GemmOp        transa,
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
                                    const float   alpha = 1.0f,
                                    const float   beta  = 0.0f);

protected:
    IAllocator*    allocator_ = nullptr;
    cudaStream_t   stream_;
    std::mutex*    mutex_           = nullptr;
    cublasAlgoMap* cublas_algo_map_ = nullptr;

    cublasHandle_t   cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    void*            workspace_ = nullptr;

    // use FP32 as default
    DataType a_type_       = TYPE_FP32;
    DataType b_type_       = TYPE_FP32;
    DataType c_type_       = TYPE_FP32;
    DataType compute_type_ = TYPE_FP32;

    // Check if data and inputs are valid in the Gemm class.
    virtual void checkDataTypeValidity(const DataType& type);
};

// class Int8Gemm : public Gemm {

// protected:
//     bool use_ORDER_COL32_2R_4R4_; // what is this?
// };

#ifdef SPARSITY_ENABLED

/**
 * A Sparse Gemm class.
 *
 * NOTE:
 *   A, B, C are assumed to have a row major layout.
 *   There are two restrictions:
 *   - It supports the case when the matrix B is sparse.
 *   - Supported only TYPE_FP16 for in/out data or compute types.
 */
class SpGemm: public Gemm {

protected:
    cusparseLtHandle_t                               cusparselt_handle_;
    std::map<std::string, cusparseLtMatDescriptor_t> a_desc_map_;
    std::map<std::string, cusparseLtMatDescriptor_t> b_desc_map_;
    std::map<std::string, cusparseLtMatDescriptor_t> c_desc_map_;
    bool                                             useBaseGemm(size_t batch_size, size_t m, size_t n, size_t k);

public:
    using Gemm::setComputeType;
    using Gemm::setTypes;
    using Gemm::setDefaultTypes;
    using Gemm::setAllocator;
    using Gemm::setCudaStream;
    using Gemm::gemm;
    using Gemm::batchedGemm;
    using Gemm::stridedBatchedGemm;

    /**
     * @param allocator   Resource allocator.
     * @param stream      A CUDA stream.
     * @param config_file A file path of a GEMM configuration.
     */
    // TODO: Let's unify algo map loading part.
    SpGemm(IAllocator*  allocator,
           cudaStream_t stream,
           std::string  config_file   = GEMM_CONFIG,
           std::string  spconfig_file = SPGEMM_CONFIG);
    ~SpGemm();
    std::string toString() override;
    void        loadGemmConfig(std::string config_file, std::string spconfig_file);

    // Template method cannot be overridden.
    void gemm(const GemmOp              transa,
              const GemmOp              transb,
              const size_t              m,
              const size_t              n,
              const size_t              k,
              const void*               input,
              const DenseWeight<float>& weight,
              void*                     output,
              const float               alpha = 1.0f,
              const float               beta  = 0.0f) override;
    void gemm(const GemmOp             transa,
              const GemmOp             transb,
              const size_t             m,
              const size_t             n,
              const size_t             k,
              const void*              input,
              const DenseWeight<half>& weight,
              void*                    output,
              const float              alpha = 1.0f,
              const float              beta  = 0.0f) override;

    void gemm(const GemmOp   transa,
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
              const float    alpha = 1.0f,
              const float    beta  = 0.0f) override;

private:
    void checkDataTypeValidity(const DataType& type) override;

    // Temporal gemm helper mtehod to use template T.
    template<typename T>
    void weightGemmHelper(const GemmOp          transa,
                          const GemmOp          transb,
                          const size_t          m,
                          const size_t          n,
                          const size_t          k,
                          const void*           input,
                          const DenseWeight<T>& weight,
                          void*                 output,
                          const float           alpha,
                          const float           beta);
};

// class Int8SpGemm : public Int8Gemm, public SpGemm {

// };
#endif

/* ***************************** GEMM Exceptions ******************************* */

class GemmInvalidShapeException: public std::exception {
private:
    std::string msg_ = "Invalid matrix shapes.";

public:
    explicit GemmInvalidShapeException() = default;

    template<typename... Args>
    explicit GemmInvalidShapeException(const std::string format, const Args&... args): msg_(fmtstr(format, args...))
    {
    }

    const char* what() const throw()
    {
        return msg_.c_str();
    }
};

class GemmNotSupportedException: public std::exception {
private:
    std::string msg_ = "Not supported exception.";

public:
    explicit GemmNotSupportedException() = default;

    template<typename... Args>
    explicit GemmNotSupportedException(const std::string format, const Args&... args): msg_(fmtstr(format, args...))
    {
    }

    const char* what() const throw()
    {
        return msg_.c_str();
    }
};

class GemmInvalidException: public std::exception {
private:
    std::string msg_ = "Invalid use of gemm.";

public:
    explicit GemmInvalidException() = default;

    template<typename... Args>
    explicit GemmInvalidException(const std::string format, const Args&... args): msg_(fmtstr(format, args...))
    {
    }

    const char* what() const throw()
    {
        return msg_.c_str();
    }
};

/* ************************ End of GEMM Exceptions ************************ */

/* ***************************** GEMM utils ******************************* */

/**
 * @brief Create method for the Gemm family.
 *
 * @param allocator  Resource allocator.
 * @param stream     A CUDA stream.
 * @param sparse     Whether to use sparse GEMM
 * @param quantized  Whether to use int8 quantized GEMM.
 * @return A shared pointer of a GemmCls instance.
 */
std::shared_ptr<Gemm>
createGemm(IAllocator* allocator, cudaStream_t stream, bool sparse = false, bool quantized = false);

cudaDataType_t getCublasDataType(DataType dtype);
#if (CUDART_VERSION >= 11000)
cublasComputeType_t getCublasComputeType(DataType dtype);
#else
cudaDataType_t getCublasComputeType(DataType dtype);
#endif
cublasOperation_t getCublasOperation(GemmOp op);
std::string       getGemmOpString(const GemmOp& op);

#ifdef SPARSITY_ENABLED
cusparseOperation_t getCusparseOperation(GemmOp op);
cusparseComputeType getCusparseComputeType(DataType dtype);

/**
 * @brief Prune a weight matrix (in-place).
 *
 * SpGemm supports a case when the sparse matrix is B in C=A*B.
 *
 * @param data    A data pointer
 * @param stream  A cuda stream object.
 * @param k       A number of rows of op(B).
 * @param n       A number of columns of op(B).
 * @param trans   A transpose operation that will be applied to the matrix
 *                (default: GEMM_OP_N).
 */
void pruneMatrixB(
    void* data, const cudaStream_t& stream, const size_t k, const size_t n, const GemmOp trans = GEMM_OP_N);

/**
 * @brief Compress the B matrix in a specific sparsity format.
 *
 * @param output A pointer where to allocate memory buffer to store a compressed matrix.
 * @param alloactor  A resource allocator.
 * @param stream A cuda stream object.
 * @param input  An input matrix to compress.
 * @param k      A number of rows of op(B).
 * @param n      A number of columns of op(B).
 * @param trans  A transpose operation that will be applied to the matrix (default: GEMM_OP_N).
 *
 * @return A size of the allocated device buffer of the compressed matrix.
 *
 * @throw GemmInvalidException  if the input matrix does not have 2:4 sparsity.
 *              or if fail to compute a correct buffer size to store the compressed matrix.
 * @throw std::runtime_error  if any exception inside CUDA.
 */
size_t compressMatrixB(void**              output,
                       IAllocator&         allocator,
                       const cudaStream_t& stream,
                       const void*         input,
                       const size_t        k,
                       const size_t        n,
                       const GemmOp        trans = GEMM_OP_N);

#endif

/* ************************* End of GEMM utils **************************** */

}  // end of namespace turbomind
