/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "encoder_igemm_func.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace fastertransformer {

int batch_size_;
int seq_len_;
int head_num_;
int size_per_head_;

static const char* showStatus(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

// Utility function to print customMatmulPerf_t structure
int printPerfStructure(int m, int n, int k, const customMatmulPerf_t& perf, FILE* fout, int hasPrint)
{
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;

    const cublasLtMatmulAlgo_t* matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
#else
    stages                     = 0;
#endif

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d} status %d "
           "time %f workspace=%d mathMode=%d waves=%f\n",
           algoId,
           tile,
           matmulTileName[tile],
           numSplitsK,
           reductionScheme,
           swizzle,
           customOption,
           stages,
           perf.status,
           perf.time,
           (int)perf.workspaceSize,
           (int)perf.mathMode,
           perf.wavesCount);

    // chose the fastest algo that does not need workspace
    if ((int)perf.workspaceSize == 0 && hasPrint == 0) {
        fprintf(fout,
                "%d %d %d %d %d ### 1 %d %d %d %d %d %d %d %d %d %d %d %f\n",
                batch_size_,
                seq_len_,
                head_num_,
                size_per_head_,
                INT8_DATATYPE,
                m,
                n,
                k,
                algoId,
                customOption,
                tile,
                numSplitsK,
                swizzle,
                reductionScheme,
                (int)perf.workspaceSize,
                stages,
                perf.time);
        return 1;
    }
    else {
        return hasPrint;
    }
}

int printBatchPerfStructure(
    int batchCount, int m, int n, int k, const customMatmulPerf_t& perf, FILE* fout, int hasPrint)
{
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;

    const cublasLtMatmulAlgo_t* matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
#else
    stages                     = 0;
#endif

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d} status %d "
           "time %f workspace=%d mathMode=%d waves=%f\n",
           algoId,
           tile,
           matmulTileName[tile],
           numSplitsK,
           reductionScheme,
           swizzle,
           customOption,
           stages,
           perf.status,
           perf.time,
           (int)perf.workspaceSize,
           (int)perf.mathMode,
           perf.wavesCount);

    // chose the fastest algo that does not need workspace
    if ((int)perf.workspaceSize == 0 && hasPrint == 0) {
        fprintf(fout,
                "%d %d %d %d %d ### %d %d %d %d %d %d %d %d %d %d %d %d %f\n",
                batch_size_,
                seq_len_,
                head_num_,
                size_per_head_,
                INT8_DATATYPE,
                batchCount,
                m,
                n,
                k,
                algoId,
                customOption,
                tile,
                numSplitsK,
                swizzle,
                reductionScheme,
                (int)perf.workspaceSize,
                stages,
                perf.time);
        return 1;
    }
    else {
        return hasPrint;
    }
}

static inline bool time_compare(const customMatmulPerf_t& perf_a, const customMatmulPerf_t& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t            ltHandle,  // to get the capabilities (required a GPU)
                                      cublasLtMatmulDesc_t        operationDesc,
                                      const void*                 alpha, /* host or device pointer */
                                      const void*                 A,
                                      cublasLtMatrixLayout_t      Adesc,
                                      const void*                 B,
                                      cublasLtMatrixLayout_t      Bdesc,
                                      const void*                 beta, /* host or device pointer */
                                      const void*                 C,
                                      cublasLtMatrixLayout_t      Cdesc,
                                      void*                       D,
                                      cublasLtMatrixLayout_t      Ddesc,
                                      const cublasLtMatmulAlgo_t& algo,
                                      int                         kernelRepeats,
                                      void*                       workSpace,
                                      size_t                      workSpaceSizeInBytes,
                                      customMatmulPerf_t&         perfResults,
                                      cudaStream_t                stream)
{
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int            repeats = kernelRepeats;
    cublasStatus_t algoStatus =
        cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);
    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            struct timeval start, end;
            cublasStatus_t oneRunStatus;
            cudaDeviceSynchronize();
            gettimeofday(&start, NULL);
            for (int loop = 0; loop < repeats; loop++) {
                oneRunStatus = cublasLtMatmul(ltHandle,
                                              operationDesc,
                                              alpha,
                                              A,
                                              Adesc,
                                              B,
                                              Bdesc,
                                              beta,
                                              C,
                                              Cdesc,
                                              D,
                                              Ddesc,
                                              &algo,
                                              workSpace,
                                              workSpaceSizeInBytes,
                                              stream);
            }
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                algoStatus = oneRunStatus;
            }
            float time = diffTime(start, end);
            // For the moment only add successful findings
            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                perfResults.algo          = algo;
                perfResults.time          = time / repeats;
                perfResults.workspaceSize = heurResult.workspaceSize;
                perfResults.wavesCount    = heurResult.wavesCount;
            }
        }
        else {
            // printf("not enough workspace! %ld\n", heurResult.workspaceSize);
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
        }
    }
    else {
        // printf("check fail!\n");
    }
    return algoStatus;
}

// Sample wrapper running through multiple algo and config attributes combination for INT8 gemm using cublasLt low-level
// API
template<typename T, typename scaleT>
int LtIgemmCustomFind(cublasLtHandle_t ltHandle,
                      int              m,
                      int              n,
                      int              k,
                      const scaleT*    alpha, /* host pointer */
                      const int8_t*    A,
                      const int8_t*    B,
                      const scaleT*    beta, /* host pointer */
                      T*               C,
                      void*            workSpace,
                      size_t           workSpaceSize,
                      FILE*            fout)
{
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaStream_t           stream = 0;
    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    // Let try a fixed number of combinations
#define ALGO_COMBINATIONS 50000
    int                AlgoCombinations = ALGO_COMBINATIONS;
    int                AlgoCount        = 0;
    int                kernelRepeats    = 100;  // number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    int                nbAlgoIds = 0;
#define ALGO_IDS 100
    int algoIdA[ALGO_IDS];

    cudaDataType_t Atype, Btype, Ctype, scaleType;
    Atype = CUDA_R_8I;
    Btype = CUDA_R_8I;

    if (std::is_same<T, int32_t>::value && std::is_same<scaleT, int>::value) {
        Ctype     = CUDA_R_32I;
        scaleType = CUDA_R_32I;
    }
    else if (std::is_same<T, int8_t>::value && std::is_same<scaleT, float>::value) {
        Ctype     = CUDA_R_8I;
        scaleType = CUDA_R_32F;
    }
    else {
        printf("[ERROR]<T,scaleT> of igemm is invalid\n");
        exit(-1);
    }

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t computeType = CUDA_R_32I;
#endif
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    bool use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    int device{-1};
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    if (props.major * 10 + props.minor >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
    if (use_ORDER_COL32_2R_4R4) {
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
    if (use_ORDER_COL32_2R_4R4) {
        ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
    }
    else {
        ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
    }

    int ldcTransform = 32 * m;

#if (CUDART_VERSION >= 11000)
    status = cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    status                     = cublasLtMatmulDescCreate(&operationDesc, scaleType);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));

    // Create matrix descriptors.
    status = cublasLtMatrixLayoutCreate(&Adesc, Atype, m, k, ldaTransform);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutCreate(&Bdesc, Btype, n, k, ldbTransform);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status =
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldcTransform);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    // Request AlgoId available for IGEMM
    status = cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {
        cublasLtMatmulAlgo_t algo;
        size_t               sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status =
            cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int  nbTiles = int(sizeWritten / sizeof(int));
        int* tileA   = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles  = 1;
        }
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
        int              nbStages = int(sizeWritten / sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages   = 1;
        }
        else {
            cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int) * nbStages, &sizeWritten);
        }
#endif
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);
        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
#if (CUDART_VERSION >= 11000)
            /* Loop over different stages count */
            for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx]));
#endif
                /* Loop over the different custom option if any */
                for (int customOption = 0; customOption <= customOptionMax; customOption++) {
                    cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                    /* Loop over the CTAs swizzling support */
                    for (int k = 0; k <= swizzlingMax; k++) {
                        int splitK_trial = 0;
                        if (splitkSupport) {
                            splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                        }
                        // Loop over the splitK value over a fixed sequence splitKSequenceA in addition to the case
                        // where splitK is not enabled
                        for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                            /* Setup attribute of the algo to run */
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                            int splitK_val = 0;
                            int redScheme  = CUBLASLT_REDUCTION_SCHEME_NONE;
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));

                            if (l > 0) {  // Split-K case
                                splitK_val = splitKSequenceA[l - 1];
                                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                     &splitKSequenceA[l - 1],
                                                                     sizeof(splitKSequenceA[l - 1]));
                                /* Going over all the reduction scheme  */
                                for (redScheme = 1;
                                     redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations);
                                     redScheme = redScheme << 1) {
                                    if (redScheme & redMask) {
                                        cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                             &redScheme,
                                                                             sizeof(redScheme));
                                        status                        = customMatmulRun(ltHandle,
                                                                 operationDesc,
                                                                 alpha, /* host or device pointer */
                                                                 A,
                                                                 Adesc,
                                                                 B,
                                                                 Bdesc,
                                                                 beta, /* host or device pointer */
                                                                 C,
                                                                 Cdesc,
                                                                 C,
                                                                 Cdesc,
                                                                 algo,
                                                                 kernelRepeats,
                                                                 workSpace,
                                                                 workSpaceSize,
                                                                 perfResults[AlgoCount],
                                                                 stream);
                                        perfResults[AlgoCount].status = status;
                                        if (status == CUBLAS_STATUS_SUCCESS) {
                                            AlgoCount++;
                                        }
                                    }  // end if
                                }      // end for
                            }
                            else {  // Non-splitK case
                                /* if user preference is ok with workspace */
                                if (AlgoCount < AlgoCombinations) {
                                    status                        = customMatmulRun(ltHandle,
                                                             operationDesc,
                                                             alpha, /* host or device pointer */
                                                             A,
                                                             Adesc,
                                                             B,
                                                             Bdesc,
                                                             beta, /* host or device pointer */
                                                             C,
                                                             Cdesc,
                                                             C,
                                                             Cdesc,
                                                             algo,
                                                             kernelRepeats,
                                                             workSpace,
                                                             workSpaceSize,
                                                             perfResults[AlgoCount],
                                                             stream);
                                    perfResults[AlgoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS) {
                                        AlgoCount++;
                                    }
                                }
                            }
                        }  // end l
                    }      // end k
                }          // end customOption
#if (CUDART_VERSION >= 11000)
            }  // end stagesIdx
#endif
        }  // end tileIdx
        delete[] tileA;
    }  // end idx
    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details
    for (int i = 0, hasPrint = 0; i < AlgoCount; i++) {
        printf("result %03d : ", i);
        hasPrint = printPerfStructure(m, n, k, perfResults[i], fout, hasPrint);
    }

CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) {
        cublasLtMatrixLayoutDestroy(Cdesc);
    }
    if (Bdesc) {
        cublasLtMatrixLayoutDestroy(Bdesc);
    }
    if (Adesc) {
        cublasLtMatrixLayoutDestroy(Adesc);
    }
    if (operationDesc) {
        cublasLtMatmulDescDestroy(operationDesc);
    }
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

template int LtIgemmCustomFind(cublasLtHandle_t ltHandle,
                               int              m,
                               int              n,
                               int              k,
                               const int*       alpha, /* host pointer */
                               const int8_t*    A,
                               const int8_t*    B,
                               const int*       beta, /* host pointer */
                               int32_t*         C,
                               void*            workSpace,
                               size_t           workSpaceSize,
                               FILE*            fout);

template int LtIgemmCustomFind(cublasLtHandle_t ltHandle,
                               int              m,
                               int              n,
                               int              k,
                               const float*     alpha, /* host pointer */
                               const int8_t*    A,
                               const int8_t*    B,
                               const float*     beta, /* host pointer */
                               int8_t*          C,
                               void*            workSpace,
                               size_t           workSpaceSize,
                               FILE*            fout);

template<typename T, typename scaleT>
int LtBatchIgemmCustomFind(cublasLtHandle_t ltHandle,
                           int              batchCount,
                           int              m,
                           int              n,
                           int              k,
                           const scaleT*    alpha, /* host pointer */
                           const int8_t*    A,
                           const int8_t*    B,
                           const scaleT*    beta, /* host pointer */
                           T*               C,
                           void*            workSpace,
                           size_t           workSpaceSize,
                           FILE*            fout)
{
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaStream_t           stream = 0;
    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    // Let try a fixed number of combinations
#define ALGO_COMBINATIONS 50000
    int                AlgoCombinations = ALGO_COMBINATIONS;
    int                AlgoCount        = 0;
    int                kernelRepeats    = 100;  // number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    int                nbAlgoIds = 0;
#define ALGO_IDS 100
    int algoIdA[ALGO_IDS];

    cudaDataType_t Atype, Btype, Ctype, scaleType;
    Atype = CUDA_R_8I;
    Btype = CUDA_R_8I;

    if (std::is_same<T, int32_t>::value && std::is_same<scaleT, int>::value) {
        Ctype     = CUDA_R_32I;
        scaleType = CUDA_R_32I;
    }
    else if (std::is_same<T, int8_t>::value && std::is_same<scaleT, float>::value) {
        Ctype     = CUDA_R_8I;
        scaleType = CUDA_R_32F;
    }
    else {
        printf("[ERROR]<T,scaleT> of igemm is invalid\n");
        exit(-1);
    }

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
    cudaDataType_t computeType = CUDA_R_32I;
#endif
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    bool use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    int device{-1};
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    if (props.major * 10 + props.minor >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
    if (use_ORDER_COL32_2R_4R4) {
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
    if (use_ORDER_COL32_2R_4R4) {
        ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
    }
    else {
        ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
    }

    int ldcTransform = 32 * m;

    int64_t stridea, strideb, stridec;
    stridea = m * k;
    strideb = n * k;
    stridec = m * n;

#if (CUDART_VERSION >= 11000)
    status = cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    status                     = cublasLtMatmulDescCreate(&operationDesc, scaleType);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));

    // Create matrix descriptors.
    status = cublasLtMatrixLayoutCreate(&Adesc, Atype, m, k, ldaTransform);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));

    status = cublasLtMatrixLayoutCreate(&Bdesc, Btype, n, k, ldbTransform);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status =
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));

    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldcTransform);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));

    // Request AlgoId available for IGEMM
    status = cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {
        cublasLtMatmulAlgo_t algo;
        size_t               sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status =
            cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int  nbTiles = int(sizeWritten / sizeof(int));
        int* tileA   = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles  = 1;
        }
#if (CUDART_VERSION >= 11000)
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
        int              nbStages = int(sizeWritten / sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages   = 1;
        }
        else {
            cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int) * nbStages, &sizeWritten);
        }
#endif
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);
        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
#if (CUDART_VERSION >= 11000)
            /* Loop over different stages count */
            for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx]));
#endif
                /* Loop over the different custom option if any */
                for (int customOption = 0; customOption <= customOptionMax; customOption++) {
                    cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                    /* Loop over the CTAs swizzling support */
                    for (int k = 0; k <= swizzlingMax; k++) {
                        int splitK_trial = 0;
                        if (splitkSupport) {
                            splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                        }
                        // Loop over the splitK value over a fixed sequence splitKSequenceA in addition to the case
                        // where splitK is not enabled
                        for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                            /* Setup attribute of the algo to run */
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                            int splitK_val = 0;
                            int redScheme  = CUBLASLT_REDUCTION_SCHEME_NONE;
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));

                            if (l > 0) {  // Split-K case
                                splitK_val = splitKSequenceA[l - 1];
                                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                     &splitKSequenceA[l - 1],
                                                                     sizeof(splitKSequenceA[l - 1]));
                                /* Going over all the reduction scheme  */
                                for (redScheme = 1;
                                     redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations);
                                     redScheme = redScheme << 1) {
                                    if (redScheme & redMask) {
                                        cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                             &redScheme,
                                                                             sizeof(redScheme));
                                        status                        = customMatmulRun(ltHandle,
                                                                 operationDesc,
                                                                 alpha, /* host or device pointer */
                                                                 A,
                                                                 Adesc,
                                                                 B,
                                                                 Bdesc,
                                                                 beta, /* host or device pointer */
                                                                 C,
                                                                 Cdesc,
                                                                 C,
                                                                 Cdesc,
                                                                 algo,
                                                                 kernelRepeats,
                                                                 workSpace,
                                                                 workSpaceSize,
                                                                 perfResults[AlgoCount],
                                                                 stream);
                                        perfResults[AlgoCount].status = status;
                                        if (status == CUBLAS_STATUS_SUCCESS) {
                                            AlgoCount++;
                                        }
                                    }  // end if
                                }      // end for
                            }
                            else {  // Non-splitK case
                                /* if user preference is ok with workspace */
                                if (AlgoCount < AlgoCombinations) {
                                    status                        = customMatmulRun(ltHandle,
                                                             operationDesc,
                                                             alpha, /* host or device pointer */
                                                             A,
                                                             Adesc,
                                                             B,
                                                             Bdesc,
                                                             beta, /* host or device pointer */
                                                             C,
                                                             Cdesc,
                                                             C,
                                                             Cdesc,
                                                             algo,
                                                             kernelRepeats,
                                                             workSpace,
                                                             workSpaceSize,
                                                             perfResults[AlgoCount],
                                                             stream);
                                    perfResults[AlgoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS) {
                                        AlgoCount++;
                                    }
                                }
                            }
                        }  // end l
                    }      // end k
                }          // end customOption
#if (CUDART_VERSION >= 11000)
            }  // end stagesIdx
#endif
        }  // end tileIdx
        delete[] tileA;
    }  // end idx
    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details
    for (int i = 0, hasPrint = 0; i < AlgoCount; i++) {
        printf("result %03d : ", i);
        hasPrint = printBatchPerfStructure(batchCount, m, n, k, perfResults[i], fout, hasPrint);
    }

CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) {
        cublasLtMatrixLayoutDestroy(Cdesc);
    }
    if (Bdesc) {
        cublasLtMatrixLayoutDestroy(Bdesc);
    }
    if (Adesc) {
        cublasLtMatrixLayoutDestroy(Adesc);
    }
    if (operationDesc) {
        cublasLtMatmulDescDestroy(operationDesc);
    }
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

template int LtBatchIgemmCustomFind(cublasLtHandle_t ltHandle,
                                    int              batchCount,
                                    int              m,
                                    int              n,
                                    int              k,
                                    const int*       alpha, /* host pointer */
                                    const int8_t*    A,
                                    const int8_t*    B,
                                    const int*       beta, /* host pointer */
                                    int32_t*         C,
                                    void*            workSpace,
                                    size_t           workSpaceSize,
                                    FILE*            fout);

template int LtBatchIgemmCustomFind(cublasLtHandle_t ltHandle,
                                    int              batchCount,
                                    int              m,
                                    int              n,
                                    int              k,
                                    const float*     alpha, /* host pointer */
                                    const int8_t*    A,
                                    const int8_t*    B,
                                    const float*     beta, /* host pointer */
                                    int8_t*          C,
                                    void*            workSpace,
                                    size_t           workSpaceSize,
                                    FILE*            fout);

// initialize matrix in column-major
void matInit(int rows, int cols, int8_t* p, int ld)
{
    srand(time(NULL));

    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            int index = r + c * ld;

            p[index] = rand() % 255 - 127;
        }
    }
}

int batch_igemm_config(int batchCount, int m, int n, int k, FILE* fout, void* buffer)
{
    printf("batchCount %d m %d n %d k %d\n", batchCount, m, n, k);
    int alpha = 1;
    int beta  = 0;

    int8_t*  d_A = (int8_t*)buffer;                       // m * k, stored in column-major
    int8_t*  d_B = d_A + batchCount * m * k;              // k * n, stored in column-major
    int32_t* d_C = (int32_t*)(d_B + batchCount * k * n);  // m * n, stored in column-major

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    LtBatchIgemmCustomFind(ltHandle,
                           batchCount,
                           m,
                           n,
                           k,
                           &alpha, /* host pointer */
                           d_A,
                           d_B,
                           &beta, /* host pointer */
                           d_C,
                           NULL,
                           0,
                           fout);
    // free memory
    cublasLtDestroy(ltHandle);
    return 0;
}

int igemm_config(int m, int n, int k, FILE* fout, void* buffer)
{
    printf("batchCount %d m %d n %d k %d\n", 1, m, n, k);
    int alpha = 1;
    int beta  = 0;

    int8_t*  d_A = (int8_t*)buffer;          // m * k, stored in column-major
    int8_t*  d_B = d_A + m * k;              // k * n, stored in column-major
    int32_t* d_C = (int32_t*)(d_B + k * n);  // m * n, stored in column-major

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    LtIgemmCustomFind(ltHandle,
                      m,
                      n,
                      k,
                      &alpha, /* host pointer */
                      d_A,
                      d_B,
                      &beta, /* host pointer */
                      d_C,
                      NULL,
                      0,
                      fout);

    cublasLtDestroy(ltHandle);
    return 0;
}

int generate_encoder_igemm_config(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend)
{

    // ensure program running on SM >= 7.5
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    if (!(prop.major >= 8 || (prop.major >= 7 && prop.minor >= 5))) {
        printf("[ERROR] INT8 mode > 0 is only supported on device with sm >= 7.5\n ");
        exit(-1);
    }
    printf("Device %s\n", prop.name);

    // check config
    FILE* fout;
    if (!isAppend) {
        fout = fopen(IGEMM_CONFIG, "w+");
        fprintf(
            fout,
            "batch_size seq_len head_num size_per_head dataType ### batchCount m n k algoId customOption tile splitK_val swizzle reductionScheme workspaceSize stages exec_time\n");
    }
    else {
        fout = fopen(IGEMM_CONFIG, "a+");
        std::vector<std::string> config;
        char                     line[1024];
        while (fgets(line, 1024, fout) != NULL) {
            config.push_back(std::string(line));
        }
        if (config.size() >= MAX_CONFIG_NUM * GEMM_NUM) {
            int startIdx = config.size() - (MAX_CONFIG_NUM - 1) * GEMM_NUM;
            fclose(fout);
            fout = fopen(IGEMM_CONFIG, "w+");
            for (int i = startIdx; i < (int)config.size(); i++) {
                fprintf(fout, "%s", config[i].c_str());
            }
        }
    }

    batch_size_    = batch_size;
    seq_len_       = seq_len;
    head_num_      = head_num;
    size_per_head_ = size_per_head;
    int m          = batch_size * seq_len;
    int n          = head_num * size_per_head;
    int k          = n;
    int batchCount;

    printf("***Encoder IGemm Testing Begin***\n");
    printf("\n-----------------------------\n");

    batchCount = 3;
    m          = batch_size * seq_len;
    k          = head_num * size_per_head;
    n          = k;
    if (n % 32 != 0 || k % 32 != 0) {
        printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
    }
    else {
        batch_igemm_config(batchCount, m, n, k, fout, buffer);
    }

    printf("\n-----------------------------\n");
    m          = seq_len;
    n          = seq_len;
    k          = size_per_head;
    batchCount = batch_size * head_num;
    if (n % 32 != 0 || k % 32 != 0) {
        printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
    }
    else {
        batch_igemm_config(batchCount, m, n, k, fout, buffer);
    }

    printf("\n-----------------------------\n");
    m          = seq_len;
    n          = size_per_head;
    k          = seq_len;
    batchCount = batch_size * head_num;
    if (n % 32 != 0 || k % 32 != 0) {
        printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
    }
    else {
        batch_igemm_config(batchCount, m, n, k, fout, buffer);
    }

    printf("\n-----------------------------\n");
    m = batch_size * seq_len;
    n = head_num * size_per_head;
    k = head_num * size_per_head;
    if (n % 32 != 0 || k % 32 != 0) {
        printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
    }
    else {
        igemm_config(m, n, k, fout, buffer);
    }

    printf("\n-----------------------------\n");
    n = 4 * n;
    if (n % 32 != 0 || k % 32 != 0) {
        printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
    }
    else {
        igemm_config(m, n, k, fout, buffer);
    }

    printf("\n-----------------------------\n");
    n = k;
    k = 4 * n;
    if (n % 32 != 0 || k % 32 != 0) {
        printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
    }
    else {
        igemm_config(m, n, k, fout, buffer);
    }

    fclose(fout);
    printf("\n-----------------------------\n");
    printf("***Encoder IGemm Testing End***\n");

#ifdef SPARSITY_ENABLED
    bool do_sparse_test = false;
    if (prop.major == 8 && (prop.minor == 0 || prop.minor == 6)) {
        do_sparse_test = true;
    }
    if (do_sparse_test) {
        printf("***cusparseLt Gemm Testing Begin***\n");
        const int      spgemm_num = 3;
        FILE*          fd;
        int            line_count = 0;
        const int      ites       = 100;
        struct timeval start, end;
        if (!isAppend) {
            fd = fopen(SPIGEMM_CONFIG, "w+");
        }
        else {
            fd = fopen(SPIGEMM_CONFIG, "a+");
            std::vector<std::string> config;
            char                     line[1024];
            while (fgets(line, 1024, fd) != NULL) {
                config.push_back(std::string(line));
            }
            line_count = config.size();
            if (config.size() >= (MAX_CONFIG_NUM * spgemm_num + 1))  // 6 cublas/cublasLt, first row is not included
            {
                int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * spgemm_num);
                fclose(fd);
                fd = fopen(SPIGEMM_CONFIG, "w+");
                fprintf(fd, "%s", config[0].c_str());
                for (uint i = startIdx; i < config.size(); i++) {
                    fprintf(fd, "%s", config[i].c_str());
                }
                line_count = config.size() - (spgemm_num + 3);
            }
        }
        if (line_count == 0) {
            fprintf(
                fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, m, n, k, algoId, exec_time\n");
        }

        int M[spgemm_num];
        int N[spgemm_num];
        int K[spgemm_num];
        // gemm1
        M[0] = batch_size * seq_len;
        K[0] = head_num * size_per_head;
        N[0] = K[0];
        // gemm2
        M[1] = M[0];
        K[1] = K[0];
        N[1] = 4 * N[0];
        // gemm3
        M[2] = M[0];
        K[2] = 4 * K[0];
        N[2] = N[0];

        cusparseLtHandle_t handle;
        CHECK_CUSPARSE(cusparseLtInit(&handle));
        cusparseOrder_t     col_order    = CUSPARSE_ORDER_COL;
        cusparseOrder_t     row_order    = CUSPARSE_ORDER_ROW;
        cusparseOperation_t opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB          = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseComputeType compute_type = CUSPARSE_COMPUTE_32I;
        unsigned            alignment    = 16;
        cudaStream_t        stream       = 0;
        float               alpha2       = 1.0f;
        float               beta2        = 0.0f;
        for (int i = 0; i < spgemm_num; ++i) {
            // to be compatible with spgemm wrapper, we let A be the weight matrix
            // so m and n are swapped
            // A: mxk B: kxn C:mxn
            int m = N[i], n = M[i], k = K[i];
            printf("\n-----------------------------\n");
            printf("GEMM test %d: [M: %d, K: %d, N: %d]\n", i, m, k, n);
            int8_t* d_A = (int8_t*)buffer;
            int8_t* d_B = d_A + m * k;
            int8_t* d_C = d_B + k * n;
            int8_t* dA_compressed;
            {
                cusparseLtMatDescriptor_t mat_A;
                CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                    &handle, &mat_A, m, k, k, alignment, CUDA_R_8I, row_order, CUSPARSELT_SPARSITY_50_PERCENT))
                CHECK_CUSPARSE(
                    cusparseLtSpMMAPrune2(&handle, &mat_A, true, opA, d_A, d_A, CUSPARSELT_PRUNE_SPMMA_STRIP, stream))
                size_t compressed_size;
                CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, &mat_A, &compressed_size))
                check_cuda_error(cudaMalloc((void**)&dA_compressed, compressed_size));
                CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, &mat_A, true, opA, d_A, dA_compressed, stream))
            }
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: "));
            }

            float exec_time = 99999.0f;
            int   fast_algo = 0;
            for (int alg = 0; alg < 4; ++alg) {
                cudaDeviceSynchronize();
                cusparseLtMatDescriptor_t mat_A, matB, mat_C;
                void*                     d_workspace = nullptr;
                int                       num_streams = 1;
                cudaStream_t              streams[1]  = {stream};
                CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                    &handle, &mat_A, m, k, k, alignment, CUDA_R_8I, row_order, CUSPARSELT_SPARSITY_50_PERCENT))
                CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, k, alignment, CUDA_R_8I, col_order))
                CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &mat_C, m, n, m, alignment, CUDA_R_8I, col_order))
                gettimeofday(&start, NULL);
                for (int ite = 0; ite < ites; ++ite) {
                    // initializing MatDesc takes a lot of time
                    // and these descs can be stored to other place
                    // whereas storing MatMulPlan to other place will cause errors
                    cusparseLtMatmulDescriptor_t   matmul;
                    cusparseLtMatmulAlgSelection_t alg_sel;
                    cusparseLtMatmulPlan_t         plan;
                    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
                        &handle, &matmul, opA, opB, &mat_A, &matB, &mat_C, &mat_C, compute_type))
                    CHECK_CUSPARSE(
                        cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
                    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
                        &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
                    size_t workspace_size;
                    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
                    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size))
                    CHECK_CUSPARSE(cusparseLtMatmul(&handle,
                                                    &plan,
                                                    &alpha2,
                                                    dA_compressed,
                                                    d_B,
                                                    &beta2,
                                                    d_C,
                                                    d_C,
                                                    d_workspace,
                                                    streams,
                                                    num_streams))
                    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
                }
                cudaDeviceSynchronize();
                gettimeofday(&end, NULL);
                printf("algo_%d costs %.3fms \n", alg, diffTime(start, end) / ites);
                if (diffTime(start, end) < exec_time) {
                    exec_time = diffTime(start, end);
                    fast_algo = alg;
                }
            }
            exec_time /= ites;
            printf("fast_algo %d\n", fast_algo);
            fprintf(fd,
                    "%d %d %d %d %d ### 1 %d %d %d %d %f\n",
                    batch_size,
                    seq_len,
                    head_num,
                    size_per_head,
                    HALF_DATATYPE,
                    m,
                    n,
                    k,
                    fast_algo,
                    exec_time);
            cudaFree(dA_compressed);
        }
        CHECK_CUSPARSE(cusparseLtDestroy(&handle))
        fclose(fd);
        printf("***cusparseLt Gemm Testing End***\n");
    }
#endif
    return 0;
}

}  // namespace fastertransformer
