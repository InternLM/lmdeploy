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

#include "encoder_gemm_func.h"
#include <assert.h>
#include <sys/types.h>
#include <vector>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace turbomind {

// Utility function to print customMatmulPerf_t structure
int printPerfStructure(int                       batch_size,
                       int                       seq_len,
                       int                       head_num,
                       int                       size_per_head,
                       int                       m,
                       int                       n,
                       int                       k,
                       const customMatmulPerf_t& perf,
                       FILE*                     fout,
                       CublasDataType            data_type,
                       int                       hasPrint,
                       int                       batch_count)
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
    stages = 0;
#endif
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
    uint16_t inner_shapeId, cluster_shapeId;
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner_shapeId, sizeof(inner_shapeId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster_shapeId, sizeof(cluster_shapeId), NULL);
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
    uint16_t mma_shapeId, cga_shapeId, sche_mode;
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &mma_shapeId, sizeof(mma_shapeId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &cga_shapeId, sizeof(cga_shapeId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &sche_mode, sizeof(sche_mode), NULL);
#endif

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d "
#if (CUDART_VERSION >= 11000)
           "stages=%d "
#endif
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
           "inner_shapeId=%d cluster_shapeId=%d"
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
           "mma_shapeId=%d cga_shapeId=%d schedule_mode=%d"
#endif
           "} status %d "
           "time %fms workspace=%d mathMode=%d waves=%f\n",
           algoId,
           tile,
           matmulTileName[tile],
           numSplitsK,
           reductionScheme,
           swizzle,
           customOption,
#if (CUDART_VERSION >= 11000)
           stages,
#endif
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
           inner_shapeId,
           cluster_shapeId,
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
           mma_shapeId,
           cga_shapeId,
           sche_mode,
#endif
           perf.status,
           perf.time,
           (int)perf.workspaceSize,
           (int)perf.mathMode,
           perf.wavesCount);
    if (hasPrint == 0) {
        fprintf(fout,
                "%d %d %d %d %d ### %d %d %d %d %d %d %d %d %d %d %d %d "
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
                "%d %d "
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
                "%d %d %d "
#endif
                "%f\n",
                batch_size,
                seq_len,
                head_num,
                size_per_head,
                data_type,
                batch_count,
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
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
                inner_shapeId,
                cluster_shapeId,
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
                mma_shapeId,
                cga_shapeId,
                sche_mode,
#endif
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
                                      cudaStream_t                stream,
                                      cudaEvent_t&                startEvent,
                                      cudaEvent_t&                stopEvent)
{
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int            repeats = kernelRepeats;
    cublasStatus_t algoStatus =
        cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            cudaError_t err, err1, err2, err3;
            err = cudaEventRecord(startEvent, stream);
            for (int loop = 0; loop < repeats; loop++) {
                cublasStatus_t oneRunStatus = cublasLtMatmul(ltHandle,
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
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord(stopEvent, stream);
            err2 = cudaEventSynchronize(stopEvent);
            float time;
            err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
            if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }
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

    return algoStatus;
}

template<typename T, typename scaleT>
int LtHgemmCustomFind(cublasLtHandle_t   ltHandle,
                      int                batch_size,
                      int                seq_len,
                      int                head_num,
                      int                size_per_head,
                      int                m,
                      int                n,
                      int                k,
                      const scaleT*      alpha, /* host pointer */
                      const T*           A,
                      const T*           B,
                      const scaleT*      beta, /* host pointer */
                      T*                 C,
                      void*              workSpace,
                      size_t             workSpaceSize,
                      FILE*              fout,
                      customMatmulPerf_t perfResults[],
                      int                AlgoCombinations,
                      cudaDataType_t     dtype_fp8,
                      int                batchCount,
                      int64_t            strideA,
                      int64_t            strideB,
                      int64_t            strideD)
{
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cudaEvent_t    startEvent;
    cudaEvent_t    stopEvent;
    CublasDataType data_type;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;

    cudaStream_t stream = 0;
    // SplitK value that we are going to try when SplitK is supported for a
    // given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    // Let try a fixed number of combinations
    int                               AlgoCount         = 0;
    int                               AlgoCountRestrict = 0;            // workspace == 0
    const int                         maxNumTraversal   = 50;           // max number of traversal
    std::vector<cublasLtMatmulAlgo_t> algos(AlgoCombinations);          // 0 <= workspace <= 32MB
    std::vector<cublasLtMatmulAlgo_t> algosRestrict(AlgoCombinations);  // workspace == 0
    const int                         kernelRepeats = 100;  // number of time the CUDA kernels will be run back to back
    int                               nbAlgoIds     = 0;    // Number of algorithms actually returned by
                                                            // cublasLtMatmulAlgoGetIds function.
#define ALGO_IDS 100                                        // Number of algorithms requested.
    int algoIdA[ALGO_IDS];                                  // Array containing the algorithm IDs returned by
                                                            // cublasLtMatmulAlgoGetIds function.
    cudaDataType_t Atype, Btype, Ctype, scaleType, Dtype;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (std::is_same<T, float>::value) {
        data_type = FLOAT_DATATYPE;
        Atype = CUDA_R_32F, Btype = CUDA_R_32F, Ctype = CUDA_R_32F, Dtype = CUDA_R_32F;
    }
    else if (std::is_same<T, half>::value) {
        data_type = HALF_DATATYPE;
        Atype = CUDA_R_16F, Btype = CUDA_R_16F, Ctype = CUDA_R_16F, Dtype = CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        data_type = BFLOAT16_DATATYPE;
        Atype = CUDA_R_16BF, Btype = CUDA_R_16BF, Ctype = CUDA_R_16BF, Dtype = CUDA_R_16BF;
    }
#endif
#ifdef ENABLE_FP8
    else if (std::is_same<T, __nv_fp8_e4m3>::value) {
        data_type = FP8_DATATYPE;
        Atype = CUDA_R_8F_E4M3, Btype = CUDA_R_8F_E4M3, Ctype = CUDA_R_16BF;
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
        Dtype = CUDA_R_16BF;
#else
        Dtype = dtype_fp8;
#endif
    }
#endif

    if (sizeof(scaleT) == sizeof(float)) {
        scaleType = CUDA_R_32F;
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
    }
    else {
        scaleType = CUDA_R_16F;
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
    }

    const cublasOperation_t tA = data_type == FP8_DATATYPE ? CUBLAS_OP_T : CUBLAS_OP_N;

// Create operation descriptor; see cublasLtMatmulDescAttributes_t for
// details about defaults; here we just need to set the transforms for A and
// B
#if (CUDART_VERSION >= 11000)
    status = cublasLtMatmulDescCreate(&operationDesc, computeType,
                                      scaleType);  //  creates a matrix multiply descriptor
#else
    status = cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
#ifdef ENABLE_FP8
    if (data_type == FP8_DATATYPE) {
        const int8_t fastAccuMode = 1;  // enable fast imprecise accum
        status                    = cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(decltype(fastAccuMode)));
        if (status != CUBLAS_STATUS_SUCCESS) {
            goto CLEANUP;
        }
    }
#endif

    // Create matrix descriptors. We are good with the details here so no need
    // to set any extra attributes
    if (data_type == FP8_DATATYPE) {
        status = cublasLtMatrixLayoutCreate(&Adesc, Atype, k, m, k);
    }
    else {
        status = cublasLtMatrixLayoutCreate(&Adesc, Atype, m, k, m);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    status = cublasLtMatrixLayoutCreate(&Bdesc, Btype, k, n, k);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutCreate(&Ddesc, Dtype, m, n, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    if (batchCount > 1) {
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));

        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
        check_cuda_error(cublasLtMatrixLayoutSetAttribute(
            Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
    }

    // Create CUDA event to time the execution time of each algo
    if (cudaEventCreate(&startEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }
    if (cudaEventCreate(&stopEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }

    // Request the 100 first AlgoId available
    status = cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    if (nbAlgoIds > ALGO_IDS) {
        printf(
            "Warning: the algo id count is not large enough to guarantee the best algo %d, %d\n", nbAlgoIds, ALGO_IDS);
    }

    // Loop over the Algo IDs
    // This loop doesn't work for fp8 gemm
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {
        cublasLtMatmulAlgo_t algo;
        size_t               sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status =
            cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoIdA[idx], &algo);
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
        // Retrieve Algo Capabilities attributes to be able to setup loop over
        // the different combinations
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
                        // Loop over the splitK value over a fixed sequence
                        // splitKSequenceA in addition to the case where splitK
                        // is not enabled
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
                                     redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations);
                                     redScheme = redScheme << 1) {
                                    if (redScheme & redMask) {
                                        cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                             &redScheme,
                                                                             sizeof(redScheme));

                                        cublasLtMatmulHeuristicResult_t heurResult;
                                        cublasStatus_t                  algoStatus = cublasLtMatmulAlgoCheck(
                                            ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);
                                        if (heurResult.workspaceSize > workSpaceSize) {
                                            // printf("not enough workspace!
                                            // %ld\n",
                                            // heurResult.workspaceSize);
                                            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
                                        }
                                        else if (heurResult.workspaceSize == 0) {
                                            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                                algosRestrict[AlgoCountRestrict++] = algo;
                                            }
                                        }
                                        if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                            algos[AlgoCount++] = algo;
                                        }
                                    }  // end if
                                }      // end for
                            }
                            else {  // Non-splitK case
                                /* if user preference is ok with workspace */
                                if (AlgoCount < AlgoCombinations) {
                                    cublasLtMatmulHeuristicResult_t heurResult;
                                    cublasStatus_t                  algoStatus = cublasLtMatmulAlgoCheck(
                                        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);
                                    if (heurResult.workspaceSize > workSpaceSize) {
                                        // printf("not enough workspace! %ld\n",
                                        // heurResult.workspaceSize);
                                        algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not
                                                                                   // enough
                                                                                   // workspace
                                    }
                                    else if (heurResult.workspaceSize == 0) {
                                        if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                            algosRestrict[AlgoCountRestrict++] = algo;
                                        }
                                    }
                                    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                        algos[AlgoCount++] = algo;
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

    printf("AlgoCount: %d\n", AlgoCount);
    if (data_type == FP8_DATATYPE) {
        assert(AlgoCount == 0);
    }
    if (AlgoCount < maxNumTraversal && data_type != FP8_DATATYPE) {
        // 0 <= workspacesize <= 32MB
        for (int i = 0; i < AlgoCount; i++) {
            status                = customMatmulRun(ltHandle,
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
                                     algos[i],
                                     kernelRepeats,
                                     workSpace,
                                     workSpaceSize,
                                     perfResults[i],
                                     stream,
                                     startEvent,
                                     stopEvent);
            perfResults[i].status = status;
            // if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
        }
    }
    else {
        // Heuristic + workspacesize==0
        AlgoCount = 0;
        nbAlgoIds = 0;
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        uint64_t maxWorkSpaceSize = workSpaceSize;  //(32MB)
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkSpaceSize, sizeof(maxWorkSpaceSize));
        cublasLtMatmulHeuristicResult_t heuristicResultsArray[maxNumTraversal];

        cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                       operationDesc,
                                       Adesc,
                                       Bdesc,
                                       Cdesc,
                                       Ddesc,
                                       pref,
                                       maxNumTraversal,
                                       heuristicResultsArray,
                                       &nbAlgoIds);
        cublasLtMatmulPreferenceDestroy(pref);
        printf("return %d and run heuristic algo\n", nbAlgoIds);
        for (int i = 0; i < nbAlgoIds; i++) {
            if (heuristicResultsArray[i].state == CUBLAS_STATUS_SUCCESS) {
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
                                         Ddesc,
                                         heuristicResultsArray[i].algo,
                                         kernelRepeats,
                                         workSpace,
                                         workSpaceSize,
                                         perfResults[AlgoCount],
                                         stream,
                                         startEvent,
                                         stopEvent);
                perfResults[AlgoCount].status = status;
                if (status == CUBLAS_STATUS_SUCCESS) {
                    AlgoCount++;
                }
            }
        }

        // workspacesize==0
        printf("workspacesize==0, run %d algos\n", AlgoCountRestrict);
        for (int i = 0; i < AlgoCountRestrict && i < (maxNumTraversal - nbAlgoIds); i++) {
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
                                     Ddesc,
                                     algosRestrict[i],
                                     kernelRepeats,
                                     NULL,
                                     0,
                                     perfResults[AlgoCount],
                                     stream,
                                     startEvent,
                                     stopEvent);
            perfResults[AlgoCount].status = status;
            if (status == CUBLAS_STATUS_SUCCESS) {
                AlgoCount++;
            }
        }
    }

    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details
    for (int i = 0, hasPrint = 1; i < AlgoCount; i++) {
        printf("result %03d : ", i);
        hasPrint = printPerfStructure(batch_size,
                                      seq_len,
                                      head_num,
                                      size_per_head,
                                      m,
                                      n,
                                      k,
                                      perfResults[i],
                                      fout,
                                      data_type,
                                      hasPrint,
                                      batchCount);
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
    if (startEvent) {
        cudaEventDestroy(startEvent);
    }
    if (stopEvent) {
        cudaEventDestroy(stopEvent);
    }
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

template int LtHgemmCustomFind(cublasLtHandle_t   ltHandle,
                               int                batch_size,
                               int                seq_len,
                               int                head_num,
                               int                size_per_head,
                               int                m,
                               int                n,
                               int                k,
                               const float*       alpha, /* host pointer */
                               const float*       A,
                               const float*       B,
                               const float*       beta, /* host pointer */
                               float*             C,
                               void*              workSpace,
                               size_t             workSpaceSize,
                               FILE*              fout,
                               customMatmulPerf_t perfResults[],
                               int                AlgoCombinations,
                               cudaDataType_t     dtype_fp8,
                               int                batchCount,
                               int64_t            strideA,
                               int64_t            strideB,
                               int64_t            strideD);

template int LtHgemmCustomFind(cublasLtHandle_t   ltHandle,
                               int                batch_size,
                               int                seq_len,
                               int                head_num,
                               int                size_per_head,
                               int                m,
                               int                n,
                               int                k,
                               const half*        alpha, /* host pointer */
                               const half*        A,
                               const half*        B,
                               const half*        beta, /* host pointer */
                               half*              C,
                               void*              workSpace,
                               size_t             workSpaceSize,
                               FILE*              fout,
                               customMatmulPerf_t perfResults[],
                               int                AlgoCombinations,
                               cudaDataType_t     dtype_fp8,
                               int                batchCount,
                               int64_t            strideA,
                               int64_t            strideB,
                               int64_t            strideD);

#ifdef ENABLE_BF16
template int LtHgemmCustomFind(cublasLtHandle_t     ltHandle,
                               int                  batch_size,
                               int                  seq_len,
                               int                  head_num,
                               int                  size_per_head,
                               int                  m,
                               int                  n,
                               int                  k,
                               const float*         alpha, /* host pointer */
                               const __nv_bfloat16* A,
                               const __nv_bfloat16* B,
                               const float*         beta, /* host pointer */
                               __nv_bfloat16*       C,
                               void*                workSpace,
                               size_t               workSpaceSize,
                               FILE*                fout,
                               customMatmulPerf_t   perfResults[],
                               int                  AlgoCombinations,
                               cudaDataType_t       dtype_fp8,
                               int                  batchCount,
                               int64_t              strideA,
                               int64_t              strideB,
                               int64_t              strideD);
#endif

#ifdef ENABLE_FP8
template int LtHgemmCustomFind(cublasLtHandle_t     ltHandle,
                               int                  batch_size,
                               int                  seq_len,
                               int                  head_num,
                               int                  size_per_head,
                               int                  m,
                               int                  n,
                               int                  k,
                               const float*         alpha, /* host pointer */
                               const __nv_fp8_e4m3* A,
                               const __nv_fp8_e4m3* B,
                               const float*         beta, /* host pointer */
                               __nv_fp8_e4m3*       C,
                               void*                workSpace,
                               size_t               workSpaceSize,
                               FILE*                fout,
                               customMatmulPerf_t   perfResults[],
                               int                  AlgoCombinations,
                               cudaDataType_t       dtype_fp8,
                               int                  batchCount,
                               int64_t              strideA,
                               int64_t              strideB,
                               int64_t              strideD);
#endif

template int LtHgemmCustomFind(cublasLtHandle_t   ltHandle,
                               int                batch_size,
                               int                seq_len,
                               int                head_num,
                               int                size_per_head,
                               int                m,
                               int                n,
                               int                k,
                               const float*       alpha, /* host pointer */
                               const half*        A,
                               const half*        B,
                               const float*       beta, /* host pointer */
                               half*              C,
                               void*              workSpace,
                               size_t             workSpaceSize,
                               FILE*              fout,
                               customMatmulPerf_t perfResults[],
                               int                AlgoCombinations,
                               cudaDataType_t     dtype_fp8,
                               int                batchCount,
                               int64_t            strideA,
                               int64_t            strideB,
                               int64_t            strideD);

size_t calGemmTestBufSizeInByte(int            batch_size,
                                int            seq_len,
                                int            head_num,
                                int            size_per_head,
                                int            inter_size,
                                int            vocab_size,
                                int            int8_mode,
                                CublasDataType data_type)
{
    size_t buf_size_in_byte;
    if (int8_mode > 0) {
        int m = batch_size * seq_len;
        int n = head_num * size_per_head;
        int k = n;

        size_t size1 = 3 * (m * k * sizeof(int8_t) + k * n * sizeof(int8_t) + m * n * sizeof(int));
        size_t size2 = batch_size * head_num
                       * (seq_len * size_per_head * sizeof(int8_t) + size_per_head * seq_len * sizeof(int8_t)
                          + seq_len * seq_len * sizeof(int));
        size_t size3 = batch_size * head_num
                       * (seq_len * seq_len * sizeof(int8_t) + seq_len * size_per_head * sizeof(int8_t)
                          + seq_len * size_per_head * sizeof(int));
        size_t size4     = m * k * sizeof(int8_t) + k * inter_size * sizeof(int8_t) + m * inter_size * sizeof(int);
        size_t size5     = m * k * sizeof(int8_t) + k * vocab_size * sizeof(int8_t) + m * vocab_size * sizeof(int);
        buf_size_in_byte = size1 > size2 ? size1 : size2;
        buf_size_in_byte = buf_size_in_byte > size3 ? buf_size_in_byte : size3;
        buf_size_in_byte = buf_size_in_byte > size4 ? buf_size_in_byte : size4;
        buf_size_in_byte = buf_size_in_byte > size5 ? buf_size_in_byte : size5;
    }
    else {
        size_t m = batch_size * seq_len;
        size_t n = head_num * size_per_head;
        size_t k = n;
        // TODO need to add bfloat16 here
        int    wordSize = (data_type == FLOAT_DATATYPE ? sizeof(float) : sizeof(half));
        size_t size1    = 3 * (m * k + k * n + m * n) * wordSize;
        size_t size2    = (size_t)batch_size * (size_t)head_num
                       * ((size_t)seq_len * (size_t)seq_len + (size_t)seq_len * (size_t)size_per_head
                          + (size_t)seq_len * (size_t)size_per_head)
                       * (size_t)wordSize;
        size_t size3     = (m * k + k * inter_size + m * inter_size) * wordSize;
        size_t size4     = (m * k + k * vocab_size + m * vocab_size) * wordSize;
        buf_size_in_byte = size1 > size2 ? size1 : size2;
        buf_size_in_byte = buf_size_in_byte > size3 ? buf_size_in_byte : size3;
        buf_size_in_byte = buf_size_in_byte > size4 ? buf_size_in_byte : size4;
        buf_size_in_byte +=
            ((data_type == HALF_DATATYPE || data_type == BFLOAT16_DATATYPE) ? CUBLAS_WORKSPACE_SIZE : 0);
    }
    return buf_size_in_byte;
}

size_t calGemmTestBufSizeInByteXlnet(
    int batch_size, int seq_len, int head_num, int size_per_head, int inter_size, int hidden_units, int is_fp16)
{
    int M[10]          = {0};
    int N[10]          = {0};
    int K[10]          = {0};
    int batchCount[10] = {0};

    // gemm1
    M[0]          = hidden_units;
    N[0]          = seq_len * batch_size;
    K[0]          = hidden_units;
    batchCount[0] = 3;

    // gemm2
    M[1]          = hidden_units;
    N[1]          = seq_len * 2;
    K[1]          = hidden_units;
    batchCount[1] = 1;

    // gemm3
    M[2]          = seq_len;
    N[2]          = seq_len;
    K[2]          = size_per_head;
    batchCount[2] = batch_size * head_num;

    // gemm4
    M[3]          = seq_len * 2;
    N[3]          = seq_len;
    K[3]          = size_per_head;
    batchCount[3] = batch_size * head_num;

    // gemm5
    M[4]          = 2;
    N[4]          = seq_len;
    K[4]          = size_per_head;
    batchCount[4] = batch_size * head_num;

    // gemm6
    M[5] = head_num;
    N[5] = seq_len;
    K[5] = 2;
    // gemm7
    M[6]          = size_per_head;
    N[6]          = seq_len;
    K[6]          = seq_len;
    batchCount[6] = batch_size * head_num;

    // gemm8
    M[7]          = hidden_units;
    N[7]          = seq_len;
    K[7]          = hidden_units;
    batchCount[7] = batch_size;

    // gemm9
    M[8]          = inter_size;
    N[8]          = seq_len;
    K[8]          = hidden_units;
    batchCount[8] = batch_size;

    // gemm10
    M[9]          = hidden_units;
    N[9]          = seq_len;
    K[9]          = inter_size;
    batchCount[9] = batch_size;

    size_t max_size = 0;

    for (int i = 0; i < 10; ++i) {
        int    m = M[i], n = N[i], k = K[i];
        size_t size = (M[i] * N[i] + M[i] * K[i] + N[i] * K[i]) * batchCount[i];
        if (size > max_size) {
            max_size = size;
        }
    }

    int size_per_ele = 4;
    if (is_fp16 == true) {
        size_per_ele = 2;
    }
    return max_size * size_per_ele;
}

}  // namespace turbomind
