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

#include "swin_igemm_func.h"

namespace turbomind {

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
            auto start = std::chrono::high_resolution_clock::now();
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
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration<float, std::milli>(end - start);
            if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                algoStatus = oneRunStatus;
            }
            float time = dur.count();
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

int igemm_config_INT8IO(int m, int n, int k, FILE* fout, void* buffer)
{
    printf("batchCount %d m %d n %d k %d\n", 1, m, n, k);
    float alpha = 1.0f;
    float beta  = 0.0f;

    int8_t* d_A = (int8_t*)buffer;         // m * k, stored in column-major
    int8_t* d_B = d_A + m * k;             // k * n, stored in column-major
    int8_t* d_C = (int8_t*)(d_B + k * n);  // m * n, stored in column-major

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

int generate_swin_igemm_config(
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

    int       m = batch_size * seq_len;
    int       n = head_num * size_per_head;
    int       k = n;
    int       batchCount;
    const int NUM_OF_BASIC_LAYERS = 4;

    printf("***Swin IGemm Testing Begin***\n");

    for (int basic_layer = 0; basic_layer < NUM_OF_BASIC_LAYERS; basic_layer++) {
        printf("\n-----------------------------\n");
        batchCount = 1;
        m          = batch_size * seq_len;
        k          = head_num * size_per_head;
        n          = 3 * head_num * size_per_head;
        if (n % 32 != 0 || k % 32 != 0) {
            printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
        }
        else {
            igemm_config_INT8IO(m, n, k, fout, buffer);
        }

        printf("\n-----------------------------\n");
        m = batch_size * seq_len;
        n = head_num * size_per_head;
        k = head_num * size_per_head;
        if (n % 32 != 0 || k % 32 != 0) {
            printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
        }
        else {
            igemm_config_INT8IO(m, n, k, fout, buffer);
        }

        printf("\n-----------------------------\n");
        m = batch_size * seq_len;
        n = 4 * head_num * size_per_head;
        k = head_num * size_per_head;
        if (n % 32 != 0 || k % 32 != 0) {
            printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
        }
        else {
            igemm_config_INT8IO(m, n, k, fout, buffer);
        }

        printf("\n-----------------------------\n");
        m = batch_size * seq_len;
        n = head_num * size_per_head;
        k = 4 * head_num * size_per_head;
        if (n % 32 != 0 || k % 32 != 0) {
            printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
        }
        else {
            igemm_config_INT8IO(m, n, k, fout, buffer);
        }

        if (basic_layer != NUM_OF_BASIC_LAYERS - 1) {
            printf("\n-----------------------------\n");
            batch_size = batch_size / 4;
            head_num   = head_num * 2;
            m          = batch_size * seq_len;
            n          = head_num * size_per_head;
            k          = 2 * head_num * size_per_head;
            if (n % 32 != 0 || k % 32 != 0) {
                printf("[WARNING] For INT8 gemm test, n, k should be multiples of 32 (n = %d, k = %d)\n", n, k);
            }
            else {
                igemm_config_INT8IO(m, n, k, fout, buffer);
            }
        }
        printf("\n-----------------------------\n");
    }

    fclose(fout);
    printf("\n-----------------------------\n");
    printf("***Swin IGemm Testing End***\n");
    return 0;
}

}  // namespace turbomind
