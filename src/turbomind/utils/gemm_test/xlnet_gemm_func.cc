/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/utils/gemm_test/xlnet_gemm_func.h"

namespace turbomind {

template<typename T>
void generate_xlnet_gemm_config(int   batch_size,
                                int   seq_len,
                                int   head_num,
                                int   size_per_head,
                                int   hidden_units_,
                                int   inter_size_,
                                void* buffer_in,
                                bool  isAppend)
{
    void* cublas_workspace;
    void* buffer;
    int   workSpaceSize;

#ifdef ENABLE_BF16
    if (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) {
#else
    if (std::is_same<T, half>::value) {
#endif  // ENABLE_BF16
        // cublas_workspace_ should be the start pointer of cudaMalloc()
        // to ensure 16B alignemnet
        cublas_workspace = buffer_in;
        buffer           = (void*)((char*)cublas_workspace + CUBLAS_WORKSPACE_SIZE);
        workSpaceSize    = CUBLAS_WORKSPACE_SIZE;
    }
    else {
        cublas_workspace = nullptr;
        buffer           = buffer_in;
        workSpaceSize    = 0;
    }

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    // check config
    FILE* fd;
    int   line_count = 0;
    if (!isAppend) {
        fd = fopen(GEMM_CONFIG, "w+");
    }
    else {
        fd = fopen(GEMM_CONFIG, "a+");
        std::vector<std::string> config;
        char                     line[1024];
        while (fgets(line, 1024, fd) != NULL) {
            config.push_back(std::string(line));
        }
        line_count = config.size();
        if (config.size() >= (MAX_CONFIG_NUM * GEMM_NUM + 1))  // 6 cublas/cublasLt, first row is not included
        {
            int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * GEMM_NUM);
            fclose(fd);
            fd = fopen(GEMM_CONFIG, "w+");
            fprintf(fd, "%s", config[0].c_str());
            for (uint i = startIdx; i < config.size(); i++) {
                fprintf(fd, "%s", config[i].c_str());
            }
            line_count = config.size() - (GEMM_NUM + 3);
        }
    }

    const int         gemm_num = 10;
    int               M[gemm_num];
    int               N[gemm_num];
    int               K[gemm_num];
    int               lda[gemm_num];
    int               strideA[gemm_num];
    int               ldb[gemm_num];
    int               strideB[gemm_num];
    int               ldc[gemm_num];
    int               strideC[gemm_num];
    cublasOperation_t transa[gemm_num]     = {CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N};
    cublasOperation_t transb[gemm_num]     = {CUBLAS_OP_N};
    int               batchCount[gemm_num] = {1};
    char              mess[gemm_num][256];

    // gemm1
    M[0]          = hidden_units_;
    N[0]          = seq_len * batch_size;
    K[0]          = hidden_units_;
    lda[0]        = hidden_units_;
    strideA[0]    = hidden_units_ * hidden_units_;
    ldb[0]        = hidden_units_;
    strideB[0]    = 0;
    ldc[0]        = hidden_units_;
    strideC[0]    = seq_len * batch_size * hidden_units_;
    batchCount[0] = 3;
    strcpy(mess[0], "from_tensor * weightQ/K/V");

    // gemm2
    M[1]          = hidden_units_;
    N[1]          = seq_len * 2;
    K[1]          = hidden_units_;
    batchCount[1] = 1;
    strcpy(mess[1], " k_head_r_");

    // gemm3
    M[2]          = seq_len;
    N[2]          = seq_len;
    K[2]          = size_per_head;
    lda[2]        = size_per_head;
    strideA[2]    = seq_len * size_per_head;
    ldb[2]        = size_per_head;
    strideB[2]    = seq_len * size_per_head;
    ldc[2]        = seq_len;
    strideC[2]    = seq_len * seq_len;
    batchCount[2] = batch_size * head_num;
    strcpy(mess[2], "ac");

    // gemm4
    M[3]       = seq_len * 2;
    N[3]       = seq_len;
    K[3]       = size_per_head;
    lda[3]     = size_per_head;
    strideA[3] = seq_len * 2 * size_per_head;
    ldb[3]     = size_per_head;
    strideB[3] = seq_len * size_per_head;
    ldc[3]     = seq_len * 2;
    strideC[3] = seq_len * seq_len * 2;

    batchCount[3] = batch_size * head_num;
    strcpy(mess[3], "bd");

    // gemm5
    M[4]          = 2;
    N[4]          = seq_len;
    K[4]          = size_per_head;
    lda[4]        = size_per_head;
    strideA[4]    = 2 * size_per_head;
    ldb[4]        = size_per_head;
    strideB[4]    = seq_len * size_per_head;
    ldc[4]        = 2;
    strideC[4]    = seq_len * 2;
    batchCount[4] = batch_size * head_num;
    strcpy(mess[4], "ef");

    // gemm6
    M[5]       = head_num;
    N[5]       = seq_len;
    K[5]       = 2;
    lda[5]     = 2;
    strideA[5] = 2 * head_num;
    ldb[5]     = 2;
    strideB[5] = seq_len * 2;
    ldc[5]     = head_num;
    strideC[5] = seq_len * head_num;

    batchCount[5] = batch_size * seq_len;
    strcpy(mess[5], "seg_mat");
    // gemm7
    M[6]       = size_per_head;
    N[6]       = seq_len;
    K[6]       = seq_len;
    lda[6]     = size_per_head;
    strideA[6] = seq_len * size_per_head;
    ldb[6]     = seq_len;
    strideB[6] = seq_len * seq_len;
    ldc[6]     = size_per_head;
    strideC[6] = seq_len * size_per_head;

    batchCount[6] = batch_size * head_num;
    strcpy(mess[6], "attn_vec");

    // gemm8
    M[7]          = hidden_units_;
    N[7]          = seq_len * batch_size;
    K[7]          = hidden_units_;
    lda[7]        = hidden_units_;
    batchCount[7] = 1;
    strcpy(mess[7], "attn_out");

    // gemm9
    M[8]          = inter_size_;
    N[8]          = seq_len * batch_size;
    K[8]          = hidden_units_;
    batchCount[8] = 1;
    strcpy(mess[8], "output_fc1_");

    // gemm10
    M[9]          = hidden_units_;
    N[9]          = seq_len * batch_size;
    K[9]          = inter_size_;
    batchCount[9] = 1;

    strcpy(mess[9], "output_fc2_");

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    cublasLtHandle_t ltHandle;
    check_cuda_error(cublasLtCreate(&ltHandle));

    cudaDataType_t AType;
    cudaDataType_t BType;
    cudaDataType_t CType;
    cudaDataType_t computeType;
    int            startAlgo, endAlgo;
    const int      ites = 100;
    struct timeval start, end;

    CublasDataType data_type;
    if (std::is_same<T, float>::value) {
        data_type   = FLOAT_DATATYPE;
        AType       = CUDA_R_32F;
        BType       = CUDA_R_32F;
        CType       = CUDA_R_32F;
        computeType = CUDA_R_32F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT;
        endAlgo     = (int)CUBLAS_GEMM_ALGO23;
    }
    else if (std::is_same<T, half>::value) {
        data_type   = HALF_DATATYPE;
        AType       = CUDA_R_16F;
        BType       = CUDA_R_16F;
        CType       = CUDA_R_16F;
        computeType = CUDA_R_32F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo     = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        data_type   = BFLOAT16_DATATYPE;
        AType       = CUDA_R_16BF;
        BType       = CUDA_R_16BF;
        CType       = CUDA_R_16BF;
        computeType = CUDA_R_32F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo     = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }
#endif

    using scaleT = typename ScaleTypeConverter<T, false>::Type;

    scaleT alpha = (scaleT)1.0f;
    scaleT beta  = (scaleT)0.0f;

    printf("***Xlnet Gemm Testing Begin***\n");
    printf("***Cublas Gemm Testing Begin***\n");
    if (line_count == 0) {
        fprintf(fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### "
                "batchCount, n, m, k, algoId, "
                "customOption, tile, numSplitsK, swizzle, reductionScheme, "
                "workspaceSize, stages, exec_time\n");
    }
    for (int i = 0; i < gemm_num; ++i) {
        int m = M[i], n = N[i], k = K[i];
        printf("\n-----------------------------\n");
        printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
        T* d_A = (T*)buffer;
        T* d_B = d_A + m * k * batchCount[i];
        T* d_C = d_B + k * n * batchCount[i];

        float exec_time = 99999.0f;
        int   fast_algo = 0;
        for (int algo = startAlgo; algo <= endAlgo; algo++) {
            cublasStatus_t status;
            cudaDeviceSynchronize();
            gettimeofday(&start, NULL);
            for (int ite = 0; ite < ites; ++ite) {
                if (i == 1 || i == 7 || i == 8 || i == 9) {
                    status = cublasGemmEx(cublas_handle,
                                          transa[i],
                                          transb[i],
                                          n,
                                          m,
                                          k,
                                          &alpha,
                                          d_A,
                                          AType,
                                          n,
                                          d_B,
                                          AType,
                                          k,
                                          &beta,
                                          d_C,
                                          CType,
                                          n,
                                          computeType,
                                          static_cast<cublasGemmAlgo_t>(algo));
                }
                else {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        transa[i],
                                                        transb[i],
                                                        m,
                                                        n,
                                                        k,
                                                        &alpha,
                                                        d_A,
                                                        BType,
                                                        lda[i],
                                                        strideA[i],
                                                        d_B,
                                                        AType,
                                                        ldb[i],
                                                        strideB[i],
                                                        &beta,
                                                        d_C,
                                                        CType,
                                                        ldc[i],
                                                        strideC[i],
                                                        batchCount[i],
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                if (status != CUBLAS_STATUS_SUCCESS) {
                    break;
                }
            }
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
            if (status == CUBLAS_STATUS_SUCCESS) {
                printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
                if (diffTime(start, end) / ites < exec_time) {
                    exec_time = diffTime(start, end) / ites;
                    fast_algo = algo;
                }  // end if diffTime
            }      // end status
        }          // end for algo

        printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);

        if ((i == 1 || i == 7 || i == 8 || i == 9) && data_type != FLOAT_DATATYPE) {
            printf("***cublasLt Gemm Testing Begin***\n");
            // Let try a fixed number of combinations
            int                ALGO_COMBINATIONS = 5000;
            customMatmulPerf_t perfResults[ALGO_COMBINATIONS];

            LtHgemmCustomFind<T, scaleT>(ltHandle,
                                         batch_size,
                                         seq_len,
                                         head_num,
                                         size_per_head,
                                         n,
                                         m,
                                         k,
                                         &alpha,
                                         d_B,
                                         d_A,
                                         &beta,
                                         d_C,
                                         cublas_workspace,
                                         workSpaceSize,
                                         fd,
                                         perfResults,
                                         ALGO_COMBINATIONS);
            if (perfResults[0].time < exec_time) {
                printPerfStructure(
                    batch_size, seq_len, head_num, size_per_head, n, m, k, perfResults[0], fd, data_type, 0);
                exec_time = perfResults[0].time;
            }
            else {
                fprintf(fd,
                        "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 "
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
                        "-1 -1 "
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
                        "-1 -1 -1 "
#endif
                        "%f\n",
                        batch_size,
                        seq_len,
                        head_num,
                        size_per_head,
                        data_type,
                        batchCount[i],
                        n,
                        m,
                        k,
                        fast_algo,
                        exec_time);
            }
            printf("***cublasLt Gemm Testing End***\n");
        }
        else {
            fprintf(fd,
                    "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 "
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
                    "-1 -1 "
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
                    "-1 -1 -1 "
#endif
                    "%f\n",
                    batch_size,
                    seq_len,
                    head_num,
                    size_per_head,
                    data_type,
                    batchCount[i],
                    n,
                    m,
                    k,
                    fast_algo,
                    exec_time);
        }  // end else fp16
    }      // end i
    printf("***cublas Gemm Testing End***\n\n");
    fclose(fd);
    printf("***Xlnet Gemm Testing End***\n");

    return;
}

template void generate_xlnet_gemm_config<float>(int   batch_size,
                                                int   seq_len,
                                                int   head_num,
                                                int   size_per_head,
                                                int   hidden_units_,
                                                int   inter_size_,
                                                void* buffer_in,
                                                bool  isAppend);
template void generate_xlnet_gemm_config<half>(int   batch_size,
                                               int   seq_len,
                                               int   head_num,
                                               int   size_per_head,
                                               int   hidden_units_,
                                               int   inter_size_,
                                               void* buffer_in,
                                               bool  isAppend);
#ifdef ENABLE_BF16
template void generate_xlnet_gemm_config<__nv_bfloat16>(int   batch_size,
                                                        int   seq_len,
                                                        int   head_num,
                                                        int   size_per_head,
                                                        int   hidden_units_,
                                                        int   inter_size_,
                                                        void* buffer_in,
                                                        bool  isAppend);
#endif

}  // namespace turbomind
