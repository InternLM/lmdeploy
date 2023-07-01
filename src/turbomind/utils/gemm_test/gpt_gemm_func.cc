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

#include "src/turbomind/utils/gemm_test/gpt_gemm_func.h"

namespace turbomind {

bool isSparseGemmAvailable(size_t m, size_t n, size_t k)
{
    return m % 8 == 0 && n % 8 == 0 && k % 8 == 0;
}

template<typename T>
void generate_gpt_gemm_config(int   batch_size,
                              int   beam_width,
                              int   max_input_len,
                              int   head_num,
                              int   size_per_head,
                              int   inter_size,
                              int   vocab_size,
                              int   tensor_para_size,
                              void* buffer_in,
                              bool  isAppend)
{
    FT_CHECK(head_num % tensor_para_size == 0);
    void* cublas_workspace;
    void* buffer;
    int   workSpaceSize;
#if 0
    bool  workspace_flag = std::is_same<T, half>::value;
#ifdef ENABLE_FP8
    workspace_flag = workspace_flag || std::is_same<T, __nv_fp8_e4m3>::value;
#endif
#if ENABLE_BF16
    workspace_flag = workspace_flag || std::is_same<T, __nv_bfloat16>::value;
#endif
#endif
    // algorithms with workspace perform worse than evaluated
    const bool workspace_flag = 0;
    if (workspace_flag) {
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
        // if (config.size() >= (MAX_CONFIG_NUM * GEMM_NUM + 1))  // 6 cublas/cublasLt, first row is not included
        // {
        //     int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * GEMM_NUM);
        //     fclose(fd);
        //     fd = fopen(GEMM_CONFIG, "w+");
        //     fprintf(fd, "%s", config[0].c_str());
        //     for (uint i = startIdx; i < config.size(); i++) {
        //         fprintf(fd, "%s", config[i].c_str());
        //     }
        //     line_count = config.size() - (GEMM_NUM + 3);
        // }
    }

    const int hidden_units         = head_num * size_per_head;
    const int local_head_num       = head_num / tensor_para_size;
    const int local_hidden_units   = local_head_num * size_per_head;
    const int max_input_len_padded = (max_input_len + 15) / 16 * 16;
    const int gemm_num             = 11;
    int       M[gemm_num];
    int       N[gemm_num];
    int       K[gemm_num];
    int       batchCount[gemm_num];
    int64_t   strideA[gemm_num];
    int64_t   strideB[gemm_num];
    int64_t   strideD[gemm_num];
    char      mess[gemm_num][256];
    float     exec_times[gemm_num];

    // gemm 0
    M[0]          = batch_size * beam_width * max_input_len;
    K[0]          = hidden_units;
    N[0]          = 3 * local_hidden_units;
    batchCount[0] = 1;
    strideA[0]    = 0;
    strideB[0]    = 0;
    strideD[0]    = 0;
    strcpy(mess[0], "context from_tensor * weightQKV");

    // gemm 1
    M[1]          = max_input_len_padded;
    K[1]          = size_per_head;
    N[1]          = max_input_len_padded;
    batchCount[1] = batch_size * beam_width * local_head_num;
    strideA[1]    = max_input_len_padded * size_per_head;
    strideB[1]    = max_input_len_padded * size_per_head;
    strideD[1]    = max_input_len_padded * max_input_len_padded;
    strcpy(mess[1], "context batch gemm Q*K^T");

    // gemm 2
    M[2]          = max_input_len_padded;
    K[2]          = max_input_len_padded;
    N[2]          = size_per_head;
    batchCount[2] = batch_size * beam_width * local_head_num;
    strideA[2]    = max_input_len_padded * size_per_head;
    strideB[2]    = max_input_len_padded * max_input_len_padded;
    strideD[2]    = max_input_len_padded * size_per_head;
    strcpy(mess[2], "context batch gemm QK*V^T");

    // gemm 3
    M[3]          = batch_size * beam_width * max_input_len;
    K[3]          = local_hidden_units;
    N[3]          = hidden_units;
    batchCount[3] = 1;
    strideA[3]    = 0;
    strideB[3]    = 0;
    strideD[3]    = 0;
    strcpy(mess[3], "context attr * output_kernel");

    // gemm 4
    M[4]          = batch_size * beam_width * max_input_len;
    K[4]          = hidden_units;
    N[4]          = inter_size / tensor_para_size;
    batchCount[4] = 1;
    strideA[4]    = 0;
    strideB[4]    = 0;
    strideD[4]    = 0;
    strcpy(mess[4], "context ffn gemm 1");

    // gemm 5
    M[5]          = batch_size * beam_width * max_input_len;
    K[5]          = inter_size / tensor_para_size;
    N[5]          = hidden_units;
    batchCount[5] = 1;
    strideA[5]    = 0;
    strideB[5]    = 0;
    strideD[5]    = 0;
    strcpy(mess[5], "context ffn gemm 2");

    // gemm 6
    M[6]          = batch_size * beam_width;
    K[6]          = hidden_units;
    N[6]          = 3 * local_hidden_units;
    batchCount[6] = 1;
    strideA[6]    = 0;
    strideB[6]    = 0;
    strideD[6]    = 0;
    strcpy(mess[6], "from_tensor * weightQKV");

    // gemm 7
    M[7]          = batch_size * beam_width;
    K[7]          = local_hidden_units;
    N[7]          = hidden_units;
    batchCount[7] = 1;
    strideA[7]    = 0;
    strideB[7]    = 0;
    strideD[7]    = 0;
    strcpy(mess[7], "attr * output_kernel");

    // gemm 8
    M[8]          = batch_size * beam_width;
    K[8]          = hidden_units;
    N[8]          = inter_size / tensor_para_size;
    batchCount[8] = 1;
    strideA[8]    = 0;
    strideB[8]    = 0;
    strideD[8]    = 0;
    strcpy(mess[8], "ffn gemm 1");

    // gemm 9
    M[9]          = batch_size * beam_width;
    K[9]          = inter_size / tensor_para_size;
    N[9]          = hidden_units;
    batchCount[9] = 1;
    strideA[9]    = 0;
    strideB[9]    = 0;
    strideD[9]    = 0;
    strcpy(mess[9], "ffn gemm 2");

    // gemm 10
    M[10]          = batch_size * beam_width;
    K[10]          = hidden_units;
    N[10]          = ceil(vocab_size / 8.) * 8 / tensor_para_size;
    batchCount[10] = 1;
    strideA[10]    = 0;
    strideB[10]    = 0;
    strideD[10]    = 0;
    strcpy(mess[10], "logits gemm");

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    cublasLtHandle_t ltHandle;
    check_cuda_error(cublasLtCreate(&ltHandle));

    cudaDataType_t AType;
    cudaDataType_t BType;
    cudaDataType_t CType;
    cudaDataType_t DType;
    cudaDataType_t DType_FP8[gemm_num];
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
        DType       = CUDA_R_32F;
        computeType = CUDA_R_32F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT;
        endAlgo     = (int)CUBLAS_GEMM_ALGO23;
    }
    else if (std::is_same<T, half>::value) {
        data_type   = HALF_DATATYPE;
        AType       = CUDA_R_16F;
        BType       = CUDA_R_16F;
        CType       = CUDA_R_16F;
        DType       = CUDA_R_16F;
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
        DType       = CUDA_R_16BF;
        computeType = CUDA_R_32F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo     = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }
#endif
#ifdef ENABLE_FP8
    else if (std::is_same<T, __nv_fp8_e4m3>::value) {
        data_type = FP8_DATATYPE;
        AType     = CUDA_R_8F_E4M3;
        BType     = CUDA_R_8F_E4M3;
        CType     = CUDA_R_16BF;
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
        DType = CUDA_R_16BF
#else
        DType_FP8[0] = CUDA_R_8F_E4M3;
        DType_FP8[1] = CUDA_R_16BF;
        DType_FP8[2] = CUDA_R_8F_E4M3;
        DType_FP8[3] = CUDA_R_16BF;
        DType_FP8[4] = CUDA_R_16BF;
        DType_FP8[5] = CUDA_R_16BF;
#ifdef FP8_MHA
        DType_FP8[6] = CUDA_R_8F_E4M3;
#else
        DType_FP8[6] = CUDA_R_16BF;
#endif
        DType_FP8[7] = CUDA_R_16BF;
        DType_FP8[8] = CUDA_R_16BF;
        DType_FP8[9] = CUDA_R_16BF;
#endif
            computeType = CUDA_R_32F;
        startAlgo       = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo         = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    }
#endif
    float alpha = (float)1.0f;
    float beta  = (float)0.0f;

    printf("***Encoder Gemm Testing Begin***\n");
    printf("***Cublas Gemm Testing Begin***\n");
    if (line_count == 0) {
        fprintf(fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, n, m, k, algoId, "
                "customOption, tile, numSplitsK, swizzle, reductionScheme, workspaceSize, stages, "
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
                "inner_shapeId, cluster_shapeId, "
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
                "mma_shapeId, cga_shapeId, schedule_mode, "
#endif
                "exec_time\n");
    }

    for (int i = 0; i < gemm_num; ++i) {
        // tuning of context gemm and logits gemm is not working yet
        if (i <= 5 || i == 10) {
            continue;
        }
        int seq_len = i <= 5 ? max_input_len : 1;

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
                if (i == 1) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        max_input_len,
                                                        max_input_len,
                                                        size_per_head,
                                                        &alpha,
                                                        d_B,
                                                        BType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        d_A,
                                                        AType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        &beta,
                                                        d_C,
                                                        CUDA_R_32F,  // CType,
                                                        max_input_len,
                                                        max_input_len * max_input_len,
                                                        batchCount[i],
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 2) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        size_per_head,
                                                        max_input_len,
                                                        max_input_len,
                                                        &alpha,
                                                        d_B,
                                                        BType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        d_A,
                                                        AType,
                                                        max_input_len,
                                                        max_input_len * max_input_len,
                                                        &beta,
                                                        d_C,
                                                        CType,
                                                        size_per_head,
                                                        max_input_len * size_per_head,
                                                        batchCount[i],
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 10) {
                    status = cublasGemmEx(cublas_handle,
                                          CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          n,
                                          m,
                                          k,
                                          &alpha,
                                          d_B,
                                          BType,
                                          k,
                                          d_A,
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
                    status = cublasGemmEx(cublas_handle,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          n,
                                          m,
                                          k,
                                          &alpha,
                                          d_B,
                                          BType,
                                          n,
                                          d_A,
                                          AType,
                                          k,
                                          &beta,
                                          d_C,
                                          CType,
                                          n,
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
                }
            }
            sync_check_cuda_error();
        }

        printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);

        // for fp16 and bf16, we compare cublasLt
        // for fp8, compare cublaslt for all gemm kernels
        if ((data_type != FLOAT_DATATYPE && i != 1 && i != 2 && i != 10) || data_type == FP8_DATATYPE) {
            printf("***cublasLt Gemm Testing Beign***\n");
            // Let try a fixed number of combinations
            int                ALGO_COMBINATIONS = 10000;
            customMatmulPerf_t perfResults[ALGO_COMBINATIONS];

            // for gpt, computeType & scaleType should be FP32
            LtHgemmCustomFind<T, float>(ltHandle,
                                        batch_size * beam_width,
                                        i == 1 || i == 2 ? max_input_len : 1,
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
                                        ALGO_COMBINATIONS,
                                        DType_FP8[i],
                                        batchCount[i],
                                        strideA[i],
                                        strideB[i],
                                        strideD[i]);
            if (perfResults[0].time < exec_time) {
                printPerfStructure(batch_size * beam_width,
                                   seq_len,
                                   head_num,
                                   size_per_head,
                                   n,
                                   m,
                                   k,
                                   perfResults[0],
                                   fd,
                                   data_type,
                                   0,
                                   batchCount[i]);
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
                        batch_size * beam_width,
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
                    batch_size * beam_width,
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
        sync_check_cuda_error();
        exec_times[i] = exec_time;
    }
    printf("***cublas Gemm Testing End***\n\n");
    fclose(fd);

#ifdef SPARSITY_ENABLED
    bool do_sparse_test = false;
    if (prop.major == 8 && (prop.minor == 0 || prop.minor == 6) && sizeof(T) == sizeof(half)) {
        do_sparse_test = true;
    }
    if (do_sparse_test) {
        printf("***cusparseLt Gemm Testing Begin***\n");
        // Only first 8 cases can be sparse
        // - QKV kernel, Projection, FC1, FC2 in context or decoding.
        const int spgemm_num = 8;
        if (!isAppend) {
            fd = fopen(SPGEMM_CONFIG, "w+");
        }
        else {
            fd = fopen(SPGEMM_CONFIG, "a+");
            std::vector<std::string> config;
            char                     line[1024];
            while (fgets(line, 1024, fd) != NULL) {
                config.push_back(std::string(line));
            }
            line_count = config.size();
            // gemm_num configs (cublas/cublasLt), first row is not included
            if (config.size() >= (MAX_CONFIG_NUM * spgemm_num + 1)) {
                int startIdx = config.size() - ((MAX_CONFIG_NUM - 1) * spgemm_num);
                fclose(fd);
                fd = fopen(SPGEMM_CONFIG, "w+");
                fprintf(fd, "%s", config[0].c_str());
                for (uint i = startIdx; i < config.size(); i++) {
                    fprintf(fd, "%s", config[i].c_str());
                }
                line_count = config.size() - (spgemm_num + 3);
            }
        }
        if (line_count == 0) {
            // header line
            fprintf(fd,
                    "batch_size, seq_len, head_num, size_per_head dataType "
                    "### batchCount, m, n, k, algoId, exec_time\n");
        }

        cusparseLtHandle_t handle;
        CHECK_CUSPARSE(cusparseLtInit(&handle));
        cusparseOrder_t     order = CUSPARSE_ORDER_COL;
        cusparseOperation_t opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
        // let's make this optional
        cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;
        unsigned            alignment    = 16;
        cudaStream_t        stream       = 0;
        float               alpha2       = 1.0f;
        float               beta2        = 0.0f;
        for (int i = 0; i < gemm_num; ++i) {
            // skip qk or attn or logit gemms.
            if (i == 1 || i == 2 || i == 10) {
                continue;
            }

            // seq_len is always 1 except context gemms.
            int seq_len = i <= 5 ? max_input_len : 1;

            // to be compatible with spgemm wrapper, we let A be the weight matrix
            // so m and n are swapped
            // A: mxk B: kxn C:mxn
            int m = N[i], n = M[i], k = K[i];
            printf("\n-----------------------------\n");
            printf("GEMM test %d: [M: %d, K: %d, N: %d]\n", i, m, k, n);

            if (n % 8 != 0) {
                n = div_up(n, 8) * 8;  // pad n to be multiple of 8 as FT does.
            }

            T* d_A = (T*)buffer;
            T* d_B = d_A + m * k * batchCount[i];
            T* d_C = d_B + k * n * batchCount[i];
            T* dA_compressed;
            {
                cusparseLtMatDescriptor_t mat_A;
                CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                    &handle, &mat_A, m, k, m, alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT))
                CHECK_CUSPARSE(
                    cusparseLtSpMMAPrune2(&handle, &mat_A, true, opA, d_A, d_A, CUSPARSELT_PRUNE_SPMMA_STRIP, stream))
                size_t compressed_size;
                CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, &mat_A, &compressed_size))
                check_cuda_error(cudaMalloc((void**)&dA_compressed, compressed_size));
                CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, &mat_A, true, opA, d_A, dA_compressed, stream))
            }

            float exec_time = 99999.0f;
            int   fast_algo = 0;
            if (isSparseGemmAvailable(m, n, k)) {
                for (int alg = 0; alg < 4; ++alg) {
                    cudaDeviceSynchronize();
                    cusparseLtMatDescriptor_t mat_A, mat_B, mat_C;
                    void*                     d_workspace = nullptr;
                    int                       num_streams = 1;
                    cudaStream_t              streams[1]  = {stream};
                    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                        &handle, &mat_A, m, k, m, alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT))
                    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &mat_B, k, n, k, alignment, CUDA_R_16F, order))
                    CHECK_CUSPARSE(
                        cusparseLtDenseDescriptorInit(&handle, &mat_C, m, n, m, alignment, CUDA_R_16F, order))
                    cudaDeviceSynchronize();
                    gettimeofday(&start, NULL);
                    for (int ite = 0; ite < ites; ++ite) {
                        // initializing MatDesc takes a lot of time
                        // and these descs can be stored to other place
                        // whereas storing MatMulPlan to other place will cause errors
                        cusparseLtMatmulDescriptor_t   matmul;
                        cusparseLtMatmulAlgSelection_t alg_sel;
                        cusparseLtMatmulPlan_t         plan;
                        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
                            &handle, &matmul, opA, opB, &mat_A, &mat_B, &mat_C, &mat_C, compute_type))
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
            }
            exec_time /= ites;
            if (exec_time >= exec_times[i]) {
                fast_algo = -1;
            }
            printf("fast_algo %d\n", fast_algo);
            fprintf(fd,
                    "%d %d %d %d %d ### %d %d %d %d %d %f\n",
                    batch_size * beam_width,
                    seq_len,
                    head_num,
                    size_per_head,
                    data_type,
                    batchCount[i],
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

    printf("***GPT Gemm Testing End***\n");
    return;
}

template void generate_gpt_gemm_config<float>(int   batch_size,
                                              int   beam_width,
                                              int   max_input_len,
                                              int   head_num,
                                              int   size_per_head,
                                              int   inter_size,
                                              int   vocab_size,
                                              int   tensor_para_size,
                                              void* buffer_in,
                                              bool  isAppend);

template void generate_gpt_gemm_config<half>(int   batch_size,
                                             int   beam_width,
                                             int   max_input_len,
                                             int   head_num,
                                             int   size_per_head,
                                             int   inter_size,
                                             int   vocab_size,
                                             int   tensor_para_size,
                                             void* buffer_in,
                                             bool  isAppend);

#ifdef ENABLE_BF16
template void generate_gpt_gemm_config<__nv_bfloat16>(int   batch_size,
                                                      int   beam_width,
                                                      int   max_input_len,
                                                      int   head_num,
                                                      int   size_per_head,
                                                      int   inter_size,
                                                      int   vocab_size,
                                                      int   tensor_para_size,
                                                      void* buffer_in,
                                                      bool  isAppend);
#endif

#ifdef ENABLE_FP8
template void generate_gpt_gemm_config<__nv_fp8_e4m3>(int   batch_size,
                                                      int   beam_width,
                                                      int   max_input_len,
                                                      int   head_num,
                                                      int   size_per_head,
                                                      int   inter_size,
                                                      int   vocab_size,
                                                      int   tensor_para_size,
                                                      void* buffer_in,
                                                      bool  isAppend);
#endif

size_t calGptGemmTestBufSizeInByte(int            batch_size,
                                   int            beam_width,
                                   int            max_input_len,
                                   int            head_num,
                                   int            size_per_head,
                                   int            inter_size,
                                   int            vocab_size,
                                   int            tensor_para_size,
                                   CublasDataType data_type)
{
    size_t       buf_size_in_byte   = 0;
    const size_t hidden_units       = head_num * size_per_head;
    const size_t local_head_num     = head_num / tensor_para_size;
    const size_t local_hidden_units = local_head_num * size_per_head;

    // int wordSize = (data_type == FLOAT_DATATYPE ? sizeof(float) : sizeof(half));
    // Because we always use float for some buffer, set the wordSize to float directly.
    int wordSize = sizeof(float);

    size_t              m = batch_size * beam_width * max_input_len;
    std::vector<size_t> buff_size;
    // for context qkv gemm
    buff_size.push_back(m * hidden_units + hidden_units * 3 * local_hidden_units + m * 3 * local_hidden_units);
    // for context batch gemm
    buff_size.push_back(m * local_hidden_units + m * local_hidden_units
                        + batch_size * beam_width * head_num * max_input_len * max_input_len);
    // for context ffn gemm
    buff_size.push_back(m * inter_size / tensor_para_size + hidden_units * inter_size / tensor_para_size
                        + m * hidden_units);
    // for vocab
    buff_size.push_back(m * hidden_units + hidden_units * ceil(vocab_size / 8.) * 8 / tensor_para_size
                        + m * ceil(vocab_size / 8.) * 8 / tensor_para_size);

    for (auto t : buff_size) {
        buf_size_in_byte = buf_size_in_byte > t ? buf_size_in_byte : t;
    }
    buf_size_in_byte *= wordSize;
    buf_size_in_byte += ((data_type == HALF_DATATYPE || data_type == BFLOAT16_DATATYPE || data_type == FP8_DATATYPE) ?
                             CUBLAS_WORKSPACE_SIZE :
                             0);

    return buf_size_in_byte;
}

}  // namespace turbomind
