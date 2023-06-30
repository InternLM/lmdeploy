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

#include "src/fastertransformer/utils/gemm_test/t5_gemm_func.h"

namespace fastertransformer {

bool isSparseGemmAvailable(size_t m, size_t n, size_t k)
{
    return m % 8 == 0 && n % 8 == 0 && k % 8 == 0;
}

template<typename T>
void generate_t5_gemm_config(int   batch_size,
                             int   beam_width,
                             int   max_mem_seq_len,
                             int   encoder_d_model,
                             int   encoder_head_num,
                             int   encoder_size_per_head,
                             int   encoder_inter_size,
                             int   decoder_d_model,
                             int   decoder_head_num,
                             int   decoder_size_per_head,
                             int   decoder_inter_size,
                             int   decoder_vocab_size,
                             int   tensor_para_size,
                             void* buffer_in,
                             bool  isAppend,
                             bool  is_fp16_compute_type)
{
    FT_CHECK(encoder_head_num % tensor_para_size == 0);
    FT_CHECK(decoder_head_num % tensor_para_size == 0);

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

    const int gemm_num = 12;
    int       M[gemm_num];
    int       N[gemm_num];
    int       K[gemm_num];
    int       batchCount[gemm_num];
    char      mess[gemm_num][256];
    float     exec_times[gemm_num];

    // gemm 0
    M[0]          = batch_size * max_mem_seq_len;
    K[0]          = encoder_d_model;
    N[0]          = encoder_head_num / tensor_para_size * encoder_size_per_head;
    batchCount[0] = 3;
    strcpy(mess[0], "encoder from_tensor * batched gemm weightQKV");

    // gemm 1
    M[1]          = max_mem_seq_len;
    K[1]          = encoder_size_per_head;
    N[1]          = max_mem_seq_len;
    batchCount[1] = batch_size * encoder_head_num / tensor_para_size;
    strcpy(mess[1], "encoder batch strided gemm Q*K^T");

    // gemm 2
    M[2]          = max_mem_seq_len;
    K[2]          = max_mem_seq_len;
    N[2]          = encoder_size_per_head;
    batchCount[2] = batch_size * encoder_head_num / tensor_para_size;
    strcpy(mess[2], "encoder batch strided gemm QK*V^T");

    // gemm 3
    M[3]          = batch_size * max_mem_seq_len;
    K[3]          = encoder_head_num / tensor_para_size * encoder_size_per_head;
    N[3]          = encoder_d_model;
    batchCount[3] = 1;
    strcpy(mess[3], "encoder attr * output_kernel");

    // gemm 4
    M[4]          = batch_size * max_mem_seq_len;
    K[4]          = encoder_d_model;
    N[4]          = encoder_inter_size / tensor_para_size;
    batchCount[4] = 1;
    strcpy(mess[4], "encoder ffn gemm 1");

    // gemm 5
    M[5]          = batch_size * max_mem_seq_len;
    K[5]          = encoder_inter_size / tensor_para_size;
    N[5]          = encoder_d_model;
    batchCount[5] = 1;
    strcpy(mess[5], "encoder ffn gemm 2");

    // gemm 6
    M[6]          = batch_size * beam_width;
    K[6]          = decoder_d_model;
    N[6]          = 3 * decoder_head_num / tensor_para_size * decoder_size_per_head;
    batchCount[6] = 1;
    strcpy(mess[6], "from_tensor * weightQKV");

    // gemm 7
    M[7]          = batch_size * beam_width;
    K[7]          = decoder_head_num / tensor_para_size * decoder_size_per_head;
    N[7]          = decoder_d_model;
    batchCount[7] = 1;
    strcpy(mess[7], "attr * output_kernel");

    // gemm 8
    M[8]          = batch_size * beam_width;
    K[8]          = decoder_d_model;
    N[8]          = decoder_inter_size / tensor_para_size;
    batchCount[8] = 1;
    strcpy(mess[8], "ffn gemm 1");

    // gemm 9
    M[9]          = batch_size * beam_width;
    K[9]          = decoder_inter_size / tensor_para_size;
    N[9]          = decoder_d_model;
    batchCount[9] = 1;
    strcpy(mess[9], "ffn gemm 2");

    // gemm 10
    size_t decoder_vocab_size_padded = ((size_t)ceil(decoder_vocab_size / 1. / tensor_para_size) * tensor_para_size);
    if (!std::is_same<T, float>::value) {
        decoder_vocab_size_padded = ((size_t)ceil(decoder_vocab_size_padded / 8.) * 8);
    }
    M[10]          = batch_size * beam_width;
    K[10]          = decoder_d_model;
    N[10]          = decoder_vocab_size_padded / tensor_para_size;
    batchCount[10] = 1;
    strcpy(mess[10], "logits gemm");

    // gemm 11
    M[11]          = batch_size * max_mem_seq_len;
    K[11]          = encoder_d_model;
    N[11]          = encoder_head_num / tensor_para_size * encoder_size_per_head;
    batchCount[11] = 1;
    strcpy(mess[11], "encoder from_tensor * splited qkv weight");

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
    float f_alpha = (float)1.0f;
    float f_beta  = (float)0.0f;

    half h_alpha = (half)(1.0f);
    half h_beta  = (half)(0.0f);

    void* alpha = computeType == CUDA_R_16F ? (void*)(&h_alpha) : (void*)(&f_alpha);
    void* beta  = computeType == CUDA_R_16F ? (void*)(&h_beta) : (void*)(&f_beta);

    printf("***Encoder Gemm Testing Begin***\n");
    printf("***Cublas Gemm Testing Begin***\n");
    if (line_count == 0) {
        fprintf(fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, n, m, k, algoId, "
                "customOption, tile, numSplitsK, swizzle, reductionScheme, workspaceSize, stages, exec_time\n");
    }
    for (int i = 0; i < gemm_num; ++i) {
        int seq_len       = (i <= 5 || i == 11) ? max_mem_seq_len : 1;
        int head_num      = ((i <= 5 || i == 11) ? encoder_head_num : decoder_head_num) / tensor_para_size;
        int size_per_head = (i <= 5 || i == 11) ? encoder_size_per_head : decoder_size_per_head;

        int m = M[i], n = N[i], k = K[i];
        printf("\n-----------------------------\n");
        printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
        T* d_A = (T*)buffer;
        T* d_B = d_A + m * k * batchCount[i];
        T* d_C = d_B + k * n * batchCount[i];

        // array of pointer for batchedGemm
        T* harray[12];
        harray[0]  = (T*)buffer;
        harray[1]  = (T*)((char*)buffer + sizeof(T) * m * k);
        harray[2]  = (T*)((char*)buffer + 2 * sizeof(T) * m * k);
        harray[4]  = (T*)((char*)buffer + 3 * sizeof(T) * m * k);
        harray[5]  = (T*)((char*)buffer + 3 * sizeof(T) * m * k + sizeof(T) * k * n);
        harray[6]  = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 2 * sizeof(T) * k * n);
        harray[8]  = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n);
        harray[9]  = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n + sizeof(T) * m * n);
        harray[10] = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n + 2 * sizeof(T) * m * n);

        T** darray = 0;
        check_cuda_error(cudaMalloc((void**)&darray, sizeof(T*) * 12));
        cudaMemcpy((void*)darray, (void*)harray, sizeof(T*) * 12, cudaMemcpyHostToDevice);
        T** dAarray = darray;
        T** dBarray = darray + 4;
        T** dCarray = darray + 8;

        float exec_time = 99999.0f;
        int   fast_algo = 0;
        for (int algo = startAlgo; algo <= endAlgo; algo++) {
            cublasStatus_t status;
            cudaDeviceSynchronize();
            gettimeofday(&start, NULL);
            for (int ite = 0; ite < ites; ++ite) {
                if (i == 0) {
                    status = cublasGemmBatchedEx(cublas_handle,
                                                 CUBLAS_OP_N,
                                                 CUBLAS_OP_N,
                                                 n,
                                                 m,
                                                 k,
                                                 alpha,
                                                 (const void* const*)dBarray,
                                                 BType,
                                                 n,
                                                 (const void* const*)dAarray,
                                                 AType,
                                                 k,
                                                 beta,
                                                 (void* const*)dCarray,
                                                 CType,
                                                 n,
                                                 batchCount[i],
                                                 computeType,
                                                 static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 1) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_T,
                                                        CUBLAS_OP_N,
                                                        max_mem_seq_len,
                                                        max_mem_seq_len,
                                                        encoder_size_per_head,
                                                        alpha,
                                                        d_B,
                                                        BType,
                                                        encoder_size_per_head,
                                                        max_mem_seq_len * encoder_size_per_head,
                                                        d_A,
                                                        AType,
                                                        encoder_size_per_head,
                                                        max_mem_seq_len * encoder_size_per_head,
                                                        beta,
                                                        d_C,
                                                        CType,  // CType,
                                                        max_mem_seq_len,
                                                        max_mem_seq_len * max_mem_seq_len,
                                                        batchCount[i],
                                                        computeType,
                                                        static_cast<cublasGemmAlgo_t>(algo));
                }
                else if (i == 2) {
                    status = cublasGemmStridedBatchedEx(cublas_handle,
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        encoder_size_per_head,
                                                        max_mem_seq_len,
                                                        max_mem_seq_len,
                                                        alpha,
                                                        d_B,
                                                        BType,
                                                        encoder_size_per_head,
                                                        max_mem_seq_len * encoder_size_per_head,
                                                        d_A,
                                                        AType,
                                                        max_mem_seq_len,
                                                        max_mem_seq_len * max_mem_seq_len,
                                                        beta,
                                                        d_C,
                                                        CType,
                                                        encoder_size_per_head,
                                                        max_mem_seq_len * encoder_size_per_head,
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
                                          alpha,
                                          d_B,
                                          BType,
                                          k,
                                          d_A,
                                          AType,
                                          k,
                                          beta,
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
                                          alpha,
                                          d_B,
                                          BType,
                                          n,
                                          d_A,
                                          AType,
                                          k,
                                          beta,
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

        using scaleT = float;

        if (is_fp16_compute_type) {
            using scaleT = typename ScaleTypeConverter<T, true>::Type;
        }

        // for fp16 and bf16, we compare cublasLt
        if (data_type != FLOAT_DATATYPE && i != 1 && i != 2 && i != 0 && i != 10) {
            printf("***cublasLt Gemm Testing Begin***\n");
            // Let try a fixed number of combinations
            int                ALGO_COMBINATIONS = 5000;
            customMatmulPerf_t perfResults[ALGO_COMBINATIONS];

            // for t5, computeType & scaleType should be FP32
            if (is_fp16_compute_type) {
                using scaleT       = typename ScaleTypeConverter<T, true>::Type;
                scaleT alpha_scale = (scaleT)1.0f;
                scaleT beta_scale  = (scaleT)0.0f;

                LtHgemmCustomFind<T, scaleT>(ltHandle,
                                             m,
                                             seq_len,
                                             head_num,
                                             size_per_head,
                                             n,
                                             m,
                                             k,
                                             &(alpha_scale),
                                             d_B,
                                             d_A,
                                             &(beta_scale),
                                             d_C,
                                             cublas_workspace,
                                             workSpaceSize,
                                             fd,
                                             perfResults,
                                             ALGO_COMBINATIONS);
            }
            else {
                LtHgemmCustomFind<T, float>(ltHandle,
                                            m,
                                            seq_len,
                                            head_num,
                                            size_per_head,
                                            n,
                                            m,
                                            k,
                                            &(f_alpha),
                                            d_B,
                                            d_A,
                                            &(f_beta),
                                            d_C,
                                            cublas_workspace,
                                            workSpaceSize,
                                            fd,
                                            perfResults,
                                            ALGO_COMBINATIONS);
            }

            if (perfResults[0].time < exec_time) {
                printPerfStructure(batch_size * (i <= 5 || i == 1 ? 1 : beam_width),
                                   seq_len,
                                   head_num,
                                   size_per_head,
                                   n,
                                   m,
                                   k,
                                   perfResults[0],
                                   fd,
                                   data_type,
                                   0);
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
                        batch_size * (i <= 5 || i == 1 ? 1 : beam_width),
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
                    batch_size * (i <= 5 || i == 1 ? 1 : beam_width),
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
            int seq_len       = i <= 5 ? max_mem_seq_len : 1;
            int head_num      = (i <= 5 ? encoder_head_num : decoder_head_num) / tensor_para_size;
            int size_per_head = i <= 5 ? encoder_size_per_head : decoder_size_per_head;

            // to be compatible with spgemm wrapper, we let A be the weight matrix
            // so m and n are swapped
            // A: mxk B: kxn C:mxn
            int m = N[i], n = M[i], k = K[i];
            printf("\n-----------------------------\n");
            printf("GEMM test %d: [M: %d, K: %d, N: %d]\n", i, m, k, n);
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
                    cusparseLtMatDescriptor_t mat_A, matB, mat_C;
                    void*                     d_workspace = nullptr;
                    int                       num_streams = 1;
                    cudaStream_t              streams[1]  = {stream};
                    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
                        &handle, &mat_A, m, k, m, alignment, CUDA_R_16F, order, CUSPARSELT_SPARSITY_50_PERCENT))
                    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, k, alignment, CUDA_R_16F, order))
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

    printf("***T5 Gemm Testing End***\n");
    return;
}

template void generate_t5_gemm_config<float>(int   batch_size,
                                             int   beam_width,
                                             int   max_mem_seq_len,
                                             int   encoder_d_model,
                                             int   encoder_head_num,
                                             int   encoder_size_per_head,
                                             int   encoder_inter_size,
                                             int   decoder_d_model,
                                             int   decoder_head_num,
                                             int   decoder_size_per_head,
                                             int   decoder_inter_size,
                                             int   decoder_vocab_size,
                                             int   tensor_para_size,
                                             void* buffer_in,
                                             bool  isAppend,
                                             bool  is_fp16_compute_type);

template void generate_t5_gemm_config<half>(int   batch_size,
                                            int   beam_width,
                                            int   max_mem_seq_len,
                                            int   encoder_d_model,
                                            int   encoder_head_num,
                                            int   encoder_size_per_head,
                                            int   encoder_inter_size,
                                            int   decoder_d_model,
                                            int   decoder_head_num,
                                            int   decoder_size_per_head,
                                            int   decoder_inter_size,
                                            int   decoder_vocab_size,
                                            int   tensor_para_size,
                                            void* buffer_in,
                                            bool  isAppend,
                                            bool  is_fp16_compute_type);

#ifdef ENABLE_BF16
template void generate_t5_gemm_config<__nv_bfloat16>(int   batch_size,
                                                     int   beam_width,
                                                     int   max_mem_seq_len,
                                                     int   encoder_d_model,
                                                     int   encoder_head_num,
                                                     int   encoder_size_per_head,
                                                     int   encoder_inter_size,
                                                     int   decoder_d_model,
                                                     int   decoder_head_num,
                                                     int   decoder_size_per_head,
                                                     int   decoder_inter_size,
                                                     int   decoder_vocab_size,
                                                     int   tensor_para_size,
                                                     void* buffer_in,
                                                     bool  isAppend,
                                                     bool  is_fp16_compute_type);
#endif

size_t calT5GemmTestBufSizeInByte(int            batch_size,
                                  int            beam_width,
                                  int            max_mem_seq_len,
                                  int            encoder_d_model,
                                  int            encoder_head_num,
                                  int            encoder_size_per_head,
                                  int            encoder_inter_size,
                                  int            decoder_d_model,
                                  int            decoder_head_num,
                                  int            decoder_size_per_head,
                                  int            decoder_inter_size,
                                  int            decoder_vocab_size,
                                  int            tensor_para_size,
                                  CublasDataType data_type)
{
    const size_t local_encoder_head_num     = encoder_head_num / tensor_para_size;
    const size_t local_encoder_hidden_units = local_encoder_head_num * encoder_size_per_head;
    const size_t local_encoder_inter_size   = encoder_inter_size / tensor_para_size;
    const size_t local_decoder_head_num     = decoder_head_num / tensor_para_size;
    const size_t local_decoder_hidden_units = local_decoder_head_num * decoder_size_per_head;
    const size_t local_decoder_inter_size   = decoder_inter_size / tensor_para_size;

    size_t              m = batch_size * max_mem_seq_len;
    std::vector<size_t> buff_size;

    // encoder qkv gemm
    buff_size.push_back(
        3 * (m * encoder_d_model + encoder_d_model * local_encoder_hidden_units + m * local_encoder_hidden_units));
    // encoder batch gemm
    buff_size.push_back(m * local_encoder_hidden_units + m * local_encoder_hidden_units
                        + batch_size * beam_width * local_encoder_head_num * max_mem_seq_len * max_mem_seq_len);
    // encoder ffn gemm
    buff_size.push_back(m * local_encoder_inter_size + encoder_d_model * local_encoder_inter_size
                        + m * encoder_d_model);

    m = batch_size * beam_width;
    // decoder qkv gemm
    buff_size.push_back(m * decoder_d_model + decoder_d_model * 3 * local_decoder_hidden_units
                        + 3 * m * local_decoder_hidden_units);
    // decoder cross mem gemm
    buff_size.push_back(m * max_mem_seq_len * encoder_d_model + encoder_d_model * local_decoder_hidden_units
                        + m * max_mem_seq_len * local_decoder_hidden_units);
    // decoder ffn gemm
    buff_size.push_back(m * local_decoder_inter_size + decoder_d_model * local_decoder_inter_size
                        + m * decoder_d_model);
    // decoder vocab gemm
    size_t decoder_vocab_size_padded = ((size_t)ceil(decoder_vocab_size / 1. / tensor_para_size) * tensor_para_size);
    if (data_type != FLOAT_DATATYPE) {
        decoder_vocab_size_padded = ((size_t)ceil(decoder_vocab_size_padded / 8.) * 8);
    }
    buff_size.push_back(m * decoder_d_model + decoder_d_model * decoder_vocab_size_padded / tensor_para_size
                        + m * decoder_vocab_size_padded / tensor_para_size);

    size_t buf_size_in_byte = 0;
    // int wordSize = (data_type == FLOAT_DATATYPE ? sizeof(float) : sizeof(half));
    // Because we always use float for some buffer, set the wordSize to float directly.
    int wordSize = sizeof(float);
    for (auto t : buff_size) {
        buf_size_in_byte = buf_size_in_byte > t ? buf_size_in_byte : t;
    }
    buf_size_in_byte *= wordSize;
    buf_size_in_byte += ((data_type == HALF_DATATYPE || data_type == BFLOAT16_DATATYPE) ? CUBLAS_WORKSPACE_SIZE : 0);

    return buf_size_in_byte;
}

}  // namespace fastertransformer
