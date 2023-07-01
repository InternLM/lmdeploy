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

#include "src/fastertransformer/utils/gemm_test/decoding_gemm_func.h"

namespace fastertransformer {

template<typename T>
void generate_decoding_gemm_config(int   batch_size,
                                   int   beam_width,
                                   int   max_mem_seq_len,
                                   int   head_num,
                                   int   size_per_head,
                                   int   inter_size,
                                   int   vocab_size,
                                   int   mem_hidden_units,
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

    const int hidden_units = head_num * size_per_head;
    const int gemm_num     = 6;
    int       M[gemm_num];
    int       N[gemm_num];
    int       K[gemm_num];
    int       batchCount[gemm_num] = {1, 1, 1, 1, 1, 1};
    char      mess[gemm_num][256];

    // gemm 0
    M[0] = batch_size * beam_width;
    K[0] = hidden_units;
    N[0] = K[0] * 3;
    strcpy(mess[0], "from_tensor * weightQKV");

    // gemm 1
    M[1] = batch_size * beam_width;
    K[1] = hidden_units;
    N[1] = K[1];
    strcpy(mess[1], "attr * output_kernel");

    // gemm2
    M[2] = batch_size * beam_width * max_mem_seq_len;
    K[2] = mem_hidden_units;
    N[2] = hidden_units;
    strcpy(mess[2], "mem_tensor * weightK/V in cross attention");

    // gemm 3
    M[3] = batch_size * beam_width;
    K[3] = hidden_units;
    N[3] = inter_size;
    strcpy(mess[3], "ffn gemm1 ");

    // gemm 4
    M[4] = batch_size * beam_width;
    K[4] = inter_size;
    N[4] = hidden_units;
    strcpy(mess[4], "ffn gemm2");

    // gemm5
    M[5] = batch_size * beam_width;
    K[5] = hidden_units;
    N[5] = ceil(vocab_size / 8.) * 8;
    strcpy(mess[5], "decoder_output * embedding_kernel -> embedding_output");

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
    using scaleT = typename ScaleTypeConverter<T>::Type;

    scaleT alpha = (scaleT)1.0f;
    scaleT beta  = (scaleT)0.0f;

    printf("***Encoder Gemm Testing Begin***\n");
    printf("***Cublas Gemm Testing Begin***\n");
    if (line_count == 0) {
        fprintf(fd,
                "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, n, m, k, algoId, "
                "customOption, tile, numSplitsK, swizzle, reductionScheme, workspaceSize, stages, exec_time\n");
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
        int   seq_len   = i == 2 ? max_mem_seq_len : 1;
        for (int algo = startAlgo; algo <= endAlgo; algo++) {
            cublasStatus_t status;
            cudaDeviceSynchronize();
            gettimeofday(&start, NULL);
            for (int ite = 0; ite < ites; ++ite) {
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
        }
        printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);

        // for fp16 and bf16, we compare cublasLt
        if (data_type != FLOAT_DATATYPE) {
            printf("***cublasLt Gemm Testing Begin***\n");
            // Let try a fixed number of combinations
            int                ALGO_COMBINATIONS = 5000;
            customMatmulPerf_t perfResults[ALGO_COMBINATIONS];

            LtHgemmCustomFind<T, scaleT>(ltHandle,
                                         batch_size * beam_width,
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
    }
    printf("***cublas Gemm Testing End***\n\n");
    fclose(fd);
    printf("***Decoding Gemm Testing End***\n");
    return;
}

template void generate_decoding_gemm_config<float>(int   batch_size,
                                                   int   beam_width,
                                                   int   seq_len,
                                                   int   head_num,
                                                   int   size_per_head,
                                                   int   inter_size,
                                                   int   vocab_size,
                                                   int   mem_hidden_units,
                                                   void* buffer_in,
                                                   bool  isAppend);

template void generate_decoding_gemm_config<half>(int   batch_size,
                                                  int   beam_width,
                                                  int   seq_len,
                                                  int   head_num,
                                                  int   size_per_head,
                                                  int   inter_size,
                                                  int   vocab_size,
                                                  int   mem_hidden_units,
                                                  void* buffer_in,
                                                  bool  isAppend);

#ifdef ENABLE_BF16
template void generate_decoding_gemm_config<__nv_bfloat16>(int   batch_size,
                                                           int   beam_width,
                                                           int   seq_len,
                                                           int   head_num,
                                                           int   size_per_head,
                                                           int   inter_size,
                                                           int   vocab_size,
                                                           int   mem_hidden_units,
                                                           void* buffer_in,
                                                           bool  isAppend);
#endif

size_t calDecodingGemmTestBufSizeInByte(int            batch_size,
                                        int            beam_width,
                                        int            max_mem_seq_len,
                                        int            head_num,
                                        int            size_per_head,
                                        int            inter_size,
                                        int            memory_hidden_units,
                                        int            vocab_size,
                                        CublasDataType data_type)
{
    size_t       buf_size_in_byte   = 0;
    const size_t tensor_para_size   = 1;
    const size_t hidden_units       = head_num * size_per_head;
    const size_t local_head_num     = head_num / tensor_para_size;
    const size_t local_hidden_units = local_head_num * size_per_head;

    // int wordSize = (data_type == FLOAT_DATATYPE ? sizeof(float) : sizeof(half));
    // Because we always use float for some buffer, set the wordSize to float directly.
    int wordSize = sizeof(float);

    size_t              m = batch_size * beam_width;
    std::vector<size_t> buff_size;
    // for qkv gemm
    buff_size.push_back(m * hidden_units + hidden_units * 3 * local_hidden_units + m * 3 * local_hidden_units);
    // for attention output gemm
    buff_size.push_back(m * hidden_units + hidden_units * local_hidden_units + m * local_hidden_units);
    // for memory_tensor gemm
    buff_size.push_back(m * max_mem_seq_len * memory_hidden_units + memory_hidden_units * local_hidden_units
                        + m * max_mem_seq_len * local_hidden_units);
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
    buf_size_in_byte += ((data_type == HALF_DATATYPE || data_type == BFLOAT16_DATATYPE) ? CUBLAS_WORKSPACE_SIZE : 0);

    return buf_size_in_byte;
}

}  // namespace fastertransformer
