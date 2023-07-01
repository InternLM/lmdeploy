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

// Copied from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/gpt_gemm.cc

#include "src/turbomind/utils/gemm_test/gpt_gemm_func.h"
#include "src/turbomind/utils/memory_utils.h"

namespace ft = turbomind;

int main(int argc, char* argv[])
{
    if (argc < 9 || argc > 11) {
        TM_LOG_ERROR("./bin/llama_gemm batch_size \\ \n"
                     "                 beam_width \\ \n"
                     "                 max_input_len \\ \n"
                     "                 head_number \\ \n"
                     "                 size_per_head \\ \n"
                     "                 inter_size \\ \n"
                     "                 vocab_size \\ \n"
                     "                 data_type \\ \n"
                     "                 tensor_para_size \\\n"
                     "                 is_append (append new config into exist gemm_config.ini or not)");
        TM_LOG_ERROR("e.g. ./bin/llama_gemm 8 4 32 96 128 49152 51200 1 8 1");
        return 0;
    }

    const int                batch_size    = atoi(argv[1]);
    const int                beam_width    = atoi(argv[2]);
    const int                max_input_len = atoi(argv[3]);
    const int                head_num      = atoi(argv[4]);
    const int                size_per_head = atoi(argv[5]);
    const int                inter_size    = atoi(argv[6]);
    const int                vocab_size    = atoi(argv[7]);
    const ft::CublasDataType data_type     = static_cast<ft::CublasDataType>(atoi(argv[8]));  // 0 FP32, 1 FP16, 2 BF 16
    const int                tensor_para_size = argc < 10 ? 1 : atoi(argv[9]);
    const bool               is_append        = argc < 11 ? false : (bool)(atoi(argv[10]));

    TM_LOG_INFO("Arguments:");
    TM_LOG_INFO("  batch_size: %d", batch_size);
    TM_LOG_INFO("  beam_width: %d", beam_width);
    TM_LOG_INFO("  max_input_len: %d", max_input_len);
    TM_LOG_INFO("  head_num: %d", head_num);
    TM_LOG_INFO("  size_per_head: %d", size_per_head);
    TM_LOG_INFO("  inter_size: %d", inter_size);
    TM_LOG_INFO("  vocab_size: %d", vocab_size);
    TM_LOG_INFO("  data_type: %d", data_type);
    TM_LOG_INFO("  tensor_para_size: %d", tensor_para_size);
    TM_LOG_INFO("  is_append: %d", (int)is_append);
    std::cout << std::endl;

    void*  gemm_test_buf;
    size_t buf_size_in_byte = ft::calGptGemmTestBufSizeInByte(batch_size,
                                                              beam_width,
                                                              max_input_len,
                                                              head_num,
                                                              size_per_head,
                                                              inter_size,
                                                              vocab_size,
                                                              tensor_para_size,
                                                              data_type);
    size_t total, free;
    ft::check_cuda_error(cudaMemGetInfo(&free, &total));
    if (free < buf_size_in_byte + 10 * 1024 * 1024) {
        printf("[ERROR] There is no enough device memory for gemm test!\n"
               " %ld Bytes is needed, but only %ld Bytes is free.\n",
               buf_size_in_byte,
               free);
        gemm_test_buf = NULL;
        return -1;
    }
    else {
        ft::deviceMalloc(reinterpret_cast<char**>(&gemm_test_buf), buf_size_in_byte, false);
    }

    if (data_type == ft::FLOAT_DATATYPE) {
        ft::generate_gpt_gemm_config<float>(batch_size,
                                            beam_width,
                                            max_input_len,
                                            head_num,
                                            size_per_head,
                                            inter_size,
                                            vocab_size,
                                            tensor_para_size,
                                            gemm_test_buf,
                                            is_append);
    }
    else if (data_type == ft::HALF_DATATYPE) {
        ft::generate_gpt_gemm_config<half>(batch_size,
                                           beam_width,
                                           max_input_len,
                                           head_num,
                                           size_per_head,
                                           inter_size,
                                           vocab_size,
                                           tensor_para_size,
                                           gemm_test_buf,
                                           is_append);
    }
#ifdef ENABLE_BF16
    else if (data_type == ft::BFLOAT16_DATATYPE) {
        ft::generate_gpt_gemm_config<__nv_bfloat16>(batch_size,
                                                    beam_width,
                                                    max_input_len,
                                                    head_num,
                                                    size_per_head,
                                                    inter_size,
                                                    vocab_size,
                                                    tensor_para_size,
                                                    gemm_test_buf,
                                                    is_append);
    }
#endif
#ifdef ENABLE_FP8
    else if (data_type == ft::FP8_DATATYPE) {
        ft::generate_gpt_gemm_config<__nv_fp8_e4m3>(batch_size,
                                                    beam_width,
                                                    max_input_len,
                                                    head_num,
                                                    size_per_head,
                                                    inter_size,
                                                    vocab_size,
                                                    tensor_para_size,
                                                    gemm_test_buf,
                                                    false);
    }
#endif
    else {
        printf("[ERROR] data type only supports fp32(0), fp16(1), bf16(2), fp8(4). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
