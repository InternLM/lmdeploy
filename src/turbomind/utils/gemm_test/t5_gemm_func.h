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

#pragma once

#include "src/turbomind/utils/cublasAlgoMap.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/gemm_test/gemm_func.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <map>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace turbomind {

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
                             bool  is_fp16_compute_type);

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
                                  CublasDataType data_type);

}  // namespace turbomind
