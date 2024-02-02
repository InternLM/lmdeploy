# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(ban_bad_words STATIC ban_bad_words.cu)
set_property(TARGET ban_bad_words PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ban_bad_words PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(stop_criteria STATIC stop_criteria_kernels.cu)
set_property(TARGET stop_criteria PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET stop_criteria PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(activation_kernels STATIC activation_kernels.cu)
set_property(TARGET activation_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(logprob_kernels STATIC logprob_kernels.cu)
set_property(TARGET logprob_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET logprob_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(unfused_attention_kernels STATIC unfused_attention_kernels.cu)
set_property(TARGET unfused_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET unfused_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(bert_preprocess_kernels STATIC bert_preprocess_kernels.cu)
set_property(TARGET bert_preprocess_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET bert_preprocess_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

set(decoder_masked_multihead_attention_files
    decoder_masked_multihead_attention.cu
)
file(GLOB decoder_masked_multihead_attention_files ${decoder_masked_multihead_attention_files} ./decoder_masked_multihead_attention/*.cu)
add_library(decoder_masked_multihead_attention STATIC ${decoder_masked_multihead_attention_files})
set_property(TARGET decoder_masked_multihead_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoder_masked_multihead_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(decoding_kernels STATIC decoding_kernels.cu)
set_property(TARGET decoding_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoding_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(gpt_kernels STATIC gpt_kernels.cu)
set_property(TARGET gpt_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gpt_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topk_kernels STATIC sampling_topk_kernels.cu)
set_property(TARGET sampling_topk_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topk_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topp_kernels STATIC sampling_topp_kernels.cu)
set_property(TARGET sampling_topp_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topp_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_penalty_kernels STATIC sampling_penalty_kernels.cu)
set_property(TARGET sampling_penalty_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_penalty_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(custom_ar_kernels STATIC custom_ar_kernels.cu)
set_property(TARGET custom_ar_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET custom_ar_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_subdirectory(gemm_s_f16)
add_subdirectory(attention)
