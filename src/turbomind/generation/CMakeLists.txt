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

cmake_minimum_required(VERSION 3.11)

add_library(guided_decoding STATIC guided_decoding.cc)
target_link_libraries(guided_decoding PRIVATE
    apply_token_bitmask_inplace_cuda
    xgrammar
    core)
set_property(TARGET guided_decoding PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(generation STATIC
    generation.cc
    logits_processor.cc
    sampling.cc
    stop_criteria.cc)
set_property(TARGET generation PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET generation PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(generation PUBLIC
    ban_bad_words
    sampling_penalty_kernels
    sampling_topk_kernels
    sampling_topp_kernels
    sampling_kernels
    stop_criteria
    guided_decoding
    memory_utils
    CUDA::cudart)
