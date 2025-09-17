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

find_package(CUDAToolkit REQUIRED)

add_library(LogitsProcessorLayer STATIC LogitsProcessorLayer.cc)
set_property(TARGET LogitsProcessorLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LogitsProcessorLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LogitsProcessorLayer PUBLIC CUDA::cudart ban_bad_words memory_utils
    sampling_penalty_kernels xgrammar
)

add_library(SamplingLayer STATIC SamplingLayer.cc)
set_property(TARGET SamplingLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SamplingLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SamplingLayer PUBLIC CUDA::cudart memory_utils
    sampling_topk_kernels sampling_topp_kernels sampling_kernels xgrammar
)

add_library(StopCriteriaLayer STATIC StopCriteriaLayer.cc)
set_property(TARGET StopCriteriaLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET StopCriteriaLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(StopCriteriaLayer PUBLIC CUDA::cudart stop_criteria memory_utils xgrammar)

add_library(GuidedDecodeLayer STATIC GuidedDecodeMaskLayer.cc GuidedDecodeUpdateLayer.cc)
set_property(TARGET GuidedDecodeLayer PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET GuidedDecodeLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GuidedDecodeLayer PUBLIC CUDA::cudart apply_token_bitmask_inplace_cuda xgrammar)
