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

add_library(BanBadWordsLayer STATIC BanBadWordsLayer.cc)
set_property(TARGET BanBadWordsLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BanBadWordsLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BanBadWordsLayer PUBLIC CUDA::cudart ban_bad_words memory_utils)

add_library(PenaltyLayer STATIC PenaltyLayer.cc)
set_property(TARGET PenaltyLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET PenaltyLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(PenaltyLayer PUBLIC CUDA::cudart sampling_penalty_kernels memory_utils)

add_library(SamplingLayer STATIC SamplingLayer.cc)
set_property(TARGET SamplingLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET SamplingLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(SamplingLayer PUBLIC CUDA::cudart memory_utils sampling_utils
    sampling_topk_kernels sampling_topp_kernels
)

add_library(StopCriteriaLayer STATIC StopCriteriaLayer.cc)
set_property(TARGET StopCriteriaLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET StopCriteriaLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(StopCriteriaLayer PUBLIC CUDA::cudart stop_criteria memory_utils)
