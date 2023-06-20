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

add_library(BaseBeamSearchLayer STATIC BaseBeamSearchLayer.cu)
set_property(TARGET BaseBeamSearchLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BaseBeamSearchLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BaseBeamSearchLayer PUBLIC -lcudart beam_search_penalty_kernels cuda_utils)

add_library(OnlineBeamSearchLayer STATIC OnlineBeamSearchLayer.cu)
set_property(TARGET OnlineBeamSearchLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET OnlineBeamSearchLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(OnlineBeamSearchLayer PUBLIC -lcudart BaseBeamSearchLayer online_softmax_beamsearch_kernels)

add_library(BeamSearchLayer STATIC BeamSearchLayer.cu)
set_property(TARGET BeamSearchLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BeamSearchLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BeamSearchLayer PUBLIC -lcudart BaseBeamSearchLayer beam_search_topk_kernels)
