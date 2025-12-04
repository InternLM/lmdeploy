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

find_package(CUDAToolkit REQUIRED)

add_library(logger STATIC logger.cc)
set_property(TARGET logger PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET logger PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(logger PUBLIC CUDA::cudart)


add_library(cuda_utils STATIC cuda_utils.cc)
set_property(TARGET cuda_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cuda_utils PUBLIC logger CUDA::cudart CUDA::cuda_driver)


add_library(nvtx_utils STATIC nvtx_utils.cc)
set_property(TARGET nvtx_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nvtx_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
if(${CMAKE_VERSION} VERSION_LESS "3.25")
    target_link_libraries(nvtx_utils PUBLIC CUDA::nvToolsExt -ldl)
else()
    target_link_libraries(nvtx_utils PUBLIC CUDA::nvtx3 -ldl)
endif()

add_library(memory_utils STATIC memory_utils.cu)
set_property(TARGET memory_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET memory_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(memory_utils PUBLIC cuda_utils logger)

add_library(anomaly_handler STATIC anomaly_handler.cu)
set_property(TARGET anomaly_handler PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET anomaly_handler PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(anomaly_handler PUBLIC cuda_utils logger)

add_library(parser STATIC parser.cc)
set_property(TARGET parser PROPERTY POSITION_INDEPENDENT_CODE  ON)
