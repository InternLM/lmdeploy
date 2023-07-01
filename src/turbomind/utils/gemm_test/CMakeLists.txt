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

set(gemm_func_files
  gemm_func.cc
)

set(encoder_gemm_func_files
  encoder_gemm_func.cc
)

set(encoder_igemm_func_files
  encoder_igemm_func.cc
)

set(decoding_gemm_func_files
  decoding_gemm_func.cc
)

set(gpt_gemm_func_files
  gpt_gemm_func.cc
)

set(xlnet_gemm_func_files
  xlnet_gemm_func.cc
)

set(t5_gemm_func_files
  t5_gemm_func.cc
)

set(swin_igemm_func_files
  swin_igemm_func.cc
)

set(swin_gemm_func_files
  swin_gemm_func.cc
)

add_library(gemm_func STATIC ${gemm_func_files})
target_link_libraries(gemm_func PUBLIC -lcublas -lcublasLt -lcudart cuda_utils logger)
set_property(TARGET gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(encoder_gemm_func STATIC ${encoder_gemm_func_files})
target_link_libraries(encoder_gemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func cuda_utils logger)
if (SPARSITY_SUPPORT)
target_link_libraries(encoder_gemm_func PUBLIC -lcusparse -lcusparseLt)
endif()
set_property(TARGET encoder_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET encoder_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(encoder_igemm_func STATIC ${encoder_igemm_func_files})
target_link_libraries(encoder_igemm_func PUBLIC -lcublas -lcublasLt -lcudart cuda_utils logger)
if (SPARSITY_SUPPORT)
target_link_libraries(encoder_igemm_func PUBLIC -lcusparse -lcusparseLt)
endif()
set_property(TARGET encoder_igemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET encoder_igemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(decoding_gemm_func STATIC ${decoding_gemm_func_files})
target_link_libraries(decoding_gemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func cuda_utils logger)
set_property(TARGET decoding_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET decoding_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(gpt_gemm_func STATIC ${gpt_gemm_func_files})
target_link_libraries(gpt_gemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func cuda_utils logger)
if (SPARSITY_SUPPORT)
  target_link_libraries(gpt_gemm_func PUBLIC -lcusparse -lcusparseLt)
endif()
set_property(TARGET gpt_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gpt_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(xlnet_gemm_func STATIC ${xlnet_gemm_func_files})
target_link_libraries(xlnet_gemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func cuda_utils logger)
set_property(TARGET xlnet_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET xlnet_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(t5_gemm_func STATIC ${t5_gemm_func_files})
target_link_libraries(t5_gemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func cuda_utils logger)
if (SPARSITY_SUPPORT)
  target_link_libraries(t5_gemm_func PUBLIC -lcusparse -lcusparseLt)
endif()
set_property(TARGET t5_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET t5_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(swin_igemm_func STATIC ${swin_igemm_func_files})
target_link_libraries(swin_igemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func encoder_igemm_func cuda_utils logger)
set_property(TARGET swin_igemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET swin_igemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(swin_gemm_func STATIC ${swin_gemm_func_files})
target_link_libraries(swin_gemm_func PUBLIC -lcublas -lcublasLt -lcudart gemm_func cuda_utils logger)
set_property(TARGET swin_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET swin_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
