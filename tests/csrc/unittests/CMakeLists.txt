# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

# GoogleTest Preparation - Code block copied from
#   https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

find_package(CUDAToolkit REQUIRED)

if (NOT MSVC)
  add_definitions(-DTORCH_CUDA=1)
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unittest
    test_logprob_kernels.cu
    test_penalty_kernels.cu
    test_sampling_kernels.cu
    test_sampling_layer.cu
    test_tensor.cu
)

# automatic discovery of unit tests
target_link_libraries(unittest PUBLIC "${TORCH_LIBRARIES}" gtest_main)
target_compile_features(unittest PRIVATE cxx_std_14)

# Sorted by alphabetical order of test name.
target_link_libraries(  # Libs for test_attention_kernels
  unittest PUBLIC
    CUDA::cudart CUDA::curand
    gpt_kernels gtest memory_utils tensor unfused_attention_kernels cuda_utils logger)
target_link_libraries(  # Libs for test_logprob_kernels
  unittest PUBLIC
    CUDA::cudart
    logprob_kernels memory_utils cuda_utils logger)
target_link_libraries(  # Libs for test_penalty_kernels
  unittest PUBLIC
    CUDA::cublas CUDA::cublasLt CUDA::cudart
    sampling_penalty_kernels memory_utils cuda_utils logger)
target_link_libraries(  # Libs for test_sampling_kernel
  unittest PUBLIC
    CUDA::cudart
    sampling_topk_kernels sampling_topp_kernels memory_utils tensor cuda_utils logger)
target_link_libraries(  # Libs for test_sampling_layer
  unittest PUBLIC
    CUDA::cublas CUDA::cublasLt CUDA::cudart
    cublasMMWrapper memory_utils
    DynamicDecodeLayer tensor cuda_utils logger
)
target_link_libraries(  # Libs for test_tensor
  unittest PUBLIC tensor cuda_utils logger)

remove_definitions(-DTORCH_CUDA=1)
add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PUBLIC CUDA::cublas CUDA::cudart CUDA::curand gemm cublasMMWrapper tensor cuda_utils logger)
