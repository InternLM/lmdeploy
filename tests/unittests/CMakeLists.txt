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
add_definitions(-DTORCH_CUDA=1)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unittest
    test_attention_kernels.cu
    test_logprob_kernels.cu
    test_penalty_kernels.cu
    test_sampling_kernels.cu
    test_sampling_layer.cu
    test_tensor.cu)

# automatic discovery of unit tests
target_link_libraries(unittest PUBLIC "${TORCH_LIBRARIES}" gtest_main)
target_compile_features(unittest PRIVATE cxx_std_14)

# Sorted by alphabetical order of test name.
target_link_libraries(  # Libs for test_attention_kernels
  unittest PUBLIC
    -lcudart -lcurand
    gen_relative_pos_bias gpt_kernels gtest memory_utils tensor unfused_attention_kernels cuda_utils logger)
target_link_libraries(  # Libs for test_logprob_kernels
  unittest PUBLIC
    -lcudart
    logprob_kernels memory_utils cuda_utils logger)
target_link_libraries(  # Libs for test_penalty_kernels
  unittest PUBLIC
    -lcublas -lcublasLt -lcudart
    sampling_penalty_kernels beam_search_penalty_kernels memory_utils cuda_utils logger)
target_link_libraries(  # Libs for test_sampling_kernel
  unittest PUBLIC
    -lcudart
    sampling_topk_kernels sampling_topp_kernels memory_utils tensor cuda_utils logger)
target_link_libraries(  # Libs for test_sampling_layer
  unittest PUBLIC
    -lcublas -lcublasLt -lcudart
    cublasMMWrapper memory_utils
    DynamicDecodeLayer TopKSamplingLayer TopPSamplingLayer tensor cuda_utils logger)
target_link_libraries(  # Libs for test_tensor
  unittest PUBLIC tensor cuda_utils logger)

remove_definitions(-DTORCH_CUDA=1)
add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PUBLIC -lcublas -lcudart -lcurand gemm cublasMMWrapper tensor cuda_utils logger)

add_executable(test_gpt_kernels test_gpt_kernels.cu)
target_link_libraries(test_gpt_kernels PUBLIC
                      gpt_kernels memory_utils tensor cuda_utils logger)

add_executable(test_activation test_activation.cu)
target_link_libraries(test_activation PUBLIC
                    -lcublas -lcublasLt -lcudart
                    activation_kernels memory_utils cuda_utils logger)

add_executable(test_context_attention_layer test_context_attention_layer.cu)
target_link_libraries(test_context_attention_layer PUBLIC
                      Llama -lcublas -lcublasLt -lcudart
                      unfused_attention_kernels
                      memory_utils tensor cublasMMWrapper cuda_utils logger)
