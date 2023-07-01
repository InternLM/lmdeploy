# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

# Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/triton_backend/multi_gpu_gpt/CMakeLists.txt

cmake_minimum_required(VERSION 3.8)

set(llama_triton_backend_files
    LlamaTritonModel.cc
    LlamaTritonModelInstance.cc
)

add_library(LlamaTritonBackend STATIC ${llama_triton_backend_files})
set_property(TARGET LlamaTritonBackend PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(LlamaTritonBackend PRIVATE TransformerTritonBackend Llama tensor memory_utils -lcublasLt)
target_compile_features(LlamaTritonBackend PRIVATE cxx_std_14)
