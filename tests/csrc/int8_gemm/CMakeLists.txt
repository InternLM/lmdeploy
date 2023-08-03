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

set(int8_test_files
    int8_gemm_test.cu
)

add_definitions(-DTORCH_CUDA=1)

set(EXE_NAME "int8_gemm_test")
add_executable(${EXE_NAME} ${int8_test_files})
set_target_properties(${EXE_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${EXE_NAME} PUBLIC "${TORCH_LIBRARIES}" int8_gemm tensor logger)
