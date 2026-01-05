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

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(comm)
add_subdirectory(generation)
add_subdirectory(models)
add_subdirectory(engine)

if(BUILD_PY_FFI)
    add_subdirectory(python)
endif()

add_library(turbomind STATIC turbomind.cc)
set_property(TARGET turbomind PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(turbomind PUBLIC
        engine
        models
        device_comm
        host_comm
        core
        memory_utils
        nvtx_utils
        CUDA::cublasLt
        CUDA::cudart
        yaml-cpp::yaml-cpp)
