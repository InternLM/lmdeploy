# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.18)

project(tritonturbomindbackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_PYTORCH_INCLUDE_PATHS "" CACHE PATH "Paths to Torch includes")
set(TRITON_PYTORCH_LIB_PATHS "" CACHE PATH "Paths to Torch libraries")

set(TRITON_BACKEND_REPO_TAG "r22.12" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "r22.12" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "r22.12" CACHE STRING "Tag for triton-inference-server/common repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(USE_TRITONSERVER_DATATYPE "ON")
message("-- Enable USE_TRITONSERVER_DATATYPE")

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(libtriton_fastertransformer.ldscript libtriton_fastertransformer.ldscript COPYONLY)

add_library(
  triton-turbomind-backend SHARED
  libfastertransformer.cc
)

add_library(
  TritonTurboMindBackend::triton-turbomind-backend ALIAS triton-turbomind-backend
)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA 10.1 REQUIRED)
if (${CUDA_VERSION} GREATER_EQUAL 11.0)
  message(STATUS "Add DCUDA11_MODE")
  add_definitions("-DCUDA11_MODE")
endif()

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

target_compile_definitions(triton-turbomind-backend
  PUBLIC
  USE_TRITONSERVER_DATATYPE
  BUILD_MULTI_GPU)

target_include_directories(
  triton-turbomind-backend
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${TRITON_PYTORCH_INCLUDE_PATHS}
  ${Python3_INCLUDE_DIRS}
  ${repo-ft_SOURCE_DIR}
  ${repo-ft_SOURCE_DIR}/3rdparty/cutlass/include
  ${repo-core_SOURCE_DIR}/include
  )

target_link_directories(
  triton-turbomind-backend
  PRIVATE
  ${CUDA_PATH}/lib64
  )

target_compile_features(triton-turbomind-backend PRIVATE cxx_std_14)

target_compile_options(
  triton-turbomind-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits >#-Werror>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-turbomind-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU

set_target_properties(
  triton-turbomind-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_turbomind
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_fastertransformer.ldscript
    LINK_FLAGS "-Wl,--no-as-needed,--version-script ${CMAKE_CURRENT_BINARY_DIR}/libtriton_fastertransformer.ldscript"
)

# Need to turn off unused-but-set-variable due to Torchvision
# Need to turn off unknown-pragmas due to ATen OpenMP
set_target_properties(
  triton-turbomind-backend
  PROPERTIES COMPILE_FLAGS
    "-Wno-unknown-pragmas -Wno-unused-but-set-variable"
)

set(TRITON_PYTORCH_LDFLAGS "")
FOREACH(p ${TRITON_PYTORCH_LIB_PATHS})
  set(TRITON_PYTORCH_LDFLAGS ${TRITON_PYTORCH_LDFLAGS} "-L${p}")
ENDFOREACH(p)

target_link_libraries(
  triton-turbomind-backend
  PRIVATE
    triton-core-serverapi  # from repo-core
    triton-core-backendapi # from repo-core
    triton-core-serverstub # from repo-core
    triton-backend-utils   # from repo-backend
    transformer-shared     # from repo-ft
    ${TRITON_PYTORCH_LDFLAGS}
    -lcublas
    -lcublasLt
    -lcudart
    -lcurand
)

if (BUILD_MULTI_GPU)
  target_compile_definitions(
    triton-turbomind-backend
    PUBLIC
      BUILD_MULTI_GPU
  )
  target_include_directories(
    triton-turbomind-backend
    PRIVATE
      ${MPI_INCLUDE_PATH}
  )
  target_link_directories(
    triton-turbomind-backend
    PRIVATE
      ${MPI_Libraries}
      /usr/local/mpi/lib
  )
  target_link_libraries(
    triton-turbomind-backend
    PRIVATE
      ${NCCL_LIBRARIES}
      ${MPI_LIBRARIES}
  )
endif()

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-turbomind-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TurboMindBackend)

install(
  TARGETS
    triton-turbomind-backend
  EXPORT
    triton-turbomind-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/turbomind
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/turbomind
)

install(
  EXPORT
    triton-turbomind-backend-targets
  FILE
    TritonTurboMindBackendTargets.cmake
  NAMESPACE
    TritonTurboMindBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/TritonTurboMindBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTurboMindBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTurboMindBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-turbomind-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonTurboMindBackendTargets.cmake
  NAMESPACE TritonTurboMindBackend::
)

export(PACKAGE TritonTurboMindBackend)


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

add_library(TransformerTritonBackend SHARED transformer_triton_backend.cpp)
target_link_libraries(TransformerTritonBackend PRIVATE nccl_utils mpi_utils)

add_subdirectory(llama)
