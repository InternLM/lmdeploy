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

cmake_minimum_required(VERSION 3.11 FATAL_ERROR) # for PyTorch extensions, version should be greater than 3.13
cmake_policy(SET CMP0074 NEW)
project(TurboMind LANGUAGES CXX CUDA)

if (MSVC)
    # use standard conformant preprocessor
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus")
endif ()

find_package(CUDAToolkit REQUIRED)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(BUILD_MULTI_GPU "Build multi-gpu support" ON)
option(BUILD_PY_FFI "Build python ffi" ON)
option(BUILD_TEST "Build tests" OFF)
option(SPARSITY_SUPPORT "Build project with Ampere sparsity feature support" OFF)
option(BUILD_FAST_MATH "Build in fast math mode" ON)

include(FetchContent)

if (BUILD_TEST)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.0
    GIT_SHALLOW ON
    GIT_PROGRESS            TRUE
    USES_TERMINAL_DOWNLOAD  TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(Catch2)
endif()


FetchContent_Declare(
  repo-cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG                 v3.9.2
  GIT_SHALLOW             ON
  GIT_PROGRESS            TRUE
  USES_TERMINAL_DOWNLOAD  TRUE
  EXCLUDE_FROM_ALL
)

set(CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES ON CACHE BOOL "Enable extended GMMA shapes")
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

FetchContent_MakeAvailable(repo-cutlass)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG                 0.8.0
  GIT_PROGRESS            TRUE
  USES_TERMINAL_DOWNLOAD  TRUE
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/yaml-cpp_cmake_policy.patch
  UPDATE_DISCONNECTED     1
)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static library of yaml-cpp")
FetchContent_MakeAvailable(yaml-cpp)

FetchContent_Declare(
  xgrammar
  GIT_REPOSITORY          https://github.com/mlc-ai/xgrammar.git
  GIT_TAG                 v0.1.25
  GIT_SUBMODULES          "3rdparty/dlpack"
  GIT_PROGRESS            TRUE
  USES_TERMINAL_DOWNLOAD  TRUE
  UPDATE_DISCONNECTED     1
)

FetchContent_GetProperties(xgrammar)
if(NOT xgrammar_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(xgrammar)

  file(WRITE ${xgrammar_SOURCE_DIR}/config.cmake "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n")
  if(NOT MSVC)
    file(APPEND ${xgrammar_SOURCE_DIR}/config.cmake "set(CMAKE_CXX_FLAGS \"-Wno-error\")\n")
  endif()

  # Bring the populated content into the build
  add_subdirectory(${xgrammar_SOURCE_DIR} ${xgrammar_BINARY_DIR})
endif()

# the environment variable
#   ASAN_OPTIONS=protect_shadow_gap=0,intercept_tls_get_addr=0
# must be set at runtime
# https://github.com/google/sanitizers/issues/1322
if (LMDEPLOY_ASAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_link_options(-fsanitize=address)
endif ()

# notice that ubsan has linker issues for ubuntu < 18.04, see
# https://stackoverflow.com/questions/50024731/ld-unrecognized-option-push-state-no-as-needed
if (LMDEPLOY_UBSAN_ENABLE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined>)
    add_link_options(-fsanitize=undefined)
endif ()

if(BUILD_MULTI_GPU)
    execute_process(
      COMMAND python -c "import importlib.util; print(importlib.util.find_spec('nvidia.nccl').submodule_search_locations[0])"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE nccl_path
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(result EQUAL 0 AND NOT nccl_path STREQUAL "")
      set(NCCL_ROOT ${nccl_path})
      message(STATUS "Found NCCL at: ${nccl_path}")

      if(result EQUAL 0 AND NOT nccl_path STREQUAL "")
        file(GLOB nccl_lib_files "${nccl_path}/lib/libnccl.so.*")
        if(nccl_lib_files)
          list(GET nccl_lib_files -1 latest_lib)
          string(REGEX MATCH "\\.([0-9]+)$" version_match ${latest_lib})
          if(version_match)
            set(NCCL_ROOT ${nccl_path})
            set(ENV{NCCL_VERSION} ${CMAKE_MATCH_1})
          endif()
        endif()
      endif()
    endif()

    add_definitions("-DBUILD_MULTI_GPU=1")
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
    find_package(NCCL)
    if (NCCL_FOUND)
        set(USE_NCCL ON)
        add_definitions("-DUSE_NCCL=1")
    endif ()
endif()


set(CXX_STD "17" CACHE STRING "C++ standard")
# enable gold linker for binary and .so
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(CUSPARSELT_PATH "" CACHE STRING "cuSPARSELt path")

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# profiling
option(USE_NVTX "Whether or not to use nvtx" ON)
if(USE_NVTX)
  message(STATUS "NVTX is enabled.")
  add_definitions("-DUSE_NVTX")
endif()

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl") # -Xptxas -v

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
  set(ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  # cmake reports AMD64 on Windows, but we might be building for 32-bit.
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH "x86_64")
  else()
    set(ARCH "x86")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
  set(ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "i386")
  set(ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "i686")
  set(ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ARCH "aarch64")
# Apple A12 Bionic chipset which is added in iPhone XS/XS Max/XR uses arm64e architecture.
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64e")
  set(ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm*")
  set(ARCH "arm")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "mips")
  # Just to avoid the “unknown processor” error.
  set(ARCH "generic")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  set(ARCH "ppc64le")
else()
  message(FATAL_ERROR "Unknown processor:" ${CMAKE_SYSTEM_PROCESSOR})
endif()


if(ARCH STREQUAL "x86_64")
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real)  # V100, 2080
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real) # A100
    endif ()
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.1")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real) # 3090
    endif ()
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.8")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real) # 4090
    endif ()
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "12.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 90a-real) # H100
    endif ()
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "12.8")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real) # 5090
    endif ()
    if (MSVC)
      list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES 80-real 90a-real)
    endif ()
  endif ()
elseif(ARCH STREQUAL "aarch64")
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72-real 87-real)  # Jetson
  endif()
else()
  message(FATAL_ERROR "Unsupported Architecture:" ${ARCH})
endif()

message(STATUS "Building with CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

string(REPLACE "-O2" "" CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-O2" "" CMAKE_CUDA_FLAGS_RELEASE        "${CMAKE_CUDA_FLAGS_RELEASE}")
string(REPLACE "-O2" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-O2" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE}         -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}  -O3")
set(CMAKE_CUDA_FLAGS_RELEASE        "${CMAKE_CUDA_FLAGS_RELEASE}        -O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O3")

if(BUILD_FAST_MATH)
    set(CMAKE_CUDA_FLAGS_RELEASE        "${CMAKE_CUDA_FLAGS_RELEASE}        --use_fast_math")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --use_fast_math")
    message("Release build CUDA flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
  ${CUTLASS_HEADER_DIR}
)
message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if (SPARSITY_SUPPORT)
  list(APPEND COMMON_HEADER_DIRS ${CUSPARSELT_PATH}/include)
  list(APPEND COMMON_LIB_DIRS ${CUSPARSELT_PATH}/lib64)
  add_definitions(-DSPARSITY_ENABLED=1)
endif()


set(PYTHON_PATH "python" CACHE STRING "Python path")

# turn off warnings on windows
if (MSVC)
  foreach(
    flag_var
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_CUDA_FLAGS
    CMAKE_CUDA_FLAGS_DEBUG
    CMAKE_CUDA_FLAGS_RELEASE
    CMAKE_CUDA_FLAGS_MINSIZEREL
    CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "-Wall" " /W0 " ${flag_var} "${${flag_var}}")
  endforeach()
endif()

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(src)

# if(BUILD_TEST)
#     add_subdirectory(tests/csrc)
# endif()

# install python api
if (BUILD_PY_FFI)
  if (CALL_FROM_SETUP_PY)
    install(TARGETS _turbomind DESTINATION ${CMAKE_INSTALL_PREFIX})
    install(TARGETS _xgrammar DESTINATION ${CMAKE_INSTALL_PREFIX})
  else()
    install(TARGETS _turbomind DESTINATION ${CMAKE_SOURCE_DIR}/lmdeploy/lib)
    install(TARGETS _xgrammar DESTINATION ${CMAKE_SOURCE_DIR}/lmdeploy/lib)
  endif()
endif ()
