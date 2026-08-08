# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.25)

include(FetchContent)

set(_deepep_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/deepep-cpp-wrapper.patch")
set(_deepep_patch_script "${CMAKE_CURRENT_SOURCE_DIR}/patches/apply_patch.cmake")

FetchContent_Declare(
    deepep
    GIT_REPOSITORY https://github.com/deepseek-ai/DeepEP.git
    GIT_TAG dd758caf451848bd150e1046af3d0a73e5fff38d
    GIT_SUBMODULES ""
    GIT_PROGRESS TRUE
    PATCH_COMMAND
        ${CMAKE_COMMAND}
        -DSOURCE_DIR=<SOURCE_DIR>
        -DPATCH_FILE=${_deepep_patch}
        -P ${_deepep_patch_script})

FetchContent_GetProperties(deepep)
if (NOT deepep_POPULATED)
    FetchContent_Populate(deepep)
endif ()

if (NOT EXISTS "${NCCL_INCLUDE_DIRS}/nccl_device.h"
    OR NOT IS_DIRECTORY "${NCCL_INCLUDE_DIRS}/nccl_device")
    message(FATAL_ERROR
        "TM_USE_DEEPEP requires NCCL device API headers under ${NCCL_INCLUDE_DIRS}")
endif ()

file(STRINGS "${NCCL_INCLUDE_DIRS}/nccl.h" _nccl_major_line
     REGEX "^#define NCCL_MAJOR +[0-9]+")
file(STRINGS "${NCCL_INCLUDE_DIRS}/nccl.h" _nccl_minor_line
     REGEX "^#define NCCL_MINOR +[0-9]+")
file(STRINGS "${NCCL_INCLUDE_DIRS}/nccl.h" _nccl_patch_line
     REGEX "^#define NCCL_PATCH +[0-9]+")
string(REGEX REPLACE ".*NCCL_MAJOR +([0-9]+).*" "\\1" _nccl_major "${_nccl_major_line}")
string(REGEX REPLACE ".*NCCL_MINOR +([0-9]+).*" "\\1" _nccl_minor "${_nccl_minor_line}")
string(REGEX REPLACE ".*NCCL_PATCH +([0-9]+).*" "\\1" _nccl_patch "${_nccl_patch_line}")
if (_nccl_major_line STREQUAL ""
    OR _nccl_minor_line STREQUAL ""
    OR _nccl_patch_line STREQUAL "")
    message(FATAL_ERROR "Cannot determine NCCL version from ${NCCL_INCLUDE_DIRS}/nccl.h")
endif ()
set(_nccl_version "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
if (_nccl_version VERSION_LESS "2.30.4")
    message(FATAL_ERROR "TM_USE_DEEPEP requires NCCL >= 2.30.4, found ${_nccl_version}")
endif ()

get_filename_component(_nccl_build_root "${NCCL_INCLUDE_DIRS}" DIRECTORY)

find_package(Threads REQUIRED)

add_library(turbomind_nccl_stub SHARED nccl_stub.cc)
set_target_properties(turbomind_nccl_stub PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME turbomind_nccl_stub
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(turbomind_nccl_stub PRIVATE
    ${NCCL_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})

add_library(token_dispatcher STATIC
    token_dispatcher.cc
    moe_a2a_utils.cu
)
set_target_properties(token_dispatcher PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
target_compile_options(token_dispatcher PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2417>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
target_compile_definitions(token_dispatcher PRIVATE
    EP_NUM_TOPK_IDX_BITS=32
    TM_DEEPEP_BUILD_ROOT="${deepep_SOURCE_DIR}/deep_ep")
target_include_directories(token_dispatcher
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE
        ${deepep_SOURCE_DIR}
        ${deepep_SOURCE_DIR}/deep_ep/include
        ${NCCL_INCLUDE_DIRS})
target_link_libraries(token_dispatcher PRIVATE
    core
    host_comm
    fmt::fmt
    CUDA::cudart
    CUDA::cuda_driver
    # Keep the real NCCL before the weak compatibility fallbacks. The anchor in
    # token_dispatcher.cc prevents --as-needed from dropping the stub.
    ${NCCL_LIBRARIES}
    turbomind_nccl_stub
    Threads::Threads)
set_property(TARGET token_dispatcher PROPERTY POSITION_INDEPENDENT_CODE ON)

install(
    DIRECTORY "${deepep_SOURCE_DIR}/deep_ep/include/deep_ep"
    DESTINATION include)
