# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)

include(FetchContent)
FetchContent_Declare(
  gloo
  GIT_REPOSITORY https://github.com/pytorch/gloo.git
  GIT_TAG c7b7b022c124d9643957d9bd55f57ac59fce8fa2 # pytorch-v2.8.0-rc4
)

# some settings of gloo,
set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)
set(USE_NCCL OFF)
set(BUILD_TEST OFF)
set(USE_IBVERBS OFF)
FetchContent_MakeAvailable(gloo)

# gloo build doesn't add include directories as a target property...
target_include_directories(gloo PUBLIC
    $<BUILD_INTERFACE:${gloo_SOURCE_DIR}>
    $<BUILD_INTERFACE:${gloo_BINARY_DIR}> # config.h generated at cmake config time
)

target_compile_options(gloo PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W0>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-w>
)

add_library(gloo_comm STATIC
    gloo_comm.cc
    hybrid_comm.cc
    tcp_store.cc
)
set_property(TARGET gloo_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gloo_comm PUBLIC gloo host_comm logger xgrammar)

add_executable(test_ipc_comm test_ipc_comm.cc)
target_link_libraries(test_ipc_comm PRIVATE gloo_comm Threads::Threads)
