# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)

include(FetchContent)
FetchContent_Declare(
  gloo
  GIT_REPOSITORY https://github.com/pytorch/gloo.git
  GIT_TAG cbe963b5a43cd75e6eca4f74d2bb38ec8dcfdbc8
)

# some settings of gloo,
set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)
set(USE_NCCL OFF)
set(BUILD_TEST OFF)
# set(USE_REDIS ON) # TODO remove, use tcp_store instead
FetchContent_MakeAvailable(gloo)

# gloo build doesn't add include directories as a target property...
target_include_directories(gloo PUBLIC
    $<BUILD_INTERFACE:${gloo_SOURCE_DIR}>
    $<BUILD_INTERFACE:${gloo_BINARY_DIR}> # config.h generated at cmake config time
)

add_library(gloo_comm STATIC
    gloo_comm.cc
    tcp_store.cc
)
set_property(TARGET gloo_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gloo_comm PUBLIC gloo logger)
