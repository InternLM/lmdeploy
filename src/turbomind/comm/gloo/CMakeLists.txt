# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)

include(FetchContent)
FetchContent_Declare(
  gloo
  GIT_REPOSITORY https://github.com/facebookincubator/gloo.git
  GIT_TAG cbe963b5a43cd75e6eca4f74d2bb38ec8dcfdbc8
)

# some settings of gloo,
set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)
set(__USE_NCCL ${USE_NCCL})
set(__BUILD_TEST ${BUILD_TEST})
set(USE_NCCL OFF)
set(BUILD_TEST OFF)
set(USE_REDIS ON) # TODO remove, use tcp_store instead
FetchContent_MakeAvailable(gloo)

# gloo build doesn't add include directories as a target property...
target_include_directories(gloo PUBLIC
    $<BUILD_INTERFACE:${gloo_SOURCE_DIR}>
    $<BUILD_INTERFACE:${gloo_BINARY_DIR}> # config.h generated at cmake config time
)
set(USE_NCCL ${__USE_NCCL})
set(BUILD_TEST ${__BUILD_TEST})

add_library(gloo_comm STATIC
    gloo_comm.cc
    # tcp_store.cc
)
set_property(TARGET gloo_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gloo_comm PRIVATE gloo logger)

# TODO remove, use tcp_store instead
include_directories(SYSTEM ${HIREDIS_INCLUDE_DIRS})
target_link_libraries(gloo_comm PRIVATE ${HIREDIS_LIBRARIES})
