# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

add_library(comm STATIC comm.cc)
target_link_libraries(comm PRIVATE logger)

set_property(TARGET comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

if (BUILD_MULTI_GPU)
    add_subdirectory(nccl)
    add_subdirectory(custom)
    target_link_libraries(comm PRIVATE nccl_comm custom_comm)

    add_executable(test_comm test_comm.cu)
    target_link_libraries(test_comm PRIVATE comm pthread nvtx_utils)
    target_compile_options(test_comm PRIVATE -O3 -march=native -mtune=native)
endif ()

