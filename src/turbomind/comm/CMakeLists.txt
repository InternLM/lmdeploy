# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

find_package(Threads)

add_library(host_comm STATIC host_comm.cc thread_comm.cc)
target_link_libraries(host_comm PRIVATE core logger Threads::Threads)
set_property(TARGET host_comm PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(device_comm STATIC device_comm.cc)
target_link_libraries(device_comm PRIVATE core logger)
set_property(TARGET device_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET device_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if (BUILD_MULTI_GPU)
    add_subdirectory(cuda_ipc)
    target_link_libraries(device_comm INTERFACE cuda_ipc_comm)

    if (USE_NCCL)
        add_subdirectory(nccl)
        target_link_libraries(device_comm INTERFACE nccl_comm)
    endif ()

    if (BUILD_TEST)
        add_executable(test_comm test_comm.cu)
        target_link_libraries(test_comm PRIVATE device_comm host_comm core pthread nvtx_utils)
        target_compile_options(test_comm PRIVATE -march=native -mtune=native)

        add_executable(test_host_comm test_host_comm.cc)
        target_link_libraries(test_host_comm PRIVATE host_comm core Threads::Threads)
    endif ()
endif ()
