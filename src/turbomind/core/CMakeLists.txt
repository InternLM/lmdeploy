# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(core STATIC
        check.cc
        allocator.cc
        stream.cc
        context.cc
        buffer.cc
        layout.cc
        tensor.cc
        tensor.cu
        module.cc
        copy.cc)

target_link_libraries(core PUBLIC cuda_utils logger CUDA::cudart CUDA::cuda_driver)

set_property(TARGET core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

if (BUILD_TEST)
    add_executable(test_core test_core.cc)
    target_link_libraries(test_core PRIVATE core logger Catch2::Catch2WithMain)
endif ()
