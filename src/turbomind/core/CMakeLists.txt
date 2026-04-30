# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

include(FetchContent)

FetchContent_Declare(
  concurrentqueue
  GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
  GIT_TAG        v1.0.4
  GIT_SHALLOW    ON
  GIT_PROGRESS   TRUE
  USES_TERMINAL_DOWNLOAD TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(concurrentqueue)

add_library(core STATIC
        check.cc
        allocator.cc
        stream.cc
        context.cc
        scope.cc
        buffer.cc
        layout.cc
        tensor.cc
        tensor.cu
        module.cc
        copy.cc
        logger.cc)

target_link_libraries(core PUBLIC cuda_utils CUDA::cudart CUDA::cuda_driver fmt::fmt concurrentqueue)

set_property(TARGET core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

if (BUILD_TEST)
    add_executable(test_core test_core.cc)
    target_link_libraries(test_core PRIVATE core Catch2::Catch2WithMain)

    add_executable(test_scope test_scope.cc)
    target_link_libraries(test_scope PRIVATE core Catch2::Catch2WithMain)

    add_executable(test_logger test_logger.cc)
    target_link_libraries(test_logger PRIVATE core Catch2::Catch2WithMain)
endif ()
