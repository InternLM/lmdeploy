# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(memory STATIC object.cc stats.cc)
target_link_libraries(memory PRIVATE core)
set_property(TARGET memory PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TEST)
    add_executable(test_memory test_memory.cc)
    target_link_libraries(test_memory PRIVATE memory core Catch2::Catch2WithMain)
endif ()
