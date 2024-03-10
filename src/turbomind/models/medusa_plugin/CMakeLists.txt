# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

find_package(CUDAToolkit REQUIRED)

add_library(Medusa STATIC
        medusa_weight.cc
        res_block.cc
        medusa_head.cc)

set_property(TARGET Medusa PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(medusa_head_example tests/medusa_head_example.cc)
target_link_libraries(medusa_head_example PUBLIC Llama Medusa -lpthread)
install(TARGETS medusa_head_example DESTINATION ${CMAKE_SOURCE_DIR}/lmdeploy/bin)
