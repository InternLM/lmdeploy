# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    execute_process(COMMAND "pybind11-config" "--cmakedir"
                    RESULT_VARIABLE _COMMAND_SUCCESS
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG)
endif()

set(ModuleName _turboformers)
pybind11_add_module(${ModuleName} bind.cpp)
target_link_libraries(${ModuleName} PRIVATE TransformerTritonBackend 
    LlamaTritonBackend tensor
    custom_ar_comm memory_utils -lcublasLt)

install(TARGETS ${ModuleName}
DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer)