# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)
project(_turbomind)

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    execute_process(COMMAND "pybind11-config" "--cmakedir"
                    RESULT_VARIABLE _COMMAND_SUCCESS
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG)
endif()

pybind11_add_module(${PROJECT_NAME} bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE TransformerTritonBackend 
    LlamaTritonBackend custom_ar_comm memory_utils)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lmdeploy/lib)