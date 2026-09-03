# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.25)
project(_turbomind LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    execute_process(COMMAND "pybind11-config" "--cmakedir"
                    RESULT_VARIABLE _COMMAND_SUCCESS
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG)
endif()

pybind11_add_module(${PROJECT_NAME}
    bind.cpp
    linear_bind.cpp
    xgrammar_bind.cpp
    ../kernels/linear_attn/python_bind.cpp
    ../kernels/gemm/moe_gate_python_bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE turbomind xgrammar)

string(REPLACE "." ";" _cuda_version ${CMAKE_CUDA_COMPILER_VERSION})
list(GET _cuda_version 0 CUDA_MAJOR)

if(CUDA_MAJOR GREATER_EQUAL 13)
  set(_INSTALL_CUDA_RPATH
      "\$ORIGIN/lib"
      "\$ORIGIN/../../nvidia/nccl/lib"
      "\$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib")
else()
  set(_INSTALL_CUDA_RPATH
      "\$ORIGIN/lib"
      "\$ORIGIN/../../nvidia/nccl/lib"
      "\$ORIGIN/../../nvidia/cuda_runtime/lib"
      "\$ORIGIN/../../nvidia/cublas/lib"
      "\$ORIGIN/../../nvidia/curand/lib")
endif()

set_target_properties(_turbomind PROPERTIES
    BUILD_RPATH "\$ORIGIN/lib"
    INSTALL_RPATH "${_INSTALL_CUDA_RPATH}")
