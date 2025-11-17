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
target_link_libraries(${PROJECT_NAME} PRIVATE LlamaTritonBackend)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

pybind11_add_module(_xgrammar xgrammar_bind.cpp)
target_link_libraries(_xgrammar PRIVATE core xgrammar)
target_compile_features(_xgrammar PRIVATE cxx_std_14)

if (CALL_FROM_SETUP_PY)
  set(_INSTALL_CUDA_RPATH
      "\$ORIGIN"
      "\$ORIGIN/../../nvidia/nccl/lib/"
      "\$ORIGIN/../../nvidia/cuda_runtime/lib/"
      "\$ORIGIN/../../nvidia/cublas/lib/"
      "\$ORIGIN/../../nvidia/curand/lib/"
  )
  set_target_properties(${PROJECT_NAME} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "${_INSTALL_CUDA_RPATH}"
  )
endif ()
