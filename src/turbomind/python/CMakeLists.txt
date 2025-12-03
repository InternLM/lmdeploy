# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)
project(_turbomind LANGUAGES CXX CUDA)

option(USE_STABLE_ABI "Build with python limited api of abi3" OFF)

if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  get_filename_component(_py_abs "${PYTHON_EXECUTABLE}" ABSOLUTE)
  set(Python_EXECUTABLE "${_py_abs}" CACHE FILEPATH "")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

if (USE_STABLE_ABI)
  list(APPEND DEV_MODULE Development.SABIModule)
  set(MODULE_OPTION STABLE_ABI LTO)
else()
  set(MODULE_OPTION LTO)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_turbomind ${MODULE_OPTION} tm_nanobind.cpp)
target_link_libraries(_turbomind PRIVATE LlamaTritonBackend)
target_compile_features(_turbomind PRIVATE cxx_std_14)

nanobind_add_module(_xgrammar ${MODULE_OPTION} xgr_nanobind.cpp)
target_link_libraries(_xgrammar PRIVATE core xgrammar)
target_compile_features(_xgrammar PRIVATE cxx_std_14)

if (CALL_FROM_SETUP_PY)
  string(REPLACE "." ";" _ver ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET _ver 0 CUDA_MAJOR)

  if(CUDA_MAJOR GREATER_EQUAL "13")
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib/"
    )
  else()
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cuda_runtime/lib/"
        "\$ORIGIN/../../nvidia/cublas/lib/"
        "\$ORIGIN/../../nvidia/curand/lib/"
    )
  endif()
  set_target_properties(${PROJECT_NAME} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "${_INSTALL_CUDA_RPATH}"
  )
endif ()
