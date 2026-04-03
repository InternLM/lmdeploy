# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(nccl_comm STATIC nccl.cu)
target_link_libraries(nccl_comm PRIVATE rms_norm core ${NCCL_LIBRARIES} logger)
target_include_directories(nccl_comm PRIVATE ${NCCL_INCLUDE_DIRS})

set_property(TARGET nccl_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET nccl_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

file(READ "${NCCL_INCLUDE_DIRS}/nccl.h" NCCL_HEADER_CONTENTS)
string(REGEX MATCH "#define NCCL_MAJOR[ \t]+([0-9]+)" _ ${NCCL_HEADER_CONTENTS})
set(NCCL_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define NCCL_MINOR[ \t]+([0-9]+)" _ ${NCCL_HEADER_CONTENTS})
set(NCCL_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define NCCL_PATCH[ \t]+([0-9]+)" _ ${NCCL_HEADER_CONTENTS})
set(NCCL_PATCH ${CMAKE_MATCH_1})
set(NCCL_VERSION_STRING "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
message(STATUS "Detected NCCL version: ${NCCL_VERSION_STRING}")

if(NOT NCCL_VERSION_STRING VERSION_LESS "2.29.7")
  set(DEEP_EP_SOURCE_FILES
      deep_ep/deep_ep.cpp
      deep_ep/gin_backend.cu
      deep_ep/kernels/runtime.cu
      deep_ep/kernels/layout.cu
      deep_ep/kernels/intranode.cu
      deep_ep/kernels/internode.cu
      deep_ep/kernels/internode_ll.cu
      nccl_ep.cu)

  add_library(deepep STATIC ${DEEP_EP_SOURCE_FILES})
  target_link_libraries(deepep PRIVATE ${NCCL_LIBRARIES} CUDA::cudart)
  set_property(TARGET deepep PROPERTY CUDA_ARCHITECTURES 90)
  target_include_directories(deepep PRIVATE ${NCCL_INCLUDE_DIRS})
  set_property(TARGET deepep PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET deepep PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_link_libraries(nccl_comm PRIVATE deepep)
else()
  message(STATUS "Skip deepep build because NCCL ${NCCL_VERSION_STRING} < 2.29.7")
endif()
