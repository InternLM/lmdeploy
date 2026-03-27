# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

set(DEEP_EP_SOURCE_FILES
    deep_ep/deep_ep.cpp
    deep_ep/gin_backend.cu
    deep_ep/kernels/runtime.cu
    deep_ep/kernels/layout.cu
    deep_ep/kernels/intranode.cu
    deep_ep/kernels/internode_ll.cu
)

add_library(deepep STATIC ${DEEP_EP_SOURCE_FILES})
target_link_libraries(deepep PRIVATE ${NCCL_LIBRARIES} CUDA::cudart)
set_property(TARGET deepep PROPERTY CUDA_ARCHITECTURES 90)
target_include_directories(deepep PRIVATE ${NCCL_INCLUDE_DIRS})
set_property(TARGET deepep PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET deepep PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(nccl_comm STATIC nccl.cu nccl_ep.cu)
target_link_libraries(nccl_comm PRIVATE rms_norm core ${NCCL_LIBRARIES} logger deepep)
target_include_directories(nccl_comm PRIVATE ${NCCL_INCLUDE_DIRS})

set_property(TARGET nccl_comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nccl_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
