# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(nccl_comm STATIC nccl.cu)
target_link_libraries(nccl_comm PRIVATE rms_norm core ${NCCL_LIBRARIES} logger)
target_include_directories(nccl_comm PRIVATE ${NCCL_INCLUDE_DIRS})

set_property(TARGET nccl_comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nccl_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
