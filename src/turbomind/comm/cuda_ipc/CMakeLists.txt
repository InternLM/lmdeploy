# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(cuda_ipc_comm STATIC
        cuda_ipc_comm.cu
        allreduce.cu
        allgather.cu
        fused_allreduce.cu
        fused_allreduce_ex.cu
        broadcast.cu)

target_link_libraries(cuda_ipc_comm PRIVATE
        rms_norm
        host_comm
        core
        cuda_utils
        CUDA::cuda_driver
        logger)

set_property(TARGET cuda_ipc_comm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_ipc_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
