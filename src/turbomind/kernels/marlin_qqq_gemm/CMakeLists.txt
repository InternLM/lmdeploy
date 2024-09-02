# Copyright (c) OpenMMLab. All rights reserved.
add_library(marlin_qqq_gemm STATIC marlin_qqq_gemm.cc marlin_qqq_gemm_kernel.cu)
set_property(TARGET marlin_qqq_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET marlin_qqq_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
