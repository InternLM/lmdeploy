# Copyright (c) OpenMMLab. All rights reserved.

add_library(rms_norm rms_norm.cu)
set_property(TARGET rms_norm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rms_norm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
