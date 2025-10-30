# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

add_library(engine STATIC gateway.cc request_queue.cc model_request.cc)
target_link_libraries(engine PRIVATE core xgrammar)
set_property(TARGET engine PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET engine PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
