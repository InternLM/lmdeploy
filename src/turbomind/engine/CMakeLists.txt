# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)

add_library(engine STATIC
    gateway.cc
    request.cc
    request_queue.cc
    model_request.cc
    model_executor.cc
    engine.cc
    )
target_link_libraries(engine PRIVATE xgrammar core)
set_property(TARGET engine PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET engine PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
