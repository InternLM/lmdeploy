# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

find_package(CUDAToolkit REQUIRED)

add_library(Medusa STATIC
        medusa_weight.cc
        res_block.cc
        medusa_head.cc)

set_property(TARGET Medusa PROPERTY POSITION_INDEPENDENT_CODE  ON)
