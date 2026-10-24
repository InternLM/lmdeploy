# Copyright (c) OpenMMLab. All rights reserved.

set(TM_BUILD_VERIFICATION_ATTENTION_SM90 OFF)
if(NOT MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
  set(TM_BUILD_VERIFICATION_ATTENTION_SM90 ON)
endif()

if(TM_BUILD_VERIFICATION_ATTENTION_SM90)
  add_library(verification_attention STATIC dispatch.cu reduce.cu)
  target_compile_definitions(verification_attention PUBLIC
      TM_BUILD_VERIFICATION_ATTENTION_SM90=1)
  set_property(TARGET verification_attention
               PROPERTY CUDA_ARCHITECTURES 90a-real)
  set_property(TARGET verification_attention
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(verification_attention PRIVATE
      -O3
      $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr -Xptxas=-v --threads 8>)
  target_link_libraries(verification_attention PRIVATE nvidia::cutlass::cutlass)
else()
  add_library(verification_attention STATIC stub.cc)
  target_compile_definitions(verification_attention PUBLIC
      TM_BUILD_VERIFICATION_ATTENTION_SM90=0)
endif()

set_property(TARGET verification_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
