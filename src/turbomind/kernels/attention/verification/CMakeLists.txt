# Copyright (c) OpenMMLab. All rights reserved.
#
# Two per-arch verification libraries + a host-only dispatcher. This keeps the
# SM90 (90a) device code and the new SM75 (75) device code fully separate so
# neither affects the other; the dispatcher selects by device arch at runtime.

set(TM_BUILD_VERIFICATION_ATTENTION_SM90 OFF)
if(NOT MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
  set(TM_BUILD_VERIFICATION_ATTENTION_SM90 ON)
endif()

set(TM_BUILD_VERIFICATION_ATTENTION_SM75 OFF)
if(NOT MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
  set(TM_BUILD_VERIFICATION_ATTENTION_SM75 ON)
endif()

set(TM_VERIFICATION_ARCH_LIBS)

if(TM_BUILD_VERIFICATION_ATTENTION_SM90)
  add_library(verification_attention_sm90 STATIC dispatch.cu)
  target_compile_definitions(verification_attention_sm90 PUBLIC
      TM_BUILD_VERIFICATION_ATTENTION_SM90=1)
  set_property(TARGET verification_attention_sm90
               PROPERTY CUDA_ARCHITECTURES 90a-real)
  set_property(TARGET verification_attention_sm90
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET verification_attention_sm90
               PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(verification_attention_sm90 PRIVATE
      -O3
      $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr -Xptxas=-v --threads 8>)
  target_link_libraries(verification_attention_sm90 PRIVATE nvidia::cutlass::cutlass)
  list(APPEND TM_VERIFICATION_ARCH_LIBS verification_attention_sm90)
endif()

if(TM_BUILD_VERIFICATION_ATTENTION_SM75)
  add_library(verification_attention_sm75 STATIC kernel_sm75.cu)
  set_property(TARGET verification_attention_sm75
               PROPERTY CUDA_ARCHITECTURES 75-real)
  set_property(TARGET verification_attention_sm75
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET verification_attention_sm75
               PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(verification_attention_sm75 PRIVATE
      -O3
      $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr -Xptxas=-v --threads 8>)
  target_link_libraries(verification_attention_sm75 PRIVATE nvidia::cutlass::cutlass)
  list(APPEND TM_VERIFICATION_ARCH_LIBS verification_attention_sm75)
endif()

if(TM_VERIFICATION_ARCH_LIBS)
  add_library(verification_attention STATIC dispatch_arch.cc)
  target_compile_definitions(verification_attention PUBLIC
      TM_BUILD_VERIFICATION_ATTENTION_SM90=$<BOOL:${TM_BUILD_VERIFICATION_ATTENTION_SM90}>
      TM_BUILD_VERIFICATION_ATTENTION_SM75=$<BOOL:${TM_BUILD_VERIFICATION_ATTENTION_SM75}>)
  target_link_libraries(verification_attention PUBLIC ${TM_VERIFICATION_ARCH_LIBS})
  target_link_libraries(verification_attention PRIVATE nvidia::cutlass::cutlass)
else()
  add_library(verification_attention STATIC stub.cc)
  target_compile_definitions(verification_attention PUBLIC
      TM_BUILD_VERIFICATION_ATTENTION_SM90=0
      TM_BUILD_VERIFICATION_ATTENTION_SM75=0)
endif()

set_property(TARGET verification_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
