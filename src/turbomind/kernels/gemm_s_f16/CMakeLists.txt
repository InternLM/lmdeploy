# Copyright (c) OpenMMLab. All rights reserved.

add_library(gemm_s4_f16 STATIC gemm_s4_f16.cu format.cu)
target_compile_options(gemm_s4_f16 PRIVATE 
  --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
set_property(TARGET gemm_s4_f16 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemm_s4_f16 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# add_executable(test_gemm_s4_f16 test_utils.cu test_gemm_s4_f16.cu)
# target_compile_options(test_gemm_s4_f16 PRIVATE 
#   --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
# target_link_libraries(test_gemm_s4_f16 PRIVATE gemm_s4_f16 cublas)