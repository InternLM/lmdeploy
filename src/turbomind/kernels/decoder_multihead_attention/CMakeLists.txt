# Copyright (c) OpenMMLab. All rights reserved.

add_library(decoder_multihead_attention STATIC decoder_multihead_attention.cu kv_cache.cu)
# target_compile_options(decoder_multihead_attention PRIVATE
#   --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr --keep)
set_property(TARGET decoder_multihead_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET decoder_multihead_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(decoder_multihead_attention PRIVATE nvidia::cutlass::cutlass)

add_executable(test_decoder_multihead_attention test_utils.cu test_decoder_multihead_attention.cu)
# target_compile_options(test_decoder_multihead_attention PRIVATE
#   --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
target_link_libraries(test_decoder_multihead_attention PRIVATE
    decoder_multihead_attention
    decoder_masked_multihead_attention
    cublas)
