# Copyright (c) OpenMMLab. All rights reserved.

add_library(attention STATIC 
            attention.cu 
            decoding.cu 
            kv_cache_utils.cu
            utils.cc
            attention_128_f16_sm80.cu
            attention_128_f16_sm70.cu
            decoding_128_f16_sm70.cu
            decoding_128_f16_sm80.cu)
# target_compile_options(attention PRIVATE
#   --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr --keep)
set_property(TARGET attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(attention PRIVATE -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -Xptxas=-v --expt-relaxed-constexpr --generate-line-info>)

add_executable(test_attention test_utils.cu test_attention.cu reference.cu)
target_compile_options(test_attention PRIVATE
  --generate-line-info -O3 -use_fast_math --expt-relaxed-constexpr)
target_link_libraries(test_attention PRIVATE
    attention
    flash_attention
    Llama
    unfused_attention_kernels
    logger
    tensor
    decoder_masked_multihead_attention
    cublas)
