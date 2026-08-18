# Copyright (c) OpenMMLab. All rights reserved.

add_library(attention_kernels STATIC
            ../utils.cc
            ../reduce.cu
            attention_sm70_64.cu
            attention_sm70_128.cu
            attention_sm70_256.cu
            attention_sm70_576.cu
            attention_sm75_64.cu
            attention_sm75_128.cu
            attention_sm75_256.cu
            attention_sm75_576.cu
            attention_sm80_64.cu
            attention_sm80_128.cu
            attention_sm80_192.cu
            attention_sm80_256.cu
            attention_sm80_576.cu
            decoding_sm70_64.cu
            decoding_sm70_128.cu
            decoding_sm70_256.cu
            decoding_sm70_576.cu
            decoding_sm75_64.cu
            decoding_sm75_128.cu
            decoding_sm75_256.cu
            decoding_sm75_576.cu
            decoding_sm80_64.cu
            decoding_sm80_128.cu
            decoding_sm80_192.cu
            decoding_sm80_256.cu
            decoding_sm80_576.cu
            )
set_property(TARGET attention_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(attention_kernels PRIVATE -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr  -Xptxas=-v --threads 8>)
target_link_libraries(attention_kernels PRIVATE nvidia::cutlass::cutlass)
