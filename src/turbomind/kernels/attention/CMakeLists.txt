# Copyright (c) OpenMMLab. All rights reserved.

add_subdirectory(kernel)

add_library(attention STATIC
            attention.cu
            decoding.cu
            kv_cache_utils_v2.cu
            cp_utils.cu
            registry.cu
            )
set_property(TARGET attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(attention PRIVATE -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math --expt-relaxed-constexpr -Xptxas=-v --threads 16>)
target_link_libraries(attention PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,attention_kernels>)
target_link_libraries(attention PRIVATE nvidia::cutlass::cutlass)

if (BUILD_TEST)
    target_compile_options(attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info>)

    add_executable(test_attention
        test_utils.cu
        test_attention.cu
        reference.cu)
    target_compile_options(test_attention PRIVATE
        --generate-line-info -O3 -use_fast_math --expt-relaxed-constexpr)
    target_link_libraries(test_attention PRIVATE
        attention
        # flash_attention
        nvidia::cutlass::cutlass
        Llama
        unfused_attention_kernels
        logger
        cublas)

    add_executable(test_quant test_quant.cu test_utils.cu)
    target_compile_options(test_quant PRIVATE
        --generate-line-info -O3 -use_fast_math --expt-relaxed-constexpr)
    target_link_libraries(test_quant PRIVATE
        nvidia::cutlass::cutlass
    )
endif ()
