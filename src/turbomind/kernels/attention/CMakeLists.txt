#Copyright(c) OpenMMLab.All rights reserved.

add_library(attention STATIC attention.cu decoding.cu kv_cache_utils_v2.cu utils.cc codegen
            / attention_sm70_128_f16.cu codegen / attention_sm75_128_f16.cu codegen / attention_sm80_128_bf16.cu codegen
            / attention_sm80_128_f16.cu codegen / decoding_sm70_128_f16_f16.cu codegen
            / decoding_sm70_128_f16_u4.cu codegen / decoding_sm70_128_f16_u8.cu codegen
            / decoding_sm75_128_f16_f16.cu codegen / decoding_sm75_128_f16_u4.cu codegen
            / decoding_sm75_128_f16_u8.cu codegen / decoding_sm80_128_bf16_bf16.cu codegen
            / decoding_sm80_128_bf16_u4.cu codegen / decoding_sm80_128_bf16_u8.cu codegen
            / decoding_sm80_128_f16_f16.cu codegen / decoding_sm80_128_f16_u4.cu codegen / decoding_sm80_128_f16_u8.cu)
    set_property(TARGET attention PROPERTY POSITION_INDEPENDENT_CODE ON)
        set_property(TARGET attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
            target_compile_options(attention PRIVATE - O3 $ < $ < COMPILE_LANGUAGE
                                   : CUDA >
                                   : -use_fast_math-- expt - relaxed - constexpr >)

                if (BUILD_TEST) target_compile_options(
                    attention PRIVATE $<$<COMPILE_LANGUAGE : CUDA> : -Xptxas = -v-- generate - line - info>)

                    add_executable(test_attention test_utils.cu test_attention.cu reference.cu)
                        target_compile_options(test_attention PRIVATE-- generate - line - info - O3
                                               - use_fast_math-- expt - relaxed - constexpr)
                            target_link_libraries(test_attention PRIVATE attention flash_attention Llama
                                                      unfused_attention_kernels logger tensor cublas)

                                add_executable(test_quant test_quant.cu test_utils.cu)
                                    target_compile_options(test_quant PRIVATE-- generate - line - info - O3
                                                           - use_fast_math-- expt - relaxed - constexpr) endif()
