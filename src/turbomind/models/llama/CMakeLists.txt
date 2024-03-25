# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)


find_package(CUDAToolkit REQUIRED)

add_library(Llama STATIC
        LlamaV2.cc
        LlamaBatch.cc
        BlockManager.cc
        SequenceManager.cc
        LlamaWeight.cc
        LlamaDecoderLayerWeight.cc
        LlamaFfnLayer.cc
        unified_decoder.cc
        unified_attention_layer.cc
        llama_kernels.cu
        llama_decoder_kernels.cu
        llama_utils.cu)
set_property(TARGET Llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama PUBLIC CUDA::cudart
        gemm_s4_f16
        cublasMMWrapper
        DynamicDecodeLayer
        activation_kernels
        attention
        bert_preprocess_kernels
        decoding_kernels
        unfused_attention_kernels
        custom_ar_kernels
        custom_ar_comm
        gpt_kernels
        tensor
        memory_utils
        nccl_utils
        cuda_utils
        logger
        llama_fmha)


add_executable(llama_gemm llama_gemm.cc)
target_link_libraries(llama_gemm PUBLIC CUDA::cudart gpt_gemm_func memory_utils cuda_utils logger)

install(TARGETS llama_gemm DESTINATION ${CMAKE_SOURCE_DIR}/lmdeploy/bin)

# find_package(Catch2 3 QUIET)
# if (Catch2_FOUND)
#         add_executable(test_cache_manager test_cache_manager.cc)
#         target_link_libraries(test_cache_manager PRIVATE Llama Catch2::Catch2WithMain)
# endif ()
