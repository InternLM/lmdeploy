# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

add_subdirectory(fused_multi_head_attention)

find_package(CUDAToolkit REQUIRED)

add_library(Llama STATIC
        LlamaV2.cc
        LlamaBatch.cc
        LlamaCacheManager.cc
        BlockManager.cc
        SequenceManager.cc
        LlamaContextDecoder.cc
        LlamaContextAttentionLayer.cc
        LlamaDecoderSelfAttentionLayer.cc
        LlamaDecoder.cc
        LlamaWeight.cc
        LlamaDecoderLayerWeight.cc
        LlamaFfnLayer.cc
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
        decoder_masked_multihead_attention
        decoder_multihead_attention
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

if (NOT MSVC)
        add_subdirectory(flash_attention2)
        target_link_libraries(Llama PUBLIC flash_attention2)
endif()

add_executable(llama_gemm llama_gemm.cc)
target_link_libraries(llama_gemm PUBLIC CUDA::cudart gpt_gemm_func memory_utils cuda_utils logger)

install(TARGETS llama_gemm DESTINATION ${CMAKE_SOURCE_DIR}/lmdeploy/bin)

find_package(Catch2 3 QUIET)
if (Catch2_FOUND)
        add_executable(test_cache_manager test_cache_manager.cc)
        target_link_libraries(test_cache_manager PRIVATE Llama Catch2::Catch2WithMain)
endif ()

