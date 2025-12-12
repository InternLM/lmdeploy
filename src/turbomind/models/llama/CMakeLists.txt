# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)


find_package(CUDAToolkit REQUIRED)

add_library(Llama STATIC
        LlamaV2.cc
        LlamaBatch.cc
        LlamaLinear.cu
        BlockManager.cc
        BlockTrie.cc
        SequenceManager.cc
        LlamaWeight.cc
        LlamaDenseWeight.cc
        LlamaDecoderLayerWeight.cc
        LlamaFfnLayer.cc
        moe_ffn_layer.cc
        unified_decoder.cc
        unified_attention_layer.cc
        llama_kernels.cu
        llama_utils.cu
        mla_utils.cu
)
set_property(TARGET Llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama PUBLIC CUDA::cudart
        engine
        core
        gemm2
        CUDA::cublas
        nvidia::cutlass::cutlass
        rms_norm
        DynamicDecodeLayer
        activation_kernels
        activation
        attention
        decoding_kernels
        quantization_kernels
        unfused_attention_kernels
        gpt_kernels
        memory_utils
        cuda_utils
        logger
        anomaly_handler)
