cmake_minimum_required(VERSION 3.8)

add_library(models STATIC
        language_model.cc
        input_processor.cc
        output_processor.cc
        llama/LlamaLinear.cu
        llama/BlockManager.cc
        llama/BlockTrie.cc
        llama/SequenceManager.cc
        llama/LlamaWeight.cc
        llama/LlamaDenseWeight.cc
        llama/LlamaDecoderLayerWeight.cc
        llama/LlamaFfnLayer.cc
        llama/moe_ffn_layer.cc
        llama/unified_decoder.cc
        llama/unified_attention_layer.cc
        llama/llama_kernels.cu
        llama/llama_utils.cu
        llama/mla_utils.cu)
set_property(TARGET models PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET models PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(models PUBLIC
        generation
        core
        gemm2
        rms_norm
        CUDA::cublas
        CUDA::cudart
        nvidia::cutlass::cutlass
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
