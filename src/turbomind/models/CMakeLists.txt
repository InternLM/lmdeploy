cmake_minimum_required(VERSION 3.25)

add_library(models STATIC
        language_model.cc
        input_processor.cc
        output_processor.cc
        linear_weight.cc
        norm_weight.cc
        layer_norm_weight.cc
        attention_weight.cc
        ffn_weight.cc
        moe_weight.cc
        delta_net_weight.cc
        decoder_layer_weight.cc
        model_weight.cc
        model_root.cc
        vision_model.cc
        qwenvit/qwenvit_kernels.cu
        qwenvit/qwenvit_block_weight.cc
        qwenvit/qwenvit_weight.cc
        qwenvit/qwenvit.cc
        internvit/internvit_kernels.cu
        internvit/internvit_block_weight.cc
        internvit/internvit_weight.cc
        internvit/internvit.cc
        llama/LlamaLinear.cu
        llama/BlockManager.cc
        llama/BlockTrie.cc
        llama/SequenceManager.cc
        llama/LlamaFfnLayer.cc
        llama/moe_ffn_layer.cc
        llama/unified_decoder.cc
        llama/unified_attention_layer.cc
        llama/llama_kernels.cu
        llama/llama_utils.cu
        llama/mla_utils.cu
        llama/GatedDeltaNetLayer.cc
        llama/gated_delta_net_kernels.cu)
set_property(TARGET models PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET models PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(models PUBLIC
        generation
        core
        gemm2
        rms_norm
        layer_norm
        CUDA::cublas
        CUDA::cudart
        nvidia::cutlass::cutlass
        activation_kernels
        activation
        attention
        decoding_kernels
        quantization_kernels
        cross_entropy_kernels
        unfused_attention_kernels
        gpt_kernels
        memory_utils
        cuda_utils
        anomaly_handler)
target_compile_options(models PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info --threads 8>)

if(BUILD_TEST)
    add_executable(bench_gated_delta_net
            llama/bench_gated_delta_net.cc)
    target_link_libraries(bench_gated_delta_net PRIVATE
            models
            CUDA::cudart)

    add_executable(bench_conv1d_silu
            llama/bench_conv1d_silu.cc)
    target_link_libraries(bench_conv1d_silu PRIVATE
            models
            CUDA::cudart)

    add_executable(test_mrope_position_ids
            qwenvit/test_mrope_position_ids.cu)
    set_property(TARGET test_mrope_position_ids PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(test_mrope_position_ids PRIVATE
            models
            attention
            core
            cuda_utils
            fmt::fmt
            CUDA::cudart)
endif()
