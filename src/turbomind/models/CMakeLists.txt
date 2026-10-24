cmake_minimum_required(VERSION 3.25)

add_library(target_hidden_projection_kernels STATIC
        speculative/eagle3/target_hidden_projection_kernels.cu)
set_property(TARGET target_hidden_projection_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET target_hidden_projection_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(models STATIC
        batch_status.cc
        language_model.cc
        input_processor.cc
        input_processor_speculative.cc
        output_processor.cc
        speculative/registry.cc
        speculative/fixed_chain_policy.cc
        speculative/fixed_chain_setup.cc
        speculative/fixed_chain_model.cc
        speculative/collect_hidden_states.cc
        speculative/eagle3/eagle3_model.cc
        speculative/eagle3/target_hidden_projection.cc
        speculative/eagle3/eagle3_weight.cc
        speculative/qwen3_5_mtp/qwen3_5_mtp_weight.cc
        speculative/qwen3_5_mtp/target_final_hidden.cc
        speculative/qwen3_5_mtp/qwen3_5_mtp_model.cc
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
        linear_attn
        generation
        core
        memory
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
target_link_libraries(models PRIVATE
        target_hidden_projection_kernels
        draft_carry_kernels
        speculative_sequence_kernels
        device_comm)
target_compile_options(models PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --generate-line-info --threads=${NVCC_THREADS}>)

if(BUILD_TEST)
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
