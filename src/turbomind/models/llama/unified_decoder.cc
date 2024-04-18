
#include "src/turbomind/models/llama/unified_decoder.h"
// #include "src/turbomind/kernels/bert_preprocess_kernels.h"
// #include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void UnifiedDecoder<T>::allocateBuffer(size_t batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    cu_q_len_ = (int*)allocator_->reMalloc(cu_q_len_, 2 * sizeof(int) * (batch_size + 1), false);

    h_cu_q_len_ = (int*)allocator_->reMalloc(h_cu_q_len_, 2 * sizeof(int) * (batch_size + 1), false, true);
}

template<typename T>
void UnifiedDecoder<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    allocator_->free((void**)&cu_q_len_);

    allocator_->free((void**)&h_cu_q_len_, true);
}

template<typename T>
void UnifiedDecoder<T>::initialize(const LlamaAttentionParams& attn_params,
                                   size_t                      kv_head_num,
                                   int                         cache_block_seq_len,
                                   int                         quant_policy)
{
    attn_layer_ = new UnifiedAttentionLayer<T>(head_num_,
                                               kv_head_num,
                                               size_per_head_,
                                               attn_params,
                                               tensor_para_,
                                               lora_params_,
                                               stream_,
                                               cublas_wrapper_,
                                               allocator_,
                                               is_free_buffer_after_forward_,
                                               cache_block_seq_len,
                                               quant_policy);

    ffn_layer_ = new LlamaFfnLayer<T>(head_num_,
                                      size_per_head_,
                                      inter_size_,
                                      tensor_para_,
                                      stream_,
                                      cublas_wrapper_,
                                      allocator_,
                                      is_free_buffer_after_forward_);
}

template<typename T>
void UnifiedDecoder<T>::forwardSelfAttn(T*                             attn_io,
                                        TensorMap*                     _outputs,
                                        const TensorMap*               _inputs,
                                        size_t                         token_num,
                                        size_t                         batch_size,
                                        int                            layer_id,
                                        const LlamaAttentionWeight<T>* weight)
{
    TensorMap inputs(*_inputs);
    inputs.insert("input_query", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, attn_io});
    inputs.insert("layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id});
    inputs.insert("cu_q_len", {MEMORY_GPU, TYPE_INT32, {batch_size + 1}, cu_q_len_});
    inputs.insert("cu_k_len", {MEMORY_GPU, TYPE_INT32, {batch_size + 1}, cu_k_len_});
    inputs.insert("h_cu_q_len", {MEMORY_CPU, TYPE_INT32, {batch_size + 1}, h_cu_q_len_});
    inputs.insert("h_cu_k_len", {MEMORY_CPU, TYPE_INT32, {batch_size + 1}, h_cu_k_len_});

    TensorMap outputs(*_outputs);
    outputs.insert("hidden_features", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, attn_io});

    attn_layer_->forward(&outputs, &inputs, weight);
}

template<typename T>
UnifiedDecoder<T>::~UnifiedDecoder()
{
    delete attn_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void UnifiedDecoder<T>::forward(TensorMap* outputs, const TensorMap* inputs, const std::vector<WeightType*>* weights)
{
    /**
     * input tensors:
     *   \param decoder_input [token_num, hidden_units], float
     *   \param output_norm_weight [hidden_dims], float
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param last_token_hidden_units [batch_size, hidden_units]
     *   \param block_ptrs [total_block_counts], void*
     */

    const size_t token_num = inputs->at("decoder_input").shape[0];

    const int pf_batch_size = inputs->getVal<int>("pf_batch_size");
    const int dc_batch_size = inputs->getVal<int>("dc_batch_size");
    const int batch_size    = pf_batch_size + dc_batch_size;

    const int* h_q_len = inputs->getPtr<int>("h_q_len");
    const int* h_k_len = inputs->getPtr<int>("h_k_len");

    T* decoder_input_output    = inputs->getPtr<T>("decoder_input");
    T* decoder_output          = outputs->getPtr<T>("decoder_output");
    T* last_token_hidden_units = outputs->getPtr<T>("last_token_hidden_units");

    {  // compute cumulative lengths
        h_cu_k_len_ = h_cu_q_len_ + batch_size + 1;
        cu_k_len_   = cu_q_len_ + batch_size + 1;

        h_cu_q_len_[0] = h_cu_k_len_[0] = 0;

        for (int i = 1; i <= batch_size; ++i) {
            h_cu_q_len_[i] = h_cu_q_len_[i - 1] + h_q_len[i - 1];
            h_cu_k_len_[i] = h_cu_k_len_[i - 1] + h_k_len[i - 1];
        }

        check_cuda_error(
            cudaMemcpyAsync(cu_q_len_, h_cu_q_len_, 2 * sizeof(int) * (batch_size + 1), cudaMemcpyDefault, stream_));
    }

    const int pf_offset = dc_batch_size;

    // Compare(decoder_input_output, token_num * hidden_units_, "decoder_input", kCmpRead, stream_);

    // printf("%d %f\n", (int)token_num, rmsnorm_eps_);

    /////////////////////////////////////////////
    /// RMSNorm
    invokeRootMeanSquareNorm(decoder_output,
                             decoder_input_output,
                             weights->at(0)->self_attn_norm_weights,
                             rmsnorm_eps_,
                             token_num,
                             hidden_units_,
                             stream_);
    sync_check_cuda_error();

    for (size_t layer = 0; layer < num_layer_; ++layer) {

        // Compare(decoder_output, token_num * hidden_units_, "attn_input", kCmpRead, stream_);

        /////////////////////////////////////////////
        /// self-attention
        forwardSelfAttn(decoder_output,  //
                        outputs,
                        inputs,
                        token_num,
                        batch_size,
                        layer,
                        &weights->at(layer)->self_attn_weights);

        invokeFusedAddBiasResidualRMSNorm(decoder_input_output,
                                          decoder_output,
                                          weights->at(layer)->self_attn_weights.output.bias,
                                          weights->at(layer)->ffn_norm_weights,
                                          rmsnorm_eps_,
                                          token_num,
                                          hidden_units_,
                                          stream_);
        sync_check_cuda_error();

        ////////////////////////////////////////////
        /// feed-forward network
        TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, decoder_output}}};
        TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, decoder_output}}};
        if (lora_params_.policy == 1 && inputs->isExist("lora_mask")) {
            ffn_inputs.insert({"lora_mask", inputs->at("lora_mask")});
        }

        ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &weights->at(layer)->ffn_weights);

        const bool is_last_layer = layer == num_layer_ - 1;

        auto scale_weight = !is_last_layer ? weights->at(layer + 1)->self_attn_norm_weights :
                                             inputs->at("output_norm_weight").getPtr<T>();
        invokeFusedAddBiasResidualRMSNorm(decoder_input_output,
                                          decoder_output,
                                          weights->at(layer)->ffn_weights.output.bias,
                                          scale_weight,
                                          rmsnorm_eps_,
                                          token_num,
                                          hidden_units_,
                                          stream_);
        sync_check_cuda_error();
    }

    if (dc_batch_size) {
        check_cuda_error(cudaMemcpyAsync(last_token_hidden_units,
                                         decoder_output,
                                         sizeof(T) * dc_batch_size * hidden_units_,
                                         cudaMemcpyDefault,
                                         stream_));
    }

    if (pf_batch_size) {
        invokeGetFeatureOfLastToken(last_token_hidden_units + pf_offset * hidden_units_,  //
                                    decoder_output,
                                    cu_q_len_ + pf_offset,
                                    hidden_units_,
                                    pf_batch_size,
                                    stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

#ifdef ENABLE_FP32
template class UnifiedDecoder<float>;
#endif
template class UnifiedDecoder<half>;
#ifdef ENABLE_BF16
template class UnifiedDecoder<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
