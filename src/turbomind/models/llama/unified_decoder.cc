
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/kernels/bert_preprocess_kernels.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void UnifiedDecoder<T>::allocateBuffer(size_t num_token, size_t pf_batch_size, size_t pf_max_q_len, size_t pf_max_k_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (pf_batch_size) {
        attention_mask_ =
            (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * pf_batch_size * pf_max_q_len * pf_max_k_len, false);
        padding_offset_ =
            (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * pf_batch_size * pf_max_q_len, false);
        cu_seqlens_ = (int*)allocator_->reMalloc(cu_seqlens_, sizeof(int) * (pf_batch_size + 1), false);
    }
}

template<typename T>
void UnifiedDecoder<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    allocator_->free((void**)&padding_offset_);
    allocator_->free((void**)&cu_seqlens_);
    allocator_->free((void**)&attention_mask_);
    allocator_->free((void**)&h_pinned_token_num_ptr_, true);
}

template<typename T>
void UnifiedDecoder<T>::initialize(const LlamaAttentionParams& attn_params,
                                   size_t                      kv_head_num,
                                   bool                        use_fmha,
                                   int                         cache_block_seq_len,
                                   int                         quant_policy)
{
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);

    attn_layer_ = new UnifiedAttentionLayer<T>(head_num_,
                                               kv_head_num,
                                               size_per_head_,
                                               attn_params,
                                               tensor_para_,
                                               stream_,
                                               cublas_wrapper_,
                                               allocator_,
                                               is_free_buffer_after_forward_,
                                               use_fmha,
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
                                        size_t                         pf_batch_size,
                                        size_t                         pf_max_q_len,
                                        size_t                         pf_max_k_len,
                                        size_t                         dc_batch_size,
                                        int                            layer_id,
                                        const LlamaAttentionWeight<T>* weight)
{
    TensorMap inputs(*_inputs);
    inputs.insert("input_query", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, attn_io});
    inputs.insert("layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id});
    if (pf_batch_size) {
        inputs.insert("attention_mask",
                      {MEMORY_GPU, dtype_, {pf_batch_size, 1, pf_max_q_len, pf_max_k_len}, attention_mask_});
        const size_t pf_token_num = token_num - dc_batch_size;
        inputs.insert("padding_offset", {MEMORY_GPU, TYPE_INT32, {pf_token_num}, padding_offset_});
        inputs.insert("cu_seqlens", {MEMORY_GPU, TYPE_INT32, {pf_batch_size + 1}, cu_seqlens_});
    }

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
     *   \param decoder_input [num_token, hidden_units], float
     *   \param input_lengths [batch_size], int
     *   \param history_lengths [batch_size], int
     *   \param context_legnths [batch_size], int
     *   \param output_norm_weight [hidden_dims], float
     *   \param max_q_len [1], int on cpu
     *   \param max_kv_len [1], int on cpu
     *   \param max_seq_len [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
     *   \param value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
     *   \param last_token_hidden_units [batch_size, hidden_units]
     */

    // Session sess{};

    const size_t token_num = inputs->at("decoder_input").shape[0];

    const int pf_max_q_len  = inputs->getVal<int>("pf_max_q_len");
    const int pf_max_k_len  = inputs->getVal<int>("pf_max_k_len");
    const int pf_batch_size = inputs->getVal<int>("pf_batch_size");
    const int dc_batch_size = inputs->getVal<int>("dc_batch_size");

    const int* input_length   = inputs->getPtr<int>("input_lengths");
    const int* context_length = inputs->getPtr<int>("context_lengths");

    T* decoder_input_output = inputs->getPtr<T>("decoder_input");
    T* decoder_output       = outputs->getPtr<T>("decoder_output");

    T* last_token_hidden_units = outputs->getPtr<T>("last_token_hidden_units");

    allocateBuffer(token_num, pf_batch_size, pf_max_q_len, pf_max_k_len);

    const int pf_offset = dc_batch_size;

    if (pf_batch_size) {
        FT_CHECK(padding_offset_);

        size_t tmp_token_num{};
        // `cu_seqlens` is exclusive sum of "input_lengths"
        invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                           &tmp_token_num,  // updated token num
                                           padding_offset_,
                                           cu_seqlens_,
                                           input_length + pf_offset,
                                           pf_batch_size,
                                           pf_max_q_len,
                                           stream_);
        sync_check_cuda_error();

        FT_CHECK(tmp_token_num == token_num - dc_batch_size);

        invokeCreateCausalMasks(attention_mask_,
                                input_length + pf_offset,
                                context_length + pf_offset,
                                pf_max_q_len,
                                pf_max_k_len,
                                pf_batch_size,
                                stream_);
        sync_check_cuda_error();
    }

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
        /////////////////////////////////////////////
        /// self-attention
        forwardSelfAttn(decoder_output,
                        outputs,
                        inputs,
                        token_num,
                        pf_batch_size,
                        pf_max_q_len,
                        pf_max_k_len,
                        dc_batch_size,
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
                                    decoder_output + pf_offset * hidden_units_,
                                    cu_seqlens_,
                                    hidden_units_,
                                    pf_batch_size,
                                    stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class UnifiedDecoder<float>;
template class UnifiedDecoder<half>;

}  // namespace turbomind
