

#include <cuda_runtime.h>

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
UnifiedDecoder<T>::UnifiedDecoder(const ModelParam&     model,
                                  const AttentionParam& attn,
                                  const MoeParam&       moe,
                                  const LoraParam&      lora,
                                  const NcclParam&      tp,
                                  const EngineParam&    engine,
                                  const Context<T>&     ctx):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    rmsnorm_eps_(model.norm_eps),
    stream_(ctx.stream),
    allocator_(ctx.allocator.get()),
    tp_(tp),
    dtype_(getTensorType<T>()),
    tune_layer_num_(model.tune_layer_num)
{

    attn_layer_ = std::make_unique<UnifiedAttentionLayer<T>>(model, attn, lora, tp, ctx);

    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer<T>>(model, moe, tp, ctx);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer<T>>(model, tp, ctx);
    }

    rotary_emb_ = std::make_unique<RotaryEmbeddingV2<T>>(attn, engine.session_len, ctx.stream, ctx.allocator.get());

    check_cuda_error(cudaEventCreateWithFlags(&ev_h_cu_x_, cudaEventDisableTiming));
}

template<typename T>
UnifiedDecoder<T>::~UnifiedDecoder()
{
    freeBuffer();
    check_cuda_error(cudaEventDestroy(ev_h_cu_x_));
}

template<typename T>
void UnifiedDecoder<T>::allocateBuffer(size_t batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    cu_q_len_   = (int*)allocator_->reMalloc(cu_q_len_, 2 * sizeof(int) * (batch_size + 1), false);
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
void UnifiedDecoder<T>::forwardSelfAttn(T*                attn_io,
                                        TensorMap*        _outputs,
                                        const TensorMap*  _inputs,
                                        size_t            token_num,
                                        size_t            batch_size,
                                        int               layer_id,
                                        const WeightType* weight)
{
    TensorMap inputs(*_inputs);
    inputs.insert("input_query", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, attn_io});
    inputs.insert("layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id});
    inputs.insert("cu_q_len", {MEMORY_GPU, TYPE_INT32, {batch_size + 1}, cu_q_len_});
    inputs.insert("cu_k_len", {MEMORY_GPU, TYPE_INT32, {batch_size + 1}, cu_k_len_});
    inputs.insert("h_cu_q_len", {MEMORY_CPU, TYPE_INT32, {batch_size + 1}, h_cu_q_len_});
    inputs.insert("h_cu_k_len", {MEMORY_CPU, TYPE_INT32, {batch_size + 1}, h_cu_k_len_});
    inputs.insert("cos_sin",
                  {MEMORY_GPU, getTensorType<T>(), {token_num, (size_t)rotary_emb_->dim_}, rotary_emb_->cos_sin_});
    inputs.insert("q2p", {MEMORY_GPU, TYPE_INT32, {token_num}, rotary_emb_->q2p_});

    TensorMap outputs(*_outputs);
    outputs.insert("hidden_features", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, attn_io});

    attn_layer_->forward(&outputs, &inputs, &weight->self_attn_weights);
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

        check_cuda_error(cudaEventRecord(ev_h_cu_x_, stream_));
    }

    const int pf_offset = dc_batch_size;

    /////////////////////////////////////////////
    /// RMSNorm
    invokeRMSNorm(decoder_output,
                  decoder_input_output,
                  weights->at(0)->self_attn_norm_weights,
                  hidden_units_,
                  token_num,
                  rmsnorm_eps_,
                  stream_);
    sync_check_cuda_error();

    count_and_fix(decoder_output, token_num * hidden_units_, Concat("norm0", 0), 2);

    {
        RotaryEmbeddingV2Param params;
        params.rope_theta = inputs->getPtr<float>("rope_theta");
        params.q_len      = cu_q_len_;
        params.k_len      = cu_k_len_;
        params.h_q_len    = h_cu_q_len_;
        params.h_k_len    = h_cu_k_len_;
        params.dc_size    = dc_batch_size;
        params.batch_size = batch_size;
        params.token_num  = token_num;
        rotary_emb_->forward(params);
    }

    for (size_t layer = 0; layer < layer_num_; ++layer) {

        /// TODO: do not skip the layers when they are heterogeneous
        if (isTuning() && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention
        forwardSelfAttn(decoder_output,  //
                        outputs,
                        inputs,
                        token_num,
                        batch_size,
                        layer,
                        weights->at(layer));

        count_and_fix(decoder_output, token_num * hidden_units_, Concat("attn_block", layer), 2);

        invokeBiasResidualRMSNorm(decoder_input_output,
                                  decoder_output,
                                  weights->at(layer)->ffn_norm_weights,
                                  weights->at(layer)->self_attn_weights.output.bias,
                                  hidden_units_,
                                  token_num,
                                  rmsnorm_eps_,
                                  stream_);
        sync_check_cuda_error();

        count_and_fix(decoder_input_output, token_num * hidden_units_, Concat("residual0", layer), 2);
        count_and_fix(decoder_output, token_num * hidden_units_, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        const bool is_moe = !weights->at(layer)->moe_weights.experts.empty();
        if (is_moe) {
            moe_ffn_layer_->forward(nullptr, decoder_output, token_num, layer, weights->at(layer)->moe_weights);
        }

        if (weights->at(layer)->ffn_weights.output.kernel) {
            int       layer_id   = layer;  // int is needed
            bool      all_reduce = !is_moe;
            TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, decoder_output}},
                                 {"layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id}},
                                 {"all_reduce", {MEMORY_CPU, TYPE_BOOL, {1}, &all_reduce}}};
            TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, dtype_, {token_num, hidden_units_}, decoder_output}}};
            if (inputs->isExist("lora_mask")) {
                ffn_inputs.insert({"lora_mask", inputs->at("lora_mask")});
            }
            ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &weights->at(layer)->ffn_weights);
        }

        if (is_moe) {
            moe_ffn_layer_->reduce(decoder_output, token_num, (bool)ffn_layer_, layer, weights->at(layer)->moe_weights);
        }

        count_and_fix(decoder_output, token_num * hidden_units_, Concat("ffn_block", layer), 2);

        const bool is_last_layer = layer == layer_num_ - 1;

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

        count_and_fix(decoder_input_output, token_num * hidden_units_, Concat("residual1", layer), 2);
        count_and_fix(decoder_output, token_num * hidden_units_, Concat("norm0", layer + 1), 2);
    }

    if (dc_batch_size) {
        check_cuda_error(cudaMemcpyAsync(last_token_hidden_units,
                                         decoder_output,
                                         sizeof(T) * dc_batch_size * hidden_units_,
                                         cudaMemcpyDefault,
                                         stream_));
        count_and_fix(last_token_hidden_units, dc_batch_size * hidden_units_, "dc_out", 2);
    }

    if (pf_batch_size) {
        invokeGetFeatureOfLastToken(last_token_hidden_units + pf_offset * hidden_units_,  //
                                    decoder_output,
                                    cu_q_len_ + pf_offset,
                                    hidden_units_,
                                    pf_batch_size,
                                    stream_);
        sync_check_cuda_error();
        count_and_fix(last_token_hidden_units + pf_offset * hidden_units_, pf_batch_size * hidden_units_, "pf_out", 2);
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }

    // Wait for `h_cu_q/k_len_` to be consumed
    check_cuda_error(cudaEventSynchronize(ev_h_cu_x_));
}

#ifdef ENABLE_FP32
template class UnifiedDecoder<float>;
#endif
template class UnifiedDecoder<half>;
#ifdef ENABLE_BF16
template class UnifiedDecoder<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
