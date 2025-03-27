

#include <cuda_runtime.h>
#include <iterator>
#include <numeric>

#include "src/turbomind/kernels/core/math.h"
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
                                  const EngineParam&    engine,
                                  const AttentionParam& attn,
                                  const MoeParam&       moe,
                                  const LoraParam&      lora,
                                  const Context<T>&     ctx):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    rmsnorm_eps_(model.norm_eps),
    stream_(ctx.stream),
    allocator_(ctx.allocator.get()),
    d_comm_(ctx.comm.d_comm),
    dtype_(getTensorType<T>()),
    tune_layer_num_(model.tune_layer_num)
{
    attn_layer_ = std::make_unique<UnifiedAttentionLayer<T>>(model, attn, lora, attn_tp_size_, ctx);

    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer<T>>(model, moe, mlp_tp_size_, ctx);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer<T>>(model, ctx);
    }

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

template<class T>
typename UnifiedAttentionLayer<T>::ForwardParam UnifiedDecoder<T>::InitAttnFwdParam(const TensorMap& inputs,
                                                                                    const TensorMap& outputs)
{
    int* h_q_len = inputs.getPtr<int>("h_q_len");
    int* h_k_len = inputs.getPtr<int>("h_k_len");

    const int bsz = inputs.at("h_q_len").size();

    {  // compute cumulative lengths

        h_cu_k_len_ = h_cu_q_len_ + bsz + 1;
        cu_k_len_   = cu_q_len_ + bsz + 1;

        h_cu_q_len_[0] = h_cu_k_len_[0] = 0;

        for (int i = 1; i <= bsz; ++i) {
            h_cu_q_len_[i] = h_cu_q_len_[i - 1] + h_q_len[i - 1];
            h_cu_k_len_[i] = h_cu_k_len_[i - 1] + h_k_len[i - 1];
        }

        check_cuda_error(
            cudaMemcpyAsync(cu_q_len_, h_cu_q_len_, 2 * sizeof(int) * (bsz + 1), cudaMemcpyDefault, stream_));

        check_cuda_error(cudaEventRecord(ev_h_cu_x_, stream_));
    }

    typename UnifiedAttentionLayer<T>::ForwardParam param{};

    param.h_q_len        = h_q_len;
    param.h_k_len        = h_k_len;
    param.cu_q_len       = cu_q_len_;
    param.cu_k_len       = cu_k_len_;
    param.h_cu_q_len     = h_cu_q_len_;
    param.h_cu_k_len     = h_cu_k_len_;
    param.decode_num     = inputs.getVal<int>("dc_batch_size");
    param.prefil_num     = inputs.getVal<int>("pf_batch_size");
    param.is_finished    = inputs.getPtr<bool>("finished");
    param.rope_base      = inputs.getPtr<float>("rope_theta");
    param.cu_block_count = inputs.getPtr<int>("cu_block_counts");
    param.block_ptrs     = outputs.getPtr<void*>("block_ptrs");

    return param;
}

template<typename T>
void UnifiedDecoder<T>::AllreduceResidualRMSnorm(T*         hidden_states,
                                                 T*         residual,
                                                 const T*   bias,
                                                 const T*   weight,
                                                 int        token_num,
                                                 int        group0,
                                                 int        group1,
                                                 const int* local_token_nums)
{
    if (0) {}
    else if (group0 || group1) {
        d_comm_->AllreduceResidualBiasRMSnormEx(hidden_states,
                                                residual,
                                                bias,
                                                weight,
                                                rmsnorm_eps_,
                                                hidden_units_,
                                                dtype_,
                                                group0,
                                                group1,
                                                local_token_nums,
                                                stream_);
        sync_check_cuda_error();
    }
    else if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(
            hidden_states, residual, bias, weight, rmsnorm_eps_, hidden_units_, token_num, dtype_, 0, stream_);
        sync_check_cuda_error();
    }
    else {
        invokeBiasResidualRMSNorm(
            residual, hidden_states, weight, bias, hidden_units_, token_num, rmsnorm_eps_, stream_);
        sync_check_cuda_error();
    }
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

    const size_t local_token_num = inputs->at("decoder_input").shape[0];

    const int pf_batch_size = inputs->getVal<int>("pf_batch_size");
    const int dc_batch_size = inputs->getVal<int>("dc_batch_size");
    const int batch_size    = pf_batch_size + dc_batch_size;

    T* residual_data = inputs->getPtr<T>("decoder_input");
    // This is pointer to symmetric memory
    T* hidden_states_data      = outputs->getPtr<T>("decoder_output");
    T* last_token_hidden_units = outputs->getPtr<T>("last_token_hidden_units");

    const int pf_offset = dc_batch_size;

    constexpr auto device = MEMORY_GPU;

    core::Tensor global_hidden_states{hidden_states_data, {(int)local_token_num, (int)hidden_units_}, device};
    core::Tensor local_hidden_states{global_hidden_states};

    core::Tensor local_residual{residual_data, {(int)local_token_num, (int)hidden_units_}, device};

    core::ssize_t global_token_num = local_token_num;
    const int*    local_token_nums = inputs->getPtr<int>("local_token_nums", nullptr);

    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK(local_token_nums);
        std::vector cumul_token_nums(attn_dp_size_ + 1, 0);
        std::inclusive_scan(local_token_nums, local_token_nums + attn_dp_size_, cumul_token_nums.begin() + 1);
        global_token_num     = cumul_token_nums.back();
        global_hidden_states = core::Tensor{hidden_states_data, {global_token_num, (int)hidden_units_}, device};
        const int offset     = cumul_token_nums[attn_dp_rank_];
        local_hidden_states  = global_hidden_states.slice({offset, 0}, {(int)local_token_num, -1});
    }

    auto attn_fwd_param   = InitAttnFwdParam(*inputs, *outputs);
    attn_fwd_param.input  = local_hidden_states;
    attn_fwd_param.output = local_hidden_states;

    invokeRMSNorm(local_hidden_states, local_residual, weights->at(0)->self_attn_norm_weights, rmsnorm_eps_, stream_);
    sync_check_cuda_error();

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);

    for (size_t layer = 0; layer < layer_num_; ++layer) {

        /// TODO: do not skip the layers when they are heterogeneous
        if (isTuning() && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention
        {
            auto param     = attn_fwd_param;
            param.weights  = &weights->at(layer)->self_attn_weights;
            param.layer_id = layer;
            attn_layer_->forward(std::move(param));
        }

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        AllreduceResidualRMSnorm(global_hidden_states.data<T>(),
                                 local_residual.data<T>(),
                                 weights->at(layer)->self_attn_weights.output.bias,
                                 weights->at(layer)->ffn_norm_weights,
                                 local_token_num,
                                 attn_tp_group_,
                                 0,
                                 local_token_nums);

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        const bool is_moe = !weights->at(layer)->moe_weights.experts.empty();
        if (is_moe) {
            // Writes to internal buffer
            moe_ffn_layer_->forward(
                nullptr, global_hidden_states.data<T>(), global_token_num, layer, weights->at(layer)->moe_weights);
        }

        if (weights->at(layer)->ffn_weights.output.kernel) {
            ffn_layer_->forward(
                {global_hidden_states, global_hidden_states, &weights->at(layer)->ffn_weights, (int)layer});
        }

        if (is_moe) {
            moe_ffn_layer_->reduce(global_hidden_states.data<T>(),
                                   global_token_num,
                                   (bool)ffn_layer_,
                                   layer,
                                   weights->at(layer)->moe_weights);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);

        const bool is_last_layer = layer == layer_num_ - 1;

        auto scale_weight = !is_last_layer ? weights->at(layer + 1)->self_attn_norm_weights :
                                             inputs->at("output_norm_weight").getPtr<T>();

        AllreduceResidualRMSnorm(global_hidden_states.data<T>(),
                                 local_residual.data<T>(),
                                 weights->at(layer)->ffn_weights.output.bias,
                                 scale_weight,
                                 local_token_num,
                                 0,
                                 attn_tp_group_,
                                 local_token_nums);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);
    }

    if (dc_batch_size) {
        check_cuda_error(cudaMemcpyAsync(last_token_hidden_units,
                                         (T*)local_hidden_states.raw_data(),
                                         sizeof(T) * dc_batch_size * hidden_units_,
                                         cudaMemcpyDefault,
                                         stream_));
        TM_DEBUG_RAW(last_token_hidden_units, dc_batch_size * hidden_units_, "dc_out", 2);
    }

    if (pf_batch_size) {
        invokeGetFeatureOfLastToken(last_token_hidden_units + pf_offset * hidden_units_,  //
                                    (T*)local_hidden_states.raw_data(),
                                    cu_q_len_ + pf_offset,
                                    hidden_units_,
                                    pf_batch_size,
                                    stream_);
        sync_check_cuda_error();
        TM_DEBUG_RAW(last_token_hidden_units + pf_offset * hidden_units_, pf_batch_size * hidden_units_, "pf_out", 2);
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
