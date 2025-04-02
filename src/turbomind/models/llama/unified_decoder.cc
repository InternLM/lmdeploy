

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

    attn_fwd_param_ = attn_layer_->CreateForwardParam(engine.max_batch_size);

    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer<T>>(model, moe, mlp_tp_size_, ctx);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer<T>>(model, ctx);
    }
}

template<typename T>
UnifiedDecoder<T>::~UnifiedDecoder()
{
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
void UnifiedDecoder<T>::Forward(core::TensorMap& args, const std::vector<WeightType*>& weights)
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

    const int decode_num = *args.at("decode_num").data<int>();
    const int prefil_num = *args.at("prefil_num").data<int>();
    const int batch_size = prefil_num + decode_num;

    constexpr auto device = MEMORY_GPU;

    core::Tensor       local_residual   = args.at("decoder_input");
    core::Tensor_<int> local_token_nums = args.at("local_token_nums");

    core::Tensor global_hidden_states = args.at("decoder_output");
    core::Tensor local_hidden_states  = global_hidden_states;

    const auto global_token_num = global_hidden_states.shape(0);
    const auto local_token_num  = local_residual.shape(0);

    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK_EQ(local_token_nums.size(), attn_dp_size_);
        std::vector cumul_token_nums(attn_dp_size_ + 1, 0);
        std::inclusive_scan(
            local_token_nums.data(), local_token_nums.data() + attn_dp_size_, cumul_token_nums.begin() + 1);
        const int offset    = cumul_token_nums[attn_dp_rank_];
        local_hidden_states = global_hidden_states.slice({offset, 0}, {local_token_num, -1});
    }

    Initialize(*attn_fwd_param_, args, local_hidden_states, local_hidden_states);

    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->self_attn_norm_weights, rmsnorm_eps_, stream_);
    sync_check_cuda_error();

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);

    for (size_t layer = 0; layer < layer_num_; ++layer) {

        /// TODO: do not skip the layers when they are heterogeneous
        if (isTuning() && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention
        SetLayer(*attn_fwd_param_, &weights.at(layer)->self_attn_weights, layer);
        attn_layer_->forward(*attn_fwd_param_);

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        AllreduceResidualRMSnorm(global_hidden_states.data<T>(),
                                 local_residual.data<T>(),
                                 weights.at(layer)->self_attn_weights.output.bias,
                                 weights.at(layer)->ffn_norm_weights,
                                 local_token_num,
                                 attn_tp_group_,
                                 0,
                                 local_token_nums.data());

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        const bool is_moe = !weights.at(layer)->moe_weights.experts.empty();
        if (is_moe) {
            // Writes to internal buffer
            moe_ffn_layer_->forward(
                nullptr, global_hidden_states.data<T>(), global_token_num, layer, weights.at(layer)->moe_weights);
        }

        if (weights.at(layer)->ffn_weights.output.kernel) {
            ffn_layer_->forward(
                {global_hidden_states, global_hidden_states, &weights.at(layer)->ffn_weights, (int)layer});
        }

        if (is_moe) {
            moe_ffn_layer_->reduce(global_hidden_states.data<T>(),
                                   global_token_num,
                                   (bool)ffn_layer_,
                                   layer,
                                   weights.at(layer)->moe_weights);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);

        const bool is_last_layer = layer == layer_num_ - 1;

        auto scale_weight =
            !is_last_layer ? weights.at(layer + 1)->self_attn_norm_weights : args.at("output_norm_weight").data<T>();

        AllreduceResidualRMSnorm(global_hidden_states.data<T>(),
                                 local_residual.data<T>(),
                                 weights.at(layer)->ffn_weights.output.bias,
                                 scale_weight,
                                 local_token_num,
                                 0,
                                 attn_tp_group_,
                                 local_token_nums.data());
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);
    }

    T* last_token_hidden_units = args.at("last_token_hidden_units").data<T>();

    if (decode_num) {
        check_cuda_error(cudaMemcpyAsync(last_token_hidden_units,
                                         (T*)local_hidden_states.raw_data(),
                                         sizeof(T) * decode_num * hidden_units_,
                                         cudaMemcpyDefault,
                                         stream_));
        TM_DEBUG_RAW(last_token_hidden_units, decode_num * hidden_units_, "dc_out", 2);
    }

    if (prefil_num) {
        invokeGetFeatureOfLastToken(last_token_hidden_units + decode_num * hidden_units_,  //
                                    (T*)local_hidden_states.raw_data(),
                                    d_cu_q_len(*attn_fwd_param_) + decode_num,
                                    hidden_units_,
                                    prefil_num,
                                    stream_);
        sync_check_cuda_error();
        TM_DEBUG_RAW(last_token_hidden_units + decode_num * hidden_units_, prefil_num * hidden_units_, "pf_out", 2);
    }

    Finalize(*attn_fwd_param_);
}

#ifdef ENABLE_FP32
template class UnifiedDecoder<float>;
#endif
template class UnifiedDecoder<half>;
#ifdef ENABLE_BF16
template class UnifiedDecoder<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
