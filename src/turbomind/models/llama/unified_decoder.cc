

#include <numeric>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/engine/request.h"

// #include "dbg.h"

namespace turbomind {

struct UnifiedDecoder::HiddenStateLayout {
    // for tp mode
    Tensor            global_hidden_states;
    Tensor            local_hidden_states;
    Tensor            local_residual;
    int               local_token_num;
    int               global_token_num;
    std::vector<int>& local_token_nums;
    // for ep mode
    Tensor              partial_hidden_states;
    Tensor              partial_local_residual;
    std::vector<int>    token_nums;
    std::vector<size_t> counts;
    int                 max_tokens_per_rank;
};

UnifiedDecoder::HiddenStateLayout UnifiedDecoder::CreateHiddenStateLayout(TensorMap& args)
{
    Tensor      local_residual   = args.try_consume("input_embeds");
    const auto& local_token_nums = args.at("batch").data<BatchData*>()[0]->local_token_num;

    const auto local_token_num  = local_residual.shape(0);
    const auto global_token_num = std::accumulate(local_token_nums.begin(), local_token_nums.end(), ssize_t{});

    TM_CHECK_EQ(local_token_num, local_token_nums[attn_dp_rank_]);

    const DataType dtype = local_residual.dtype();

    Tensor global_hidden_states;
    if (d_comm_) {
        Buffer symm_buf      = args.at("symm_buf").buffer();
        global_hidden_states = {symm_buf.view(dtype), {global_token_num, (int)hidden_units_}};
    }
    else {
        global_hidden_states = {{global_token_num, (int)hidden_units_}, local_residual.dtype(), kDEVICE};
    }

    Tensor local_hidden_states;
    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK_EQ(local_token_nums.size(), attn_dp_size_);
        std::vector offsets(attn_dp_size_ + 1, 0);
        std::inclusive_scan(local_token_nums.data(), local_token_nums.data() + attn_dp_size_, offsets.begin() + 1);
        const int offset    = offsets[attn_dp_rank_];
        local_hidden_states = global_hidden_states.slice({offset, 0}, {local_token_num, -1});

        // dbg(attn_dp_size_, attn_dp_rank_, local_token_nums, local_token_num, global_token_num);
    }
    else {
        local_hidden_states = global_hidden_states;
    }

    Tensor              partial_hidden_states;
    Tensor              partial_local_residual;
    std::vector<int>    token_nums;
    std::vector<size_t> counts;
    int                 max_tokens_per_rank{-1};
    // split inputs for ep mode
    if (ep_size_ > 1) {
        const int tp_size = d_comm_->n_ranks(attn_tp_group_);
        const int tp_rank = d_comm_->rank(attn_tp_group_);
        token_nums.reserve(tp_size);
        counts.reserve(tp_size);

        const int local_token_num = local_token_nums[attn_dp_rank_];

        int q = local_token_num / tp_size;
        int r = local_token_num % tp_size;
        int offset{};
        for (int i = 0; i < tp_size; ++i) {
            token_nums.push_back(q + (i < r ? 1 : 0));
            counts.push_back(token_nums[i] * hidden_units_);
            if (i < tp_rank) {
                offset += token_nums[i];
            }
        }
        partial_hidden_states  = local_hidden_states.slice({offset, 0}, {token_nums[tp_rank], -1});
        partial_local_residual = local_residual.slice({offset, 0}, {token_nums[tp_rank], -1});

        const int max_tokens_per_dp_rank = *std::max_element(local_token_nums.begin(), local_token_nums.end());
        max_tokens_per_rank = max_tokens_per_dp_rank / tp_size + (max_tokens_per_dp_rank % tp_size > 0 ? 1 : 0);
    }

    return {global_hidden_states,
            local_hidden_states,
            local_residual,
            static_cast<int>(local_token_num),
            static_cast<int>(global_token_num),
            args.at("batch").data<BatchData*>()[0]->local_token_num,
            partial_hidden_states,
            partial_local_residual,
            std::move(token_nums),
            std::move(counts),
            max_tokens_per_rank};
}

void UnifiedDecoder::Run(BatchOp op, int phase, TensorMap& env)
{
    attn_layer_->Run(op, phase, env);
    if (linear_attn_layer_) {
        linear_attn_layer_->Run(op, phase, env);
    }
}

UnifiedDecoder::UnifiedDecoder(const ModelParam&     model,
                               const EngineParam&    engine,
                               const AttentionParam& attn,
                               const MoeParam&       moe,
                               const Context&        ctx,
                               int                   phases):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    ep_size_(engine.ep_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    rmsnorm_eps_(model.norm_eps),
    d_comm_(ctx.comm.d_comm),
    tune_layer_num_(model.tune_layer_num),
    is_warm_up_{*ctx.is_warm_up}
{
    // mlp_tp_size_ should be equal to ep_size_ when enable EP mode
    TM_CHECK_EQ(mlp_tp_size_, ep_size_ > 1 ? engine.ep_size : mlp_tp_size_);

    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
    }

    attn_layer_ =
        std::make_unique<UnifiedAttentionLayer>(model, attn, engine, attn_tp_size_, ctx, phases, (bool)moe_ffn_layer_);

    if (std::find(model.layer_types.begin(), model.layer_types.end(), 1) != model.layer_types.end()) {
        linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(model, attn, engine, attn_tp_size_, ctx, phases);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    fused_rmsnorm_layer_ = CreateFusedRMSNormLayer({ep_size_, hidden_units_, rmsnorm_eps_, attn_tp_group_, d_comm_});
}

void UnifiedDecoder::Forward(int phase, TensorMap& args, const std::vector<WeightType*>& weights)
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

    auto  layout               = CreateHiddenStateLayout(args);
    auto& global_hidden_states = layout.global_hidden_states;
    auto& local_hidden_states  = layout.local_hidden_states;
    auto& local_residual       = layout.local_residual;
    auto& local_token_num      = layout.local_token_num;
    auto& local_token_nums     = layout.local_token_nums;
    auto& global_token_num     = layout.global_token_num;
    auto& ffn_input            = (is_warm_up_ || ep_size_ == 1) ? global_hidden_states : layout.partial_hidden_states;

    const DataType dtype = local_residual.dtype();

    TM_DEBUG_TENSOR(local_residual, "res", 1);
    TM_DEBUG_TENSOR(weights.at(0)->self_attn_norm, "norm_weight", 2);

    const auto stream = core::Context::stream().handle();

    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->self_attn_norm, rmsnorm_eps_, stream);
    sync_check_cuda_error();

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);

    FusedRMSNormLayerForwardParam rmsnorm_fwd_param{global_hidden_states,
                                                    local_hidden_states,
                                                    local_residual,
                                                    local_token_nums,
                                                    (int)global_token_num,
                                                    layout.partial_hidden_states,
                                                    layout.partial_local_residual,
                                                    layout.token_nums,
                                                    layout.counts};

    // auto stack_alloc{core::Context::device_alloc().adapt<core::StackAllocatorImpl>()};
    // core::ContextGuard ctx{Allocator{stack_alloc}};

    for (int layer = 0; layer < layer_num_; ++layer) {

        // stack_alloc->iter();

        if (global_token_num == 0) {
            break;
        }

        if (is_warm_up_ && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention or linear-attention
        if (weights.at(layer)->linear_attn_weights) {
            linear_attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, weights.at(layer)->linear_attn_weights.get(), layer});
        }
        else {
            attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, weights.at(layer)->self_attn_weights.get(), layer});
        }

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        // For gated delta networks, we may need a different output.bias name or it doesn't have it.
        // We will just use `output.bias` from either layer.
        Tensor out_bias;
        if (weights.at(layer)->linear_attn_weights) {
            out_bias = weights.at(layer)->linear_attn_weights->out_proj.bias;
        }
        else {
            out_bias = weights.at(layer)->self_attn_weights->output.bias;
        }

        fused_rmsnorm_layer_->forward(rmsnorm_fwd_param,  //
                                      weights.at(layer)->ffn_norm,
                                      out_bias,
                                      FusedRMSNormLayerStage::kAttn);

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

        if (weights.at(layer)->moe_weights) {
            moe_fwd_param = MoeFfnLayer::ForwardParam{ffn_input,  //
                                                      ffn_input,
                                                      weights.at(layer)->moe_weights.get(),
                                                      ffn_layer_ ? 1.f : 0.f,
                                                      layout.max_tokens_per_rank,
                                                      layer};
            moe_ffn_layer_->Forward(*moe_fwd_param);
        }

        if (weights.at(layer)->ffn_weights && ffn_input.shape(0) > 0) {
            ffn_layer_->forward({ffn_input, ffn_input, weights.at(layer)->ffn_weights.get(), (int)layer});
        }

        if (moe_fwd_param) {
            moe_ffn_layer_->Combine(*moe_fwd_param);
        }

        TM_DEBUG_TENSOR(ffn_input, Concat("ffn_block", layer), 2);

        const bool last = layer == layer_num_ - 1;

        auto& scale_weight = !last ? weights.at(layer + 1)->self_attn_norm : args.at("output_norm_weight");

        fused_rmsnorm_layer_->forward(rmsnorm_fwd_param,  //
                                      scale_weight,
                                      {},
                                      FusedRMSNormLayerStage::kFfn);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);

        // if (layer == layer_num_ - 1) {
        //     args.at("batch").data<BatchData*>()[0]->Notify();
        // }
    }

    // Token indices selected for decoding
    const Buffer selected_pos = args.consume("selected_token_pos").buffer();
    // dbg(selected_pos);
    // When there are no prefill sequences, token selection is not needed
    const bool reuse_hidden_states = selected_pos.size() == local_token_num;

    const bool output_hidden_states = args.try_("output_hidden_states");

    Tensor hidden_states{local_hidden_states};

    if (d_comm_ && (output_hidden_states || reuse_hidden_states)) {
        // The full `hidden_states` buffer is needed for output but it's a ref into `symm_buf` atm.
        // Copy to residual buf so that `symm_buf` may be reused safely later
        Copy(hidden_states, local_residual);
        hidden_states = local_residual;
    }

    Tensor selected_states;
    if (reuse_hidden_states) {
        selected_states = hidden_states;
    }
    else {
        selected_states = {{selected_pos.size(), (int)hidden_units_}, dtype, kDEVICE};
        CollectHiddenStates(hidden_states, selected_pos, selected_states, stream);
    }
    args.produce("hidden_states", selected_states);

    // TM_DEBUG_TENSOR(selected_states.slice(0, selected_pos.size()), "out", 1);

    if (output_hidden_states) {
        args.produce("full_hidden_states", hidden_states);
    }
}

}  // namespace turbomind
