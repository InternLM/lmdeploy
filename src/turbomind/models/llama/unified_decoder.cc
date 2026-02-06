

#include <numeric>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

struct FusedRMSNormLayer {

    struct Param {
        Tensor                  global_hidden_states;
        Tensor                  local_hidden_states;
        Tensor                  local_residual;
        int                     attn_tp_size;
        int                     attn_tp_rank;
        int                     attn_dp_rank;
        int                     ep_size;
        size_t                  hidden_units;
        float                   rmsnorm_eps;
        int                     attn_tp_group;
        const std::vector<int>& local_token_nums;
        int                     global_token_num;
        comm::DeviceCommImpl*   d_comm;
        int                     is_warm_up;
    };

    enum class Stage : int
    {
        kAttn,
        kFfn,
    };

    enum class FFnMode : int
    {
        kTP,
        kEP,
    };

    FusedRMSNormLayer(Param param): mode_(param.ep_size > 1 ? FFnMode::kEP : FFnMode::kTP), param_(param)
    {
        if (mode_ == FFnMode::kEP) {
            token_nums_.reserve(param_.attn_tp_size);
            counts_.reserve(param_.attn_tp_size);

            const int local_token_num = param_.local_token_nums[param_.attn_dp_rank];

            int q = local_token_num / param_.attn_tp_size;
            int r = local_token_num % param_.attn_tp_size;
            int offset{};
            for (int i = 0; i < param_.attn_tp_size; ++i) {
                token_nums_.push_back(q + (i < r ? 1 : 0));
                counts_.push_back(token_nums_[i] * param_.hidden_units);
                if (i < param_.attn_tp_rank) {
                    offset += token_nums_[i];
                }
            }

            partial_hidden_states_ =
                param_.local_hidden_states.slice({offset, 0}, {token_nums_[param_.attn_tp_rank], -1});
            partial_local_residual_ = param_.local_residual.slice({offset, 0}, {token_nums_[param_.attn_tp_rank], -1});
        }
    }

    Tensor FfnInput()
    {
        auto ffn_input = param_.global_hidden_states;
        if (!param_.is_warm_up && mode_ == FFnMode::kEP) {
            ffn_input = partial_hidden_states_;
        }
        return ffn_input;
    }

    void Forward(const Tensor& weight, const Tensor& bias, Stage stage)
    {
        if (mode_ == FFnMode::kTP) {
            const int group0 = stage == Stage::kAttn ? param_.attn_tp_group : 0;
            const int group1 = stage == Stage::kAttn ? 0 : param_.attn_tp_group;
            return AllReduceResidualRMSNorm(weight, bias, group0, group1);
        }
        else {
            return ResidualRMSNormEP(weight, bias, stage);
        }
    }

    void AllReduceResidualRMSNorm(const Tensor& weight, const Tensor& bias, int group0, int group1)
    {
        const auto dtype = param_.global_hidden_states.dtype();

        const auto stream = core::Context::stream().handle();

        if (0) {}
        else if (group0 || group1) {
            param_.d_comm->AllreduceResidualBiasRMSnormEx(param_.global_hidden_states.raw_data(),
                                                          param_.local_residual.data_or((void*)nullptr),
                                                          bias.data_or((void*)nullptr),
                                                          weight.raw_data(),
                                                          param_.rmsnorm_eps,
                                                          param_.hidden_units,
                                                          dtype,
                                                          group0,
                                                          group1,
                                                          param_.local_token_nums.data(),
                                                          stream);
            sync_check_cuda_error();
        }
        else if (param_.d_comm) {
            param_.d_comm->AllreduceResidualBiasRMSnorm(param_.global_hidden_states.raw_data(),
                                                        param_.local_residual.data_or((void*)nullptr),
                                                        bias.data_or((void*)nullptr),
                                                        weight.raw_data(),
                                                        param_.rmsnorm_eps,
                                                        param_.hidden_units,
                                                        param_.global_token_num,
                                                        dtype,
                                                        0,
                                                        stream);
            sync_check_cuda_error();
        }
        else {
            invokeResidualBiasRMSNorm(param_.global_hidden_states.raw_data(),
                                      param_.local_residual.data_or((void*)nullptr),
                                      weight.raw_data(),
                                      bias.data_or((void*)nullptr),
                                      dtype,
                                      param_.hidden_units,
                                      param_.global_token_num,
                                      param_.rmsnorm_eps,
                                      stream);
            sync_check_cuda_error();
        }
    }

    void ResidualRMSNormEP(const Tensor& weight, const Tensor& bias, Stage stage)
    {
        const auto stream = core::Context::stream().handle();

        if (stage == Stage::kAttn) {
            param_.d_comm->ReduceScatterV(param_.local_hidden_states.data_or((void*)nullptr),  //
                                          partial_hidden_states_.data_or((void*)nullptr),
                                          counts_.data(),
                                          param_.local_hidden_states.dtype(),
                                          param_.attn_tp_group,
                                          stream);
            sync_check_cuda_error();
        }

        invokeResidualBiasRMSNorm(partial_hidden_states_.data_or((void*)nullptr),
                                  partial_local_residual_.data_or((void*)nullptr),
                                  weight.raw_data(),
                                  bias.data_or((void*)nullptr),
                                  partial_hidden_states_.dtype(),
                                  param_.hidden_units,
                                  token_nums_[param_.attn_tp_rank],
                                  param_.rmsnorm_eps,
                                  stream);
        sync_check_cuda_error();

        if (stage == Stage::kFfn) {
            param_.d_comm->AllGatherV(partial_hidden_states_.data_or((void*)nullptr),
                                      param_.local_hidden_states.data_or((void*)nullptr),
                                      counts_.data(),
                                      param_.local_hidden_states.dtype(),
                                      param_.attn_tp_group,
                                      stream);
            sync_check_cuda_error();
        }
    }

    FFnMode             mode_;
    Param               param_;
    std::vector<int>    token_nums_;
    std::vector<size_t> counts_;
    Tensor              partial_hidden_states_;
    Tensor              partial_local_residual_;
};

void UnifiedDecoder::Run(BatchOp op, int phase, TensorMap& env)
{
    attn_layer_->Run(op, phase, env);
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
    attn_tp_rank_(engine.attn_tp_rank),
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
    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
    }

    attn_layer_ =
        std::make_unique<UnifiedAttentionLayer>(model, attn, engine, attn_tp_size_, ctx, phases, (bool)moe_ffn_layer_);

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }
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

    constexpr auto device = kDEVICE;

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

    TM_DEBUG_TENSOR(local_residual, "res", 1);
    TM_DEBUG_TENSOR(weights.at(0)->self_attn_norm, "norm_weight", 2);

    const auto stream = core::Context::stream().handle();

    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->self_attn_norm, rmsnorm_eps_, stream);
    sync_check_cuda_error();

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);

    FusedRMSNormLayer fused_rms_norm_layer({global_hidden_states,
                                            local_hidden_states,
                                            local_residual,
                                            attn_tp_size_,
                                            attn_tp_rank_,
                                            attn_dp_rank_,
                                            ep_size_,
                                            hidden_units_,
                                            rmsnorm_eps_,
                                            attn_tp_group_,
                                            local_token_nums,
                                            (int)global_token_num,
                                            d_comm_,
                                            is_warm_up_});

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
        /// self-attention
        attn_layer_->Forward(
            {phase, local_hidden_states, local_hidden_states, weights.at(layer)->self_attn_weights.get(), layer});

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        fused_rms_norm_layer.Forward(weights.at(layer)->ffn_norm,
                                     weights.at(layer)->self_attn_weights->output.bias,
                                     FusedRMSNormLayer::Stage::kAttn);

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

        auto ffn_input = fused_rms_norm_layer.FfnInput();

        if (weights.at(layer)->moe_weights) {
            moe_fwd_param = MoeFfnLayer::ForwardParam{ffn_input,  //
                                                      ffn_input,
                                                      weights.at(layer)->moe_weights.get(),
                                                      ffn_layer_ ? 1.f : 0.f,
                                                      layer};
            moe_ffn_layer_->Forward(*moe_fwd_param);
        }

        if (weights.at(layer)->ffn_weights && ffn_input.shape(0) > 0) {
            ffn_layer_->forward({ffn_input, ffn_input, weights.at(layer)->ffn_weights.get(), (int)layer});
        }

        if (moe_fwd_param) {
            moe_ffn_layer_->Combine(*moe_fwd_param);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);

        const bool last = layer == layer_num_ - 1;

        auto& scale_weight = !last ? weights.at(layer + 1)->self_attn_norm : args.at("output_norm_weight");

        fused_rms_norm_layer.Forward(scale_weight, {}, FusedRMSNormLayer::Stage::kFfn);
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
