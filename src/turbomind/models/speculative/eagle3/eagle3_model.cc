// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/eagle3/eagle3_model.h"

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/token_ownership.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/draft_carry_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/eagle3/eagle3_weight.h"
#include "src/turbomind/models/speculative/eagle3/target_hidden_projection.h"
#include "src/turbomind/models/speculative/registry.h"

namespace turbomind {

Eagle3Model::Eagle3Model(const SpeculativeModelArgs& args):
    FixedChainSpeculativeModel(args),
    spec_weights_{*TM_CHECK_NOTNULL(args.draft_weights.get<Eagle3Weight>("spec"))},
    data_(args.phases)
{
    const auto& p  = args.param;
    const auto& dw = draft_.weights();

    for (Data& d : data_) {
        d.draft_attention_input = Tensor{{p.max_forward_token_num, 2 * dw.hidden_units}, dw.data_type, kDEVICE};

        if (comm_.d_comm) {
            auto symmetric_allocator = GetSymmAllocator(comm_.d_comm);
            d.draft_carry = Tensor{{p.max_batch_size, dw.hidden_units}, dw.data_type, symmetric_allocator};
        }
        else {
            d.draft_carry = Tensor{{p.max_batch_size, dw.hidden_units}, dw.data_type, kDEVICE};
        }
    }

    projection_ = std::make_unique<TargetHiddenProjection>(p,
                                                           args.ctx,
                                                           args.phases,
                                                           target_.weights().num_layer,
                                                           target_.weights().hidden_units,
                                                           target_.weights().data_type,
                                                           p.spec_tap_layer_ids,
                                                           *spec_weights_.target_hidden_proj);
}

Eagle3Model::~Eagle3Model() = default;

HiddenStateTap* Eagle3Model::TapSource()
{
    return projection_.get();
}

Tensor Eagle3Model::InitialCarry(int phase, cudaStream_t stream)
{
    return projection_->ProjectAndGather(phase, stream);
}

Tensor Eagle3Model::Embed(int, const Buffer_<int>& ids, int rows, const DraftContext& ctx, TensorMap& env, EmbedStage)
{
    const auto& dw = draft_.weights();

    Tensor storage = use_ag2d_ ? Tensor{env.at("symm_buf").buffer().view(dw.data_type), {rows, dw.hidden_units}}
                               : ctx.embedding_storage.slice({0, 0}, {rows, dw.hidden_units});
    return draft_.Embed(ids, storage, env);
}

FixedChainSpeculativeModel::CombineResult Eagle3Model::Combine(int phase, Tensor embeddings, Tensor carry, int rows, TensorMap&)
{
    Data&              data   = data_[phase];
    const auto&        dw     = draft_.weights();
    const cudaStream_t stream = core::Context::stream().handle();

    DecoderLayerWeight* const draft_layer       = dw.layer(0);
    const NormWeight* const   draft_hidden_norm = TM_CHECK_NOTNULL(spec_weights_.hidden_norm(0));

    Tensor attention_input = data.draft_attention_input.slice({0, 0}, {rows, 2 * dw.hidden_units});
    invokeRMSNormConcat(attention_input,
                        embeddings,
                        draft_layer->attention_norm->weight,
                        draft_layer->attention_norm->norm_eps_,
                        draft_layer->attention_norm->zero_centered_,
                        carry,
                        draft_hidden_norm->weight,
                        draft_hidden_norm->norm_eps_,
                        draft_hidden_norm->zero_centered_,
                        stream);

    CombineResult result;
    result.residual        = std::move(carry);
    result.attention_input = std::move(attention_input);
    return result;
}

Tensor Eagle3Model::NextCarry(const LanguageModel::DecoderOutputs& out,
                              int                                 step,
                              int                                 phase,
                              TensorMap&                          env,
                              cudaStream_t                        stream)
{
    Data&                data  = data_[phase];
    FixedChainPhaseData& chain = fixed_chain_.phase_data(phase);
    CommonData&          common_data = common(phase);
    const auto&          dw     = draft_.weights();

    const int candidate_count = chain.draft_extension_query_count;

    Tensor carry = data.draft_carry.slice({0, 0}, {candidate_count, dw.hidden_units});
    if (candidate_count == 0) {
        return carry;
    }

    const auto& batch_local_token_nums = env.at("batch").data<BatchData*>()[0]->local_token_num;

    const int global_rank = comm_.d_comm ? comm_.d_comm->rank(0) : 0;
    const int tp0_size    = comm_.d_comm ? comm_.d_comm->n_ranks(0) : 1;
    const int tp1_size    = comm_.d_comm ? comm_.d_comm->n_ranks(comm_.d_tp_group) : 1;

    const comm::OwnedTokenRows owned = step == 0 ?
        comm::ComputeTokenOwnership(global_rank, tp0_size, tp1_size, batch_local_token_nums.data()) :
        comm::ComputeTokenOwnership(global_rank, tp0_size, tp1_size, chain.draft_extension_local_token_nums.data());

    const Buffer_<int> selected_rows = step == 0 ? common_data.draft_selected_token_pos.slice(0, candidate_count) :
                                                   draft_identity_token_pos_.slice(0, candidate_count);

    invokeSelectDraftCarry(out.pre_final_residual.raw_data(),
                           selected_rows.data(),
                           common_data.draft_candidate_active.data(),
                           carry.raw_data(),
                           static_cast<int>(out.pre_final_residual.shape(0)),
                           candidate_count,
                           dw.hidden_units,
                           byte_size(dw.data_type) * 8,
                           owned.local_begin(),
                           owned.local_end(),
                           stream);

    if (tp1_size > 1) {
        comm_.d_comm->AllReduceSum(carry.raw_data(),
                                   carry.raw_data(),
                                   carry.size(),
                                   carry.dtype(),
                                   comm_.d_tp_group,
                                   stream);
    }

    return carry;
}

LanguageModel& Eagle3Model::HeadModel()
{
    return draft_;
}

TM_REGISTER_SPECULATIVE_MODEL("eagle3", Eagle3Model);

}  // namespace turbomind
