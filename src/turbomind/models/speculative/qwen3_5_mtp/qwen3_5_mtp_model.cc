// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/qwen3_5_mtp/qwen3_5_mtp_model.h"

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/qwen3_5_mtp/qwen3_5_mtp_weight.h"
#include "src/turbomind/models/speculative/qwen3_5_mtp/target_final_hidden.h"
#include "src/turbomind/models/speculative/registry.h"

namespace turbomind {

Qwen35MtpModel::Qwen35MtpModel(const SpeculativeModelArgs& args):
    FixedChainSpeculativeModel(args),
    spec_weights_{*args.draft_weights.get<Qwen35MtpWeight>("spec")},
    linear_{*args.ctx.linear},
    hidden_units_{target_.weights().hidden_units},
    model_tp_rank_{args.param.model_tp_rank},
    model_tp_size_{args.param.attn_cp_size * args.param.attn_tp_size},
    final_hidden_{std::make_unique<TargetFinalHidden>(args.param,
                                                      args.ctx,
                                                      args.phases,
                                                      target_.weights().num_layer,
                                                      hidden_units_,
                                                      target_.weights().data_type)},
    data_(args.phases)
{
    const EngineParam& param = args.param;

    Allocator projected_allocator = core::Context::device_alloc();
    if (model_tp_size_ > 1) {
        projected_allocator = GetSymmAllocator(comm_.d_comm);
    }

    for (Data& data : data_) {
        data.normalized_concat = {
            {param.max_forward_token_num, 2 * hidden_units_}, target_.weights().data_type, kDEVICE};
        data.projected_full = {
            {param.max_forward_token_num, hidden_units_}, target_.weights().data_type, projected_allocator};
    }
}

Qwen35MtpModel::~Qwen35MtpModel() = default;

HiddenStateTap* Qwen35MtpModel::TapSource()
{
    return final_hidden_.get();
}

Tensor Qwen35MtpModel::InitialCarry(int phase, cudaStream_t stream)
{
    return final_hidden_->Gather(phase, stream);
}

Tensor Qwen35MtpModel::Embed(int                phase,
                             const Buffer_<int>& ids,
                             int                rows,
                             const DraftContext& ctx,
                             TensorMap&         env,
                             EmbedStage         stage)
{
    Tensor storage = use_ag2d_ ? Tensor{env.at("symm_buf").buffer().view(target_.weights().data_type),
                                        {rows, hidden_units_}}
                               : ctx.embedding_storage.slice({0, 0}, {rows, hidden_units_});
    Tensor embeddings = target_.Embed(ids, storage, env);

    if (stage == EmbedStage::kRefresh) {
        auto& copy = *env.at("copy").data<core::BatchCopy*>()[0];
        ctx.input_processor->PatchSuccessorEmbedding(phase, embeddings, copy, env);
        copy.Run();
    }

    return embeddings;
}

FixedChainSpeculativeModel::CombineResult Qwen35MtpModel::Combine(int      phase,
                                                                  Tensor   embeddings,
                                                                  Tensor   carry,
                                                                  int      rows,
                                                                  TensorMap& env)
{
    Data&              data   = data_[phase];
    const cudaStream_t stream = core::Context::stream().handle();

    Tensor concat = data.normalized_concat.slice({0, 0}, {rows, 2 * hidden_units_});
    invokeRMSNormConcat(concat,
                        embeddings,
                        spec_weights_.pre_fc_norm_embedding->weight,
                        spec_weights_.pre_fc_norm_embedding->norm_eps_,
                        spec_weights_.pre_fc_norm_embedding->zero_centered_,
                        carry,
                        spec_weights_.pre_fc_norm_hidden->weight,
                        spec_weights_.pre_fc_norm_hidden->norm_eps_,
                        spec_weights_.pre_fc_norm_hidden->zero_centered_,
                        stream);

    CombineResult result;
    result.residual = ProjectAndGatherFc(data, concat, rows, stream, env);
    return result;
}

Tensor Qwen35MtpModel::NextCarry(const LanguageModel::DecoderOutputs& out,
                                 int,
                                 int,
                                 TensorMap&,
                                 cudaStream_t)
{
    return out.selected_hidden;
}

LanguageModel& Qwen35MtpModel::HeadModel()
{
    return target_;
}

Tensor Qwen35MtpModel::ProjectAndGatherFc(Data&            data,
                                          const Tensor&    normalized_concat,
                                          int              rows,
                                          cudaStream_t     stream,
                                          const TensorMap& env)
{
    const int local_hidden = hidden_units_ / model_tp_size_;
    if (rows == 0) {
        return data.projected_full.slice({0, 0}, {0, hidden_units_});
    }

    if (model_tp_size_ == 1) {
        Tensor full = data.projected_full.slice({0, 0}, {rows, hidden_units_});
        linear_.Forward(normalized_concat, *spec_weights_.fc, full);
        return full;
    }

    if (use_ag2d_) {
        Tensor gathered = data.projected_full.slice({0, 0}, {rows, hidden_units_})
                              .view({rows, model_tp_size_, local_hidden});
        Tensor local = gathered.slice({0, model_tp_rank_, 0}, {rows, 1, local_hidden}).squeeze(1);
        linear_.Forward(normalized_concat, *spec_weights_.fc, local);
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  gathered.raw_data(),
                                  hidden_units_,
                                  local_hidden,
                                  local_hidden,
                                  rows,
                                  local.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  stream);
        return gathered.view({rows, hidden_units_});
    }

    Tensor gathered{env.at("symm_buf").buffer().view(normalized_concat.dtype()),
                    {model_tp_size_, rows, local_hidden}};
    Tensor local = gathered.slice({model_tp_rank_, 0, 0}, {1, rows, local_hidden}).squeeze(0);
    linear_.Forward(normalized_concat, *spec_weights_.fc, local);
    comm_.d_comm->AllGather(local.raw_data(),
                            gathered.raw_data(),
                            local.size(),
                            local.dtype(),
                            comm_.d_tp_group,
                            stream);

    Tensor full = data.projected_full.slice({0, 0}, {rows, hidden_units_});
    invokeTransposeAxis01(static_cast<uint16_t*>(full.raw_data()),
                          static_cast<uint16_t*>(gathered.raw_data()),
                          model_tp_size_,
                          rows,
                          local_hidden,
                          stream);
    return full;
}

TM_REGISTER_SPECULATIVE_MODEL("mtp", Qwen35MtpModel);

}  // namespace turbomind
