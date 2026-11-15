// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/fixed_chain_model.h"

#include <numeric>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/kernels/speculative_sequence_kernels.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/hidden_state_tap.h"

namespace turbomind {

FixedChainSpeculativeModel::FixedChainSpeculativeModel(const SpeculativeModelArgs& args):
    target_{args.target},
    comm_{args.ctx.comm},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    draft_{args.registry, args.param, args.ctx, args.draft_weights, args.phases},
    draft_hidden_{draft_.weights().hidden_units},
    policy_{args.param.spec_num_draft_tokens},
    fixed_chain_{args.ctx.comm, args.phases, args.param.max_batch_size},
    data_(args.phases)
{
    const EngineParam& param = args.param;

    Buffer_<int> identity_host{param.max_batch_size, kCPUpinned};
    draft_identity_token_pos_ = {param.max_batch_size, kDEVICE};
    std::iota(identity_host.data(), identity_host.data() + param.max_batch_size, 0);
    Copy(identity_host, draft_identity_token_pos_);
    core::Context::stream().Sync();

    const DataType draft_type = draft_.weights().data_type;

    for (CommonData& data : data_) {
        data.draft_input_ids          = {param.max_forward_token_num, kDEVICE};
        data.draft_selected_token_pos = {param.max_batch_size, kDEVICE};
        data.draft_candidate_active   = {param.max_batch_size, kDEVICE};
        data.draft_proposal_ids       = {param.max_batch_size, kDEVICE};

        data.draft_selected_normalized_hidden = {
            {param.max_batch_size, draft_hidden_}, draft_type, kDEVICE};
    }
}

FixedChainSpeculativeModel::~FixedChainSpeculativeModel() = default;

const SpeculativePolicy& FixedChainSpeculativeModel::policy() const
{
    return policy_;
}

HiddenStateTap* FixedChainSpeculativeModel::Tap(int)
{
    return TapSource();
}

void FixedChainSpeculativeModel::Setup(int phase, TensorMap& env)
{
    fixed_chain_.Setup(phase,
                       env.at("requests").buffer(),
                       *env.at("copy").data<core::BatchCopy*>()[0]);
    draft_.Run(BatchOp::kSetup, phase, env);
}

void FixedChainSpeculativeModel::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        draft_.Run(op, phase, env);
        return;
    }
    if (op == BatchOp::kSetup) {
        Setup(phase, env);
        return;
    }
    if (op == BatchOp::kPrepare) {
        TensorMap draft_env       = env;
        draft_env.at("finished")  = env.at("finished_on_entry");
        draft_env.at("q_offsets") = env.at("q_offsets");
        draft_env.at("k_offsets") = env.at("k_offsets");
        draft_.Run(BatchOp::kPrepare, phase, draft_env);
    }
}

void FixedChainSpeculativeModel::RunDraft(int phase, const DraftContext& ctx, TensorMap& env)
{
    CommonData&          data  = data_[phase];
    FixedChainPhaseData& chain = fixed_chain_.phase_data(phase);

    const int target_query_count = chain.refresh_decode.query_count + chain.refresh_prefill.query_count;
    const int candidate_count    = chain.draft_extension_query_count;
    const int k                  = policy_.max_proposals();
    const cudaStream_t stream    = core::Context::stream().handle();

    AttentionForwardMetadata refresh_metadata{};
    refresh_metadata.decode    = chain.refresh_decode;
    refresh_metadata.prefill   = chain.refresh_prefill;
    refresh_metadata.q_offsets = ctx.target_q_offsets;
    refresh_metadata.k_offsets = ctx.target_k_offsets;

    AttentionForwardMetadata extension_metadata{};
    extension_metadata.decode    = chain.extension_decode;
    extension_metadata.prefill   = {};
    extension_metadata.q_offsets = chain.draft_extension_q_offsets.slice(0, ctx.batch_size + 1);
    extension_metadata.k_offsets = chain.draft_extension_k_offsets.slice(0, ctx.batch_size + 1);

    Tensor extension_local_token_nums{chain.draft_extension_local_token_nums.data(),
                                      Layout{{static_cast<ssize_t>(chain.draft_extension_local_token_nums.size())}},
                                      kCPU};

    const bool run_extensions = k > 1 && chain.draft_extension_global_query_count > 0;

    Tensor carry = InitialCarry(phase, stream);

    invokeBuildDraftRefreshInputs(data.draft_input_ids.data(),
                                  data.draft_selected_token_pos.data(),
                                  data.draft_candidate_active.data(),
                                  ctx.request_token_ids_ptrs,
                                  ctx.target_q_offsets.data(),
                                  ctx.target_k_offsets.data(),
                                  chain.draft_extension_q_offsets.data(),
                                  ctx.accept_len.data(),
                                  chain.limit_to_accept_len.data(),
                                  env.at("finished").data<bool>(),
                                  target_query_count,
                                  ctx.batch_size,
                                  candidate_count,
                                  stream);

    Tensor embeddings = Embed(phase,
                              data.draft_input_ids.slice(0, target_query_count),
                              target_query_count,
                              ctx,
                              env,
                              EmbedStage::kRefresh);

    TensorMap draft_env       = env;
    draft_env.at("finished")  = ctx.finished_on_entry;
    draft_env.at("q_offsets") = ctx.target_q_offsets;
    draft_env.at("k_offsets") = ctx.target_k_offsets;

    CombineResult combined = Combine(phase, std::move(embeddings), std::move(carry), target_query_count, env);

    LanguageModel::DecoderInputs refresh_in;
    refresh_in.residual               = std::move(combined.residual);
    refresh_in.attention_input        = std::move(combined.attention_input);
    refresh_in.selected_token_pos     = data.draft_selected_token_pos.slice(0, candidate_count);
    refresh_in.selected_hidden_buffer = data.draft_selected_normalized_hidden.slice(
        {0, 0}, {candidate_count, draft_hidden_});
    refresh_in.attention_metadata = &refresh_metadata;

    LanguageModel::DecoderOutputs out = draft_.RunDecoder(phase, refresh_in, draft_env);

    if (candidate_count > 0) {
        const auto& hw     = HeadModel().weights();
        Tensor      logits = ctx.head_storage.slice({0, 0}, {candidate_count, hw.vocab_size_padded});
        logits            = HeadModel().Logits(out.selected_hidden, logits, draft_env);
        invokeDraftArgmaxAndStoreToken(logits,
                                       data.draft_proposal_ids.data(),
                                       ctx.request_token_ids_ptrs,
                                       chain.draft_extension_q_offsets.data(),
                                       data.draft_candidate_active.data(),
                                       ctx.sequence_length.data(),
                                       ctx.accept_len.data(),
                                       ctx.batch_size,
                                       candidate_count,
                                       0,
                                       hw.vocab_size,
                                       stream);
    }

    if (!run_extensions) {
        return;
    }

    TensorMap extension_env       = env;
    extension_env.at("finished")  = ctx.finished_on_entry;
    extension_env.at("q_offsets") = chain.draft_extension_q_offsets.slice(0, ctx.batch_size + 1);
    extension_env.at("k_offsets") = chain.draft_extension_k_offsets.slice(0, ctx.batch_size + 1);
    extension_env.produce("decoder_local_token_nums", extension_local_token_nums);

    for (int step = 1; step < k; ++step) {
        const int i = step - 1;

        carry       = NextCarry(out, i, phase, env, stream);
        embeddings  = Embed(phase,
                            data.draft_proposal_ids.slice(0, candidate_count),
                            candidate_count,
                            ctx,
                            extension_env,
                            EmbedStage::kExtension);
        combined    = Combine(phase, std::move(embeddings), std::move(carry), candidate_count, extension_env);

        invokeBuildDraftExtensionKeyOffsets(chain.draft_extension_k_offsets.data(),
                                            chain.draft_extension_q_offsets.data(),
                                            ctx.sequence_length.data(),
                                            ctx.accept_len.data(),
                                            ctx.batch_size,
                                            i,
                                            stream);

        LanguageModel::DecoderInputs extension_in;
        extension_in.residual               = std::move(combined.residual);
        extension_in.attention_input        = std::move(combined.attention_input);
        extension_in.selected_token_pos     = draft_identity_token_pos_.slice(0, candidate_count);
        extension_in.selected_hidden_buffer = data.draft_selected_normalized_hidden.slice(
            {0, 0}, {candidate_count, draft_hidden_});
        extension_in.attention_metadata = &extension_metadata;

        out = draft_.RunDecoder(phase, extension_in, extension_env);

        if (candidate_count > 0) {
            const auto& hw     = HeadModel().weights();
            Tensor      logits = ctx.head_storage.slice({0, 0}, {candidate_count, hw.vocab_size_padded});
            logits            = HeadModel().Logits(out.selected_hidden, logits, extension_env);
            invokeDraftArgmaxAndStoreToken(logits,
                                           data.draft_proposal_ids.data(),
                                           ctx.request_token_ids_ptrs,
                                           chain.draft_extension_q_offsets.data(),
                                           data.draft_candidate_active.data(),
                                           ctx.sequence_length.data(),
                                           ctx.accept_len.data(),
                                           ctx.batch_size,
                                           candidate_count,
                                           i + 1,
                                           hw.vocab_size,
                                           stream);
        }
    }
}

}  // namespace turbomind
