
#include "src/turbomind/models/language_model.h"

#include <algorithm>
#include <memory>
#include <numeric>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nvtx_utils.h"

// #include "dbg.h"

namespace turbomind {

struct LanguageModel::Impl {
    const Communicators& comm_;
    const ModelWeight&   weights_;
    LlamaLinear&         linear_;

    const int  tp_size_;
    const int  tp_rank_;
    const bool use_ag2d_;

    // Max chunk size for compute / output full logits
    int max_logits_len_ = 0;

    std::unique_ptr<UnifiedDecoder> unified_decoder_;

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        unified_decoder_->Run(op, phase, env);
    }

    Impl(CacheRegistry&     registry,
         const EngineParam& engine,
         const Context&     ctx,
         const ModelWeight& weights,
         int                phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids,
                           const Tensor&       embedding_table,
                           Buffer              model_tp_gather_buffer,
                           Tensor              embeddings);
    Tensor PostEmbedding(const Tensor&       features,
                         const LinearWeight& output_weight,
                         Buffer              model_tp_gather_buffer,
                         Tensor              logits);

    bool has_embedding() const;
    bool has_head() const;
    Tensor Embed(const Buffer_<int>& input_ids, Tensor out, const TensorMap& env);
    LanguageModel::DecoderOutputs RunDecoder(int phase, const LanguageModel::DecoderInputs& in, TensorMap& env);
    Tensor Logits(const Tensor& hidden, Tensor out, const TensorMap& env);
    const ModelWeight& weights() const;
    int max_logits_len(const TensorMap& env) const;
    bool logits_use_workspace() const;
    void CommitAcceptedState(int phase, const Buffer_<int>& accept_len);
    size_t SpeculativeStateJournalBytes(int request_count, int verification_positions) const;

};

LanguageModel::Impl::Impl(CacheRegistry&     registry,
                          const EngineParam& engine,
                          const Context&     ctx,
                          const ModelWeight& weights,
                          int                phases):
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{comm_.h_tp_group->n_ranks()},
    tp_rank_{comm_.h_tp_group->rank()},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)}
{
    const int max_logits_rows =
        engine.max_batch_size * (engine.spec_method.empty() ? 1 : engine.spec_num_draft_tokens + 1);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(registry, engine, ctx, phases, weights_);

    if (has_head()) {
        TM_CHECK_GT(weights_.vocab_size_padded, 0) << "a model without a head has no logits length";
        max_logits_len_ = std::max<int>(
            core::ssize_t(engine.max_forward_token_num) * weights_.hidden_units / weights_.vocab_size_padded,
            max_logits_rows);
    }
}

Tensor LanguageModel::Impl::LookupEmbedding(const Buffer_<int>& input_ids,
                                            const Tensor&       embedding_table,
                                            Buffer              model_tp_gather_buffer,
                                            Tensor              embeddings)
{
    TM_FUNCTION_SCOPE();

    const int          token_count         = input_ids.size();
    const core::Device expected_device     = core::Context::device_alloc()->device();
    const bool         embeddings_supplied = embeddings.ndim() != 0;
    const int          local_hidden_size   = static_cast<int>(embedding_table.shape(1));
    const int          hidden_size         = local_hidden_size * tp_size_;
    const DataType     data_type           = embedding_table.dtype();

    if (token_count == 0) {
        if (embeddings_supplied) {
            return embeddings;
        }
        return Tensor{static_cast<void*>(nullptr), Layout{{0, hidden_size}}, data_type, expected_device};
    }

    if (!embeddings_supplied) {
        embeddings = Tensor{{token_count, hidden_size}, data_type, expected_device};
    }

    const cudaStream_t stream = core::Context::stream().handle();

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(embeddings, input_ids, embedding_table, stream);
    }
    else if (use_ag2d_) {
        Tensor gathered{model_tp_gather_buffer.view(data_type), {token_count, tp_size_, local_hidden_size}};
        Tensor local = gathered.slice({0, tp_rank_, 0}, {token_count, 1, local_hidden_size}).squeeze(1);
        Tensor flat  = gathered.view({token_count, hidden_size});

        const bool exact_alias = embeddings.raw_data() == flat.raw_data();

        invokeEmbeddingLookup(local, input_ids, embedding_table, stream);
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  gathered.raw_data(),
                                  hidden_size,
                                  local_hidden_size,
                                  local_hidden_size,
                                  token_count,
                                  gathered.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  stream);

        if (!exact_alias) {
            Copy(flat, embeddings);
        }
    }
    else {
        Tensor gathered{model_tp_gather_buffer.view(data_type), {tp_size_, token_count, local_hidden_size}};
        Tensor local = gathered.slice({tp_rank_, 0, 0}, {1, token_count, local_hidden_size}).squeeze(0);

        invokeEmbeddingLookup(local, input_ids, embedding_table, stream);
        comm_.d_comm->AllGather(
            local.raw_data(), gathered.raw_data(), local.size(), data_type, comm_.d_tp_group, stream);
        invokeInPlaceTranspose102(static_cast<uint16_t*>(embeddings.raw_data()),
                                  static_cast<uint16_t*>(gathered.raw_data()),
                                  tp_size_,
                                  token_count,
                                  local_hidden_size,
                                  false,
                                  stream);
    }

    return embeddings;
}

Tensor LanguageModel::Impl::PostEmbedding(const Tensor&       features,
                                          const LinearWeight& output_weight,
                                          Buffer              model_tp_gather_buffer,
                                          Tensor              logits)
{
    TM_FUNCTION_SCOPE();
    NvtxScope scope("postDecodeEmbedding");

    const core::Device expected_device = core::Context::device_alloc()->device();
    const bool         logits_supplied = logits.ndim() != 0;

    const int      batch_size        = static_cast<int>(features.shape(0));
    const int      local_vocab_size  = output_weight.output_dim;
    const int      padded_vocab_size = local_vocab_size * tp_size_;
    const DataType output_dtype      = output_weight.output_dtype();

    if (batch_size == 0) {
        if (logits_supplied) {
            return logits;
        }
        return Tensor{static_cast<void*>(nullptr), Layout{{0, padded_vocab_size}}, output_dtype, expected_device};
    }

    if (!logits_supplied) {
        if (tp_size_ > 1 && use_ag2d_) {
            logits = Tensor{model_tp_gather_buffer.view(output_dtype), {batch_size, padded_vocab_size}};
        }
        else {
            logits = Tensor{{batch_size, padded_vocab_size}, output_dtype, expected_device};
        }
    }

    const cudaStream_t stream = core::Context::stream().handle();

    if (tp_size_ == 1) {
        TM_SCOPE_CALL(linear_.Forward(features, output_weight, logits));
    }
    else if (use_ag2d_) {
        Tensor gathered = logits.view({batch_size, tp_size_, local_vocab_size});
        Tensor local    = gathered.slice({0, tp_rank_, 0}, {batch_size, 1, local_vocab_size}).squeeze(1);

        TM_SCOPE_CALL(linear_.Forward(features, output_weight, local));
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  gathered.raw_data(),
                                  padded_vocab_size,
                                  local_vocab_size,
                                  local_vocab_size,
                                  batch_size,
                                  gathered.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  stream);
    }
    else {
        Tensor gathered{model_tp_gather_buffer.view(output_dtype), {tp_size_, batch_size, local_vocab_size}};
        Tensor local = gathered.slice({tp_rank_, 0, 0}, {1, batch_size, local_vocab_size}).squeeze(0);

        TM_SCOPE_CALL(linear_.Forward(features, output_weight, local));
        comm_.d_comm->AllGather(
            local.raw_data(), gathered.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, stream);
        invokeTransposeAxis01(static_cast<uint16_t*>(logits.raw_data()),
                              static_cast<uint16_t*>(gathered.raw_data()),
                              tp_size_,
                              batch_size,
                              local_vocab_size,
                              stream);
    }

    return logits;
}

bool LanguageModel::Impl::has_embedding() const
{
    return !weights_.decoder_only;
}

bool LanguageModel::Impl::has_head() const
{
    return !weights_.decoder_only;
}

Tensor LanguageModel::Impl::Embed(const Buffer_<int>& input_ids, Tensor out, const TensorMap& env)
{
    Buffer symm_buf;
    if (comm_.d_comm) {
        symm_buf = env.at("symm_buf").buffer();
    }
    return LookupEmbedding(input_ids, weights_.tok_embeddings, symm_buf, std::move(out));
}

LanguageModel::DecoderOutputs
LanguageModel::Impl::RunDecoder(int phase, const LanguageModel::DecoderInputs& in, TensorMap& env)
{
    env.try_consume("hidden_states");
    env.try_consume("pre_final_residual");
    env.try_consume("full_hidden_states");

    env.insert_or_assign("residual", in.residual);

    if (in.attention_input) {
        env.insert_or_assign("attention_input", in.attention_input);
    }
    HiddenStateTap* tap = in.taps;
    if (tap) {
        env.insert_or_assign("hidden_state_tap", Buffer{&tap, 1, kCPU});
    }

    env.insert_or_assign("output_norm_weight", weights_.norm->weight);

    env.insert_or_assign("selected_token_pos", in.selected_token_pos);

    if (in.attention_metadata) {
        unified_decoder_->SetAttentionForwardMetadata(phase, *in.attention_metadata);
    }

    unified_decoder_->Forward(phase, env, weights_.layers_list(), in.selected_hidden_buffer);

    DecoderOutputs out;
    out.selected_hidden    = env.at("hidden_states");
    out.pre_final_residual = env.try_consume("pre_final_residual");
    return out;
}

Tensor LanguageModel::Impl::Logits(const Tensor& hidden, Tensor out, const TensorMap& env)
{
    Buffer symm_buf;
    if (comm_.d_comm) {
        symm_buf = env.at("symm_buf").buffer();
    }
    return PostEmbedding(hidden, *weights_.output, symm_buf, std::move(out));
}

const ModelWeight& LanguageModel::Impl::weights() const
{
    return weights_;
}

int LanguageModel::Impl::max_logits_len(const TensorMap& env) const
{
    if (has_head() && comm_.d_comm) {
        return env.at("symm_buf").buffer().view(weights_.data_type).size() / weights_.vocab_size_padded;
    }
    return max_logits_len_;
}

bool LanguageModel::Impl::logits_use_workspace() const
{
    return tp_size_ > 1 && use_ag2d_;
}

void LanguageModel::Impl::CommitAcceptedState(int phase, const Buffer_<int>& accept_len)
{
    unified_decoder_->CommitAcceptedState(phase, accept_len);
}

size_t LanguageModel::Impl::SpeculativeStateJournalBytes(int request_count, int verification_positions) const
{
    return unified_decoder_->SpeculativeStateJournalBytes(request_count, verification_positions);
}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel::LanguageModel(CacheRegistry&     registry,
                             const EngineParam& engine,
                             const Context&     ctx,
                             const ModelWeight& weights,
                             int                phases)
{
    impl_ = std::make_unique<Impl>(registry, engine, ctx, weights, phases);
}

bool LanguageModel::has_embedding() const
{
    return TM_CHECK_NOTNULL(impl_)->has_embedding();
}

bool LanguageModel::has_head() const
{
    return TM_CHECK_NOTNULL(impl_)->has_head();
}

Tensor LanguageModel::Embed(const Buffer_<int>& input_ids, Tensor out, const TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Embed(input_ids, std::move(out), env);
}

LanguageModel::DecoderOutputs
LanguageModel::RunDecoder(int phase, const DecoderInputs& in, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->RunDecoder(phase, in, env);
}

Tensor LanguageModel::Logits(const Tensor& hidden, Tensor out, const TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Logits(hidden, std::move(out), env);
}

const ModelWeight& LanguageModel::weights() const
{
    return TM_CHECK_NOTNULL(impl_)->weights();
}

int LanguageModel::max_logits_len(const TensorMap& env) const
{
    return TM_CHECK_NOTNULL(impl_)->max_logits_len(env);
}

bool LanguageModel::logits_use_workspace() const
{
    return TM_CHECK_NOTNULL(impl_)->logits_use_workspace();
}

void LanguageModel::CommitAcceptedState(int phase, const Buffer_<int>& accept_len)
{
    TM_CHECK_NOTNULL(impl_)->CommitAcceptedState(phase, accept_len);
}

size_t LanguageModel::SpeculativeStateJournalBytes(int request_count, int verification_positions) const
{
    return TM_CHECK_NOTNULL(impl_)->SpeculativeStateJournalBytes(request_count, verification_positions);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

}  // namespace turbomind
