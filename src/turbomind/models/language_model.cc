
#include "src/turbomind/models/language_model.h"

#include <memory>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/interval.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/generation/generation.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/output_processor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

struct LanguageModel::Impl {
    const DataType       dtype_;
    const ModelParam     param_;
    const AttentionParam attn_param_;
    const Communicators& comm_;
    const LlamaWeight&   weights_;
    LlamaLinear&         linear_;

    const int  tp_size_;
    const int  tp_rank_;
    const bool use_ag2d_;

    const bool debug_;

    Buffer_<bool> false_;

    // mutable state
    State finished_;
    State sequence_length_;  // length of known tokens
    // immutable state
    Buffer_<int> autoreg_ids_;
    // Buffer_<int> autoreg_ids_offsets_;

    // Symmetric buffer for holding global hidden states or logits
    Buffer_<uint8_t> symm_buf_;

    // Max chunk size for compute / output full logits
    int max_logits_len_ = 0;

    Buffer_<int>  sequence_length_buf_;
    Buffer_<bool> finished_buf_;

    struct Data {
        Buffer_<int>  sequence_length;
        Buffer_<bool> finished;

        Buffer_<bool> autoregres;
        Buffer_<bool> generating;

        int n_generating;
    };

    vector<Data> data_;

    std::optional<InputProcessor>   input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;
    std::optional<OutputProcessor>  output_processor_;
    std::unique_ptr<Generation>     generation_;  // token generator

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        switch (op) {
            case BatchOp::kSetup:
                return Setup(phase, env);
            case BatchOp::kPrepare:
                return Prepare(phase, env);
            case BatchOp::kForward:
                return Forward(phase, env);
            case BatchOp::kUnprep:
                return Unprep(phase, env);
            case BatchOp::kFetch:
                return Fetch(phase, env);
            default:
                input_processor_->Run(op, phase, env);
                unified_decoder_->Run(op, phase, env);
                generation_->Run(op, phase, env);
                output_processor_->Run(op, phase, env);
        }
    }

    Impl(DataType              dtype,
         const ModelParam&     model,
         const EngineParam&    engine,
         const AttentionParam& attn,
         const MoeParam&       moe,
         const Context&        ctx,
         const LlamaWeight&    weights,
         int                   phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer symm_buf);

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Forward(int phase, TensorMap& env);
    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);
};

LanguageModel::Impl::Impl(DataType              dtype,
                          const ModelParam&     model,
                          const EngineParam&    engine,
                          const AttentionParam& attn,
                          const MoeParam&       moe,
                          const Context&        ctx,
                          const LlamaWeight&    weights,
                          int                   phases):
    dtype_{dtype},
    param_{model},
    attn_param_{attn},
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{comm_.h_tp_group->n_ranks()},
    tp_rank_{comm_.h_tp_group->rank()},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    debug_{isDebug()}
{

    false_ = {engine.max_batch_size, kDEVICE};
    Clear(false_);

    finished_buf_ = {engine.max_batch_size, kCPUpinned};
    finished_     = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_ = {engine.max_batch_size, kDEVICE};
    // autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    // std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    sequence_length_buf_ = {engine.max_batch_size, kCPUpinned};
    sequence_length_     = {{engine.max_batch_size}, kInt, kDEVICE};
    for (int i = 0; i < phases; ++i) {
        auto& d           = data_.emplace_back();
        d.sequence_length = empty_like(sequence_length_buf_, kDEVICE);
        d.finished        = empty_like(finished_buf_, kDEVICE);
        d.autoregres      = {engine.max_batch_size, kCPU};
        d.generating      = {engine.max_batch_size, kCPU};
    }

    input_processor_.emplace(engine, param_, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, ctx, phases);

    generation_ = std::make_unique<Generation>(kFloat32,
                                               engine.max_batch_size,
                                               engine.session_len,
                                               model.vocab_size,
                                               weights.post_decoder_embedding.output_dim * tp_size_,
                                               comm_.h_tp_group,
                                               phases);

    const int     vocab_size     = weights_.post_decoder_embedding.output_dim * tp_size_;
    const ssize_t max_fwd_tokens = engine.max_forward_token_num;

    if (ctx.comm.d_comm) {
        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        // Native comm fuses allreduce & rmsnorm in token granularity
        TM_CHECK(engine.max_forward_token_num % tp_size_ == 0);

        ssize_t bytes{};
        bytes = std::max(bytes, byte_size(dtype_, max_fwd_tokens * engine.attn_dp_size * model.hidden_units));
        bytes = std::max(bytes, byte_size(dtype_, engine.max_batch_size * vocab_size));

        symm_buf_ = {bytes, symm_alloc};
        // Compute max logits length based on symm buffer size
        max_logits_len_ = symm_buf_.view(dtype_).size() / vocab_size;
    }
    else {
        max_logits_len_ = std::max<int>(max_fwd_tokens * model.hidden_units / vocab_size, engine.max_batch_size);
    }

    output_processor_.emplace(param_, max_logits_len_, tp_rank_, phases, [this](const Tensor& hstate) {
        return PostEmbedding(hstate, symm_buf_);
    });
}

Tensor LanguageModel::Impl::LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf)
{
    const auto st = core::Context::stream().handle();

    const int hidden_units = param_.hidden_units;

    const auto& embedding_table = weights_.pre_decoder_embedding.weight;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units);

    const int token_num = input_ids.size();

    Tensor input_embeds{{token_num, hidden_units}, dtype_, kDEVICE};

    if (token_num == 0) {
        return input_embeds;
    }

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, st);
        sync_check_cuda_error();
    }
    else if (use_ag2d_) {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(dtype_), {token_num, tp_size_, local_hidden_units}};
        Tensor local{temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        sync_check_cuda_error();

        comm_.d_comm->AllGather2D(local.raw_data(),
                                  temp.raw_data(),
                                  hidden_units,
                                  local_hidden_units,
                                  local_hidden_units,
                                  token_num,
                                  local.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        sync_check_cuda_error();

        Copy(temp.buffer(), input_embeds.buffer());
    }
    else {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(dtype_), {tp_size_, token_num, local_hidden_units}};
        Tensor local{temp.slice(tp_rank_).squeeze(0)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        sync_check_cuda_error();

        comm_.d_comm->AllGather(local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_.d_tp_group, st);
        sync_check_cuda_error();

        invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                  (uint16_t*)temp.raw_data(),
                                  tp_size_,
                                  token_num,
                                  local_hidden_units,
                                  false,
                                  st);
        sync_check_cuda_error();
    }

    return input_embeds;
}

Tensor LanguageModel::Impl::PostEmbedding(const Tensor& features, Buffer symm_buf)
{
    NvtxScope scope("postDecodeEmbedding");

    const auto st = core::Context::stream().handle();

    const int bsz              = features.shape(0);
    const int local_vocab_size = weights_.post_decoder_embedding.output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    if (bsz == 0) {
        return Tensor{{0, vocab_size}, dtype_, kDEVICE};
    }

    if (tp_size_ == 1) {
        Tensor logits{{bsz, vocab_size}, dtype_, kDEVICE};
        linear_.Forward(features, weights_.post_decoder_embedding, logits);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_ag2d_) {
        Tensor logits{symm_buf.view(dtype_), {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(1));
        sync_check_cuda_error();
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  logits.raw_data(),
                                  vocab_size,
                                  local_vocab_size,
                                  local_vocab_size,
                                  bsz,
                                  logits.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        sync_check_cuda_error();
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{symm_buf.view(dtype_), {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();
        comm_.d_comm->AllGather(local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        sync_check_cuda_error();
        Tensor out{{bsz, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, st);
        sync_check_cuda_error();
        return out;
    }
}

void LanguageModel::Impl::Setup(int phase, TensorMap& env)
{
    input_processor_->Run(BatchOp::kSetup, phase, env);

    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    const auto& rc = env.at("batch").data<BatchData*>()[0]->rc;

    d.n_generating = 0;

    for (int i = 0; i < rc.size(); ++i) {
        auto& c         = *rc[i];
        d.autoregres[i] = c.autoregres;
        d.generating[i] = c.generating;
        d.n_generating += c.generating;
        if (TM_UNLIKELY(!c.autoregres)) {
            sequence_length_buf_[i] = c.history_len + c.alpha + c.input_len;
        }
    }

    copy(sequence_length_buf_, rc.size(), d.sequence_length);

    unified_decoder_->Run(BatchOp::kSetup, phase, env);
    generation_->Run(BatchOp::kSetup, phase, env);
    output_processor_->Run(BatchOp::kSetup, phase, env);
}

void LanguageModel::Impl::Prepare(int phase, TensorMap& env)
{
    env.emplace("autoreg_ids", autoreg_ids_);

    input_processor_->Run(BatchOp::kPrepare, phase, env);

    auto& d = data_.at(phase);

    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    // core::CopyT copy{};

    if (auto group = copy.group()) {
        for (int i = 0; i < b.bsz; ++i) {
            if (const int j = b.perm[i]; j < b.bs0) {
                copy(finished_.front().data<bool>() + j, 1, finished_.back().data<bool>() + i);
            }
            else {
                copy(false_.data() + i, 1, finished_.back().data<bool>() + i);
            }
        }
        finished_.Swap();
    }

    if (auto group = copy.group()) {
        // sequence_length = history_len + input_len
        for (int i = 0; i < b.bsz; ++i) {
            if (const int j = b.perm[i]; j < b.bs0 && d.autoregres[i]) {
                copy(sequence_length_.front().data<int>() + j, 1, sequence_length_.back().data<int>() + i);
            }
            else {
                copy(d.sequence_length.data() + i, 1, sequence_length_.back().data<int>() + i);
            }
        }
        sequence_length_.Swap();
    }

    Buffer_<int> k_offsets{b.bsz + 1, kDEVICE};
    // PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());

    // Buffer_<int> k_offsets_tmp{k_offsets.size(), kCPU};
    // Buffer_<int> sequence_length_tmp{sequence_length_.front().size(), kCPU};

    // Copy(k_offsets, k_offsets_tmp);
    // Copy(sequence_length_.front().buffer(), sequence_length_tmp);

    // core::Context::stream().Sync();

    // dbg(core::to_vector<int>(sequence_length_tmp.slice(0, bsz)));
    // dbg(core::to_vector<int>(k_offsets_tmp.slice(0, bsz + 1)));

    env.produce("finished", finished_.front());
    env.produce("sequence_length", sequence_length_.front());
    env.produce("k_offsets", k_offsets);

    unified_decoder_->Run(BatchOp::kPrepare, phase, env);
    generation_->Run(BatchOp::kPrepare, phase, env);
    output_processor_->Run(BatchOp::kPrepare, phase, env);
}

void LanguageModel::Impl::Forward(int phase, TensorMap& env)
{

    auto& d = data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    {
        Buffer_<int> k_offsets = env.at("k_offsets").buffer();
        PrefixSum(sequence_length_.front().data<int>(), b.bsz, k_offsets.data(), core::Context::stream().handle());
    }

    {  // compute input embeddings
        auto input_ids = env.at("input_ids").buffer();

        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf_);
        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        input_processor_->PatchEmbedding(phase, input_embeds, copy);
        copy.Run();

        env.produce("input_embeds", std::move(input_embeds));
        // dbg(env);
    }

    if (symm_buf_) {
        env.produce("symm_buf", symm_buf_);
    }

    env.produce("output_norm_weight", weights_.output_norm_weight);

    unified_decoder_->Forward(phase, env, weights_.decoder_layer_weights);

    // env.at("batch").data<BatchData*>()[0]->Notify();

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 2);

    auto& hidden_states = env.at("hidden_states");

    env.produce("logits", PostEmbedding(hidden_states, symm_buf_));

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 1);

    if (d.n_generating) {
        generation_->Run(BatchOp::kForward, phase, env);
        Copy(env.at("output_ids").buffer(), autoreg_ids_);
    }
}

void LanguageModel::Impl::Unprep(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(sequence_length_.front().buffer(), d.sequence_length.size(), d.sequence_length);

    copy(finished_.front().buffer(), d.finished.size(), d.finished);

    generation_->Run(BatchOp::kUnprep, phase, env);
}

void LanguageModel::Impl::Fetch(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(d.sequence_length, d.sequence_length.size(), sequence_length_buf_);
    env.produce("sequence_length", sequence_length_buf_);

    copy(d.finished, d.finished.size(), finished_buf_);
    env.produce("finished", finished_buf_);

    env.produce("generating", d.generating);

    generation_->Run(BatchOp::kFetch, phase, env);
}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel::LanguageModel(DataType              dtype,
                             const ModelParam&     model,
                             const EngineParam&    engine,
                             const AttentionParam& attn,
                             const MoeParam&       moe,
                             const Context&        ctx,
                             const LlamaWeight&    weights,
                             int                   phases)
{
    impl_ = std::make_unique<Impl>(dtype, model, engine, attn, moe, ctx, weights, phases);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

const ModelParam& LanguageModel::model_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->param_;
}

const AttentionParam& LanguageModel::attn_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->attn_param_;
}

}  // namespace turbomind
