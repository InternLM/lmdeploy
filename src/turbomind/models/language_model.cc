
#include "src/turbomind/models/language_model.h"

#include <memory>
#include <numeric>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/interval.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/generation/generation.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/output_processor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

struct LanguageModel::Impl {
    const Communicators& comm_;
    const ModelWeight&   weights_;
    LlamaLinear&         linear_;

    const int  tp_size_;
    const int  tp_rank_;
    const bool use_ag2d_;

    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int max_batch_size_;

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

    // Global (all attention DP ranks) per-token validity mask, built at Forward time;
    // consumed by the attention layers (their DP-local slice) and, eventually, the MoE router.
    Buffer_<bool> token_mask_;

    // Symmetric gather buffer for the per-rank `[q_offsets | finished]` metadata blocks
    // ([attn_dp_size, meta_bytes], 16B-aligned rows for the in-place AllGather).
    // Only allocated when attn_dp > 1.
    Tensor_<uint8_t> symm_token_meta_;

    // Max chunk size for compute / output full logits
    int max_logits_len_ = 0;

    Buffer_<int>  sequence_length_buf_;
    Buffer_<int>  readonly_block_num_buf_;  // {max_batch_size}, kCPUpinned
    Buffer_<bool> finished_buf_;

    struct Data {
        Buffer_<int>  sequence_length;
        Buffer_<int>  readonly_block_num;
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

    Impl(
        CacheRegistry& registry, const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer symm_buf);

    // Build the global per-token validity mask for this pass (see `token_mask_`).
    void BuildTokenMask(const bool* finished, const int* q_offsets, const BatchData& b);

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Forward(int phase, TensorMap& env);
    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);
};

LanguageModel::Impl::Impl(
    CacheRegistry& registry, const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases):
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{comm_.h_tp_group->n_ranks()},
    tp_rank_{comm_.h_tp_group->rank()},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    attn_dp_size_{engine.attn_dp_size},
    attn_dp_rank_{engine.attn_dp_rank},
    max_batch_size_{engine.max_batch_size},
    debug_{isDebug()}
{

    false_ = {engine.max_batch_size, kDEVICE};
    Clear(false_);

    finished_buf_ = {engine.max_batch_size, kCPUpinned};
    finished_     = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_ = {engine.max_batch_size, kDEVICE};
    // autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    // std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    sequence_length_buf_    = {engine.max_batch_size, kCPUpinned};
    readonly_block_num_buf_ = {engine.max_batch_size, kCPUpinned};
    sequence_length_        = {{engine.max_batch_size}, kInt, kDEVICE};
    for (int i = 0; i < phases; ++i) {
        auto& d              = data_.emplace_back();
        d.sequence_length    = empty_like(sequence_length_buf_, kDEVICE);
        d.readonly_block_num = empty_like(readonly_block_num_buf_, kDEVICE);
        d.finished           = empty_like(finished_buf_, kDEVICE);
        d.autoregres         = {engine.max_batch_size, kCPU};
        d.generating         = {engine.max_batch_size, kCPU};
    }

    input_processor_.emplace(engine, weights_.hidden_units, weights_.data_type, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(registry, engine, ctx, phases, weights_);

    const int vocab_size = weights_.output->output_dim * tp_size_;

    generation_ = std::make_unique<Generation>(
        kFloat32, engine.max_batch_size, engine.session_len, weights_.vocab_size, vocab_size, comm_.h_tp_group, phases);

    const ssize_t max_fwd_tokens = engine.max_forward_token_num;

    if (ctx.comm.d_comm) {
        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        // Native comm fuses allreduce & rmsnorm in token granularity
        TM_CHECK(engine.max_forward_token_num % tp_size_ == 0);

        ssize_t bytes{};
        bytes = std::max(bytes,
                         byte_size(weights_.data_type, max_fwd_tokens * engine.attn_dp_size * weights_.hidden_units));
        bytes = std::max(bytes, byte_size(weights_.data_type, engine.max_batch_size * vocab_size));

        symm_buf_ = {bytes, symm_alloc};
        // Compute max logits length based on symm buffer size
        max_logits_len_ = symm_buf_.view(weights_.data_type).size() / vocab_size;

        if (attn_dp_size_ > 1) {
            const int q_bytes    = (max_batch_size_ + 1) * (int)sizeof(int);
            const int meta_bytes = (q_bytes + max_batch_size_ + 15) / 16 * 16;
            symm_token_meta_     = {{attn_dp_size_, meta_bytes}, symm_alloc};
        }
    }
    else {
        max_logits_len_ = std::max<int>(max_fwd_tokens * weights_.hidden_units / vocab_size, engine.max_batch_size);
    }

    token_mask_ = {max_fwd_tokens * attn_dp_size_, kDEVICE};

    output_processor_.emplace(weights_.vocab_size, max_logits_len_, tp_rank_, phases, [this](const Tensor& hstate) {
        return PostEmbedding(hstate, symm_buf_);
    });
}

Tensor LanguageModel::Impl::LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf)
{
    TM_FUNCTION_SCOPE();
    const auto st = core::Context::stream().handle();

    const int hidden_units = weights_.hidden_units;

    const auto& embedding_table = weights_.tok_embeddings;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units);

    const int token_num = input_ids.size();

    Tensor input_embeds{{token_num, hidden_units}, weights_.data_type, kDEVICE};

    if (token_num == 0) {
        return input_embeds;
    }

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, st);
        TM_CUDA_CHECK(cudaGetLastError());
    }
    else if (use_ag2d_) {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(weights_.data_type), {token_num, tp_size_, local_hidden_units}};
        Tensor local{temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        TM_CUDA_CHECK(cudaGetLastError());

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
        TM_CUDA_CHECK(cudaGetLastError());

        Copy(temp.buffer(), input_embeds.buffer());
    }
    else {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(weights_.data_type), {tp_size_, token_num, local_hidden_units}};
        Tensor local{temp.slice(tp_rank_).squeeze(0)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        TM_CUDA_CHECK(cudaGetLastError());

        comm_.d_comm->AllGather(
            local.raw_data(), temp.raw_data(), local.size(), weights_.data_type, comm_.d_tp_group, st);
        TM_CUDA_CHECK(cudaGetLastError());

        invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                  (uint16_t*)temp.raw_data(),
                                  tp_size_,
                                  token_num,
                                  local_hidden_units,
                                  false,
                                  st);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    if (weights_.embedding_norm) {
        const auto& norm = *weights_.embedding_norm;
        invokeRMSNorm(input_embeds, input_embeds, norm.weight, norm.norm_eps_, norm.zero_centered_, st);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    return input_embeds;
}

Tensor LanguageModel::Impl::PostEmbedding(const Tensor& features, Buffer symm_buf)
{
    TM_FUNCTION_SCOPE();
    NvtxScope scope("postDecodeEmbedding");

    const auto st = core::Context::stream().handle();

    const int bsz              = features.shape(0);
    const int local_vocab_size = weights_.output->output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    auto finalize = [&](Tensor logits) {
        invokeLogitTransform(logits, weights_.logit_scale, weights_.logit_softcap, st);
        return logits;
    };

    if (bsz == 0) {
        return Tensor{{0, vocab_size}, weights_.data_type, kDEVICE};
    }

    if (tp_size_ == 1) {
        Tensor logits{{bsz, vocab_size}, weights_.data_type, kDEVICE};
        TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, logits));
        TM_DEBUG_TENSOR(logits, "logits", 1);
        return finalize(std::move(logits));
    }
    else if (use_ag2d_) {
        Tensor logits{symm_buf.view(weights_.data_type), {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, local.squeeze(1)));
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
        TM_CUDA_CHECK(cudaGetLastError());
        return finalize(logits.view({bsz, -1}));
    }
    else {
        Tensor logits{symm_buf.view(weights_.data_type), {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, local.squeeze(0)));
        comm_.d_comm->AllGather(local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        TM_CUDA_CHECK(cudaGetLastError());
        Tensor out{{bsz, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, st);
        TM_CUDA_CHECK(cudaGetLastError());
        return finalize(std::move(out));
    }
}

void LanguageModel::Impl::Setup(int phase, TensorMap& env)
{
    input_processor_->Run(BatchOp::kSetup, phase, env);

    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    d.n_generating = 0;

    for (int i = 0; i < rc.size(); ++i) {
        auto& c         = *rc[i];
        d.autoregres[i] = c.autoregres;
        d.generating[i] = c.generating;
        d.n_generating += c.generating;
        if (TM_UNLIKELY(!c.autoregres)) {
            sequence_length_buf_[i] = c.history_len + c.inflight_input_len + c.input_len;
        }
        readonly_block_num_buf_[i] = c.readonly_block_num;  // all rows, batch order
    }

    copy(sequence_length_buf_, rc.size(), d.sequence_length);
    copy(readonly_block_num_buf_, rc.size(), d.readonly_block_num);

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
        // Non-autoregressive rows use the submitted prefix length:
        // sequence_length = history_len + inflight_input_len + input_len.
        // Existing autoregressive rows carry the previous sequence_length forward.
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
    env.produce("readonly_block_num", d.readonly_block_num);
    env.produce("k_offsets", k_offsets);
    if (symm_buf_) {
        env.produce("symm_buf", symm_buf_);
    }

    // Produced here so consumers may borrow the pointer at kPrepare; the content is
    // only built at Forward time (`BuildTokenMask`).
    env.produce("token_mask", token_mask_);

    unified_decoder_->Run(BatchOp::kPrepare, phase, env);
    generation_->Run(BatchOp::kPrepare, phase, env);
    output_processor_->Run(BatchOp::kPrepare, phase, env);
}

void LanguageModel::Impl::BuildTokenMask(const bool* finished, const int* q_offsets, const BatchData& b)
{
    TM_FUNCTION_SCOPE();

    if (b.global_token_num == 0) {
        return;
    }

    TM_CHECK_EQ((int)b.local_token_num.size(), attn_dp_size_);
    TM_CHECK_LE(attn_dp_size_, kMaxAttnDPSize);

    const auto st = core::Context::stream().handle();

    // Byte stride between per-rank metadata blocks (0 when attn_dp == 1).
    size_t rank_stride = 0;

    if (attn_dp_size_ > 1) {
        const int q_bytes    = (max_batch_size_ + 1) * (int)sizeof(int);
        const int meta_bytes = symm_token_meta_.shape(1);

        // Stage this rank's metadata into its row of the symmetric buffer; the finished
        // tail is zeroed so padding slots never invalidate tokens.
        TM_CHECK_LE(b.bsz, max_batch_size_);
        char* slot = (char*)symm_token_meta_.data() + (ssize_t)attn_dp_rank_ * meta_bytes;
        core::Copy(q_offsets, b.bsz + 1, (int*)slot);
        core::Copy(finished, b.bsz, (bool*)(slot + q_bytes));
        TM_CUDA_CHECK(cudaMemsetAsync(slot + q_bytes + b.bsz, 0, max_batch_size_ - b.bsz, st));

        // In-place all-gather: the peers read this rank's contribution from its own row.
        comm_.d_comm->AllGather(slot, symm_token_meta_.data(), meta_bytes, kUint8, comm_.d_dp_group, st);

        q_offsets   = (const int*)symm_token_meta_.data();
        finished    = (const bool*)(symm_token_meta_.data() + q_bytes);
        rank_stride = meta_bytes;
    }

    // Rank r's tokens occupy [token_base[r], token_base[r] + local_token_num[r]) of the mask.
    int token_base[kMaxAttnDPSize];
    token_base[0] = 0;
    std::partial_sum(b.local_token_num.begin(), b.local_token_num.end() - 1, token_base + 1);

    invokeBuildTokenMask(token_mask_.data(),
                         finished,
                         q_offsets,
                         rank_stride,
                         token_base,
                         attn_dp_size_,
                         // DP > 1 scans all gathered slots (the finished tail is zeroed);
                         // DP == 1 scans only the active batch — beyond it the local
                         // `finished`/`q_offsets` hold stale data from previous passes.
                         attn_dp_size_ > 1 ? max_batch_size_ : b.bsz,
                         b.global_token_num,
                         st);
}

void LanguageModel::Impl::Forward(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();

    auto& d = data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    // Must run at Forward time: the `finished`/`q_offsets` H2D copies are only flushed
    // after kPrepare returns. The mask is ready before the decoder (its consumers) runs.
    BuildTokenMask(
        (const bool*)env.at("finished").buffer().raw_data(), (const int*)env.at("q_offsets").buffer().raw_data(), b);

    {
        Buffer_<int> k_offsets = env.at("k_offsets").buffer();
        PrefixSum(sequence_length_.front().data<int>(), b.bsz, k_offsets.data(), core::Context::stream().handle());
    }

    {  // compute input embeddings
        auto input_ids = env.at("input_ids").buffer();

        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf_);
        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        input_processor_->PatchEmbedding(phase, input_embeds, copy, env);
        copy.Run();

        env.produce("input_embeds", std::move(input_embeds));
        // dbg(env);
    }

    env.produce("output_norm_weight", weights_.norm->weight);

    unified_decoder_->Forward(phase, env, weights_.layers_list());

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

    unified_decoder_->Run(BatchOp::kUnprep, phase, env);
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

LanguageModel::LanguageModel(
    CacheRegistry& registry, const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases)
{
    impl_ = std::make_unique<Impl>(registry, engine, ctx, weights, phases);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

}  // namespace turbomind
