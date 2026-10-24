// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/model_executor.h"

#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/model.h"
#include "src/turbomind/generation/target_verification.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/hidden_state_tap.h"
#include "src/turbomind/models/speculative/speculative_model.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nvtx_utils.h"

namespace turbomind {

using std::unique_ptr;

struct DraftContext;

struct ModelExecutor::Impl {
    struct TargetPass {
        Tensor hidden_states;
        Tensor pre_final_residual;
        Tensor embedding_storage;
        Tensor head_storage;
    };

    // The served model, owned by the engine's composition root and shared by
    // reference.
    Model& model_;

    LlamaLinear& linear_;

    const int device_id_;

    Queue<std::unique_ptr<BatchData>>& inbound_;
    Queue<std::unique_ptr<BatchData>>& outbound_;

    // Executor-internal scratch shared by the prepare step and the target pass.
    Buffer_<int> autoreg_ids_;

    std::thread internal_thread_;

    Impl(Model&                             model,
         const EngineParam&                 param,
         Context&                           context,
         int                                device_id,
         Queue<std::unique_ptr<BatchData>>& inbound,
         Queue<std::unique_ptr<BatchData>>& outbound);

    ~Impl();

    // Ordinary and shared execution.

    void InternalThreadEntry();

    static void RunCopies(std::vector<ResolvedCopy>& copies);

    void Run(BatchData& data);

    // Device-bracket steps, in execution order.

    // Prepare: non-forward component fanout plus the prepare-only cases —
    // ordinary autoregressive-id exposure, the composed engine's explicit
    // draft-input publish, and the executor-owned k-offsets buffer published
    // before the target prepares and filled at forward in both compositions.
    void PrepareStep(int phase, TensorMap& env);

    // Forward: the visible composition branch (ADR 0002) — the speculative
    // round when a speculator is composed, the ordinary target pass otherwise.
    void ForwardStep(int phase, TensorMap& env);

    // Unprep: non-forward component fanout.
    void UnprepStep(int phase, TensorMap& env);

    // Shared target-pass step: publishes the symmetric buffer, then embeds,
    // patches, and decodes the submitted rows. Publishing here holds the
    // publish-before-decode ordering at one site for both forward routines.
    // The tap arrives as an argument so the ordinary path holds no speculative
    // branch (nullptr) while the speculative round supplies its own.
    TargetPass RunTarget(int phase, TensorMap& env, HiddenStateTap* taps);

    // The ordinary forward: exactly one target pass, then logits and sampling.
    void ForwardTargetPass(int phase, TensorMap& env);

    void RunVisionPass(int phase, TensorMap& env);

    void Start();

    // Composed speculative execution.

    // The composed forward: one full speculative round owning the mixed batch.
    void ForwardSpeculativeRound(int phase, TensorMap& env);

    DraftContext MakeDraftContext(const TargetPass& target, TensorMap& env);
};

ModelExecutor::Impl::Impl(Model&                             model,
                          const EngineParam&                 param,
                          Context&                           context,
                          int                                device_id,
                          Queue<std::unique_ptr<BatchData>>& inbound,
                          Queue<std::unique_ptr<BatchData>>& outbound):
    model_{model},
    linear_{*context.linear},
    device_id_{device_id},
    inbound_{inbound},
    outbound_{outbound},
    autoreg_ids_{param.max_batch_size, kDEVICE}
{
}

ModelExecutor::Impl::~Impl()
{
    if (internal_thread_.joinable()) {
        internal_thread_.join();
    }
}

void ModelExecutor::Impl::InternalThreadEntry()
{
    TM_FUNCTION_SCOPE();
    TM_CUDA_CHECK(cudaSetDevice(device_id_));

    Stream    stream  = Stream::create();
    Allocator h_alloc = Allocator(kCPU);
    Allocator d_alloc = Allocator(stream, false);

    AnomalyHandler::instance().Init(0, 1000, 0, 1000, stream.handle());

    core::ContextGuard ctx{stream, h_alloc, d_alloc};

    // Default GEMM workspace for everything dispatched on this stream;
    // bound for the whole work loop, which is the outer-most scope that
    // drives `linear_`.
    gemm::Workspace workspace{stream.handle()};
    auto            ws_lifetime = linear_.With(workspace);

    unique_ptr<BatchData> data;
    while (inbound_.pop(data)) {
        TM_CHECK_NOTNULL(data);
        core::Context::stream().Wait(data->ready);
        Run(*data);
        data->done.Record(core::Context::stream());
        outbound_.push(std::move(data));
    }

    // Stream-ordered teardown: the frees run after the last batch's kernels.
    workspace.Release(stream.handle());
}

void ModelExecutor::Impl::RunCopies(std::vector<ResolvedCopy>& copies)
{
    for (const auto& copy : copies) {
        Copy(Buffer_<uint8_t>{static_cast<uint8_t*>(copy.src), static_cast<ssize_t>(copy.bytes), kDEVICE},
             Buffer_<uint8_t>{static_cast<uint8_t*>(copy.dst), static_cast<ssize_t>(copy.bytes), kDEVICE});
    }
    copies.clear();
}

void ModelExecutor::Impl::Run(BatchData& data)
{
    TM_FUNCTION_SCOPE();

    BatchCopy copy;
    TensorMap env{{"batch", data.buf()}, {"copy", copy.buf()}};

    RunCopies(data.restore_copies);

    PrepareStep(data.phase, env);
    copy.Run();

    ForwardStep(data.phase, env);

    UnprepStep(data.phase, env);
    copy.Run();

    RunCopies(data.publish_copies);

    AnomalyHandler::instance().Summarize([](...) {});
    AnomalyHandler::instance().Reset();
}

void ModelExecutor::Impl::PrepareStep(int phase, TensorMap& env)
{
    if (model_.vision) {
        model_.vision->Run(BatchOp::kPrepare, phase, env);
    }

    if (!model_.spec) {
        env.emplace("autoreg_ids", autoreg_ids_);
    }

    model_.status.Run(BatchOp::kPrepare, phase, env);
    model_.generation.Run(BatchOp::kPrepare, phase, env);
    if (model_.spec) {
        model_.generation.Verification()->PublishDraftInputs(phase, env);
    }
    model_.input_processor.Run(BatchOp::kPrepare, phase, env);

    const int batch_size = env.at("batch").data<BatchData*>()[0]->bsz;
    env.produce("k_offsets", Buffer_<int>{batch_size + 1, kDEVICE});

    model_.target->Run(BatchOp::kPrepare, phase, env);
    if (model_.spec) {
        model_.spec->Run(BatchOp::kPrepare, phase, env);
    }
    model_.output_processor.Run(BatchOp::kPrepare, phase, env);
}

void ModelExecutor::Impl::ForwardStep(int phase, TensorMap& env)
{
    if (model_.spec) {
        return ForwardSpeculativeRound(phase, env);
    }
    ForwardTargetPass(phase, env);
}

void ModelExecutor::Impl::UnprepStep(int phase, TensorMap& env)
{
    model_.Run(BatchOp::kUnprep, phase, env);
}

void ModelExecutor::Impl::RunVisionPass(int phase, TensorMap& env)
{
    if (model_.vision) {
        model_.vision->Run(BatchOp::kForward, phase, env);
    }
}

ModelExecutor::Impl::TargetPass ModelExecutor::Impl::RunTarget(int phase, TensorMap& env, HiddenStateTap* taps)
{
    const auto& batch = *env.at("batch").data<BatchData*>()[0];
    if (batch.symm_buf) {
        env.insert_or_assign("symm_buf", batch.symm_buf);
    }

    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Tensor residual = model_.target->Embed(env.at("input_ids").buffer(), {}, env);
    TM_DEBUG_TENSOR(residual, "embeddings", 1);

    model_.input_processor.PatchEmbedding(phase, residual, copy, env);
    copy.Run();

    TargetPass out;
    out.embedding_storage = residual;

    LanguageModel::DecoderInputs in;
    in.residual               = std::move(residual);
    in.taps                   = taps;
    in.selected_token_pos     = env.consume("selected_token_pos").buffer();
    in.selected_hidden_buffer = env.try_consume("selected_normalized_hidden_buffer");

    auto decoded = model_.target->RunDecoder(phase, in, env);

    out.hidden_states      = std::move(decoded.selected_hidden);
    out.pre_final_residual = std::move(decoded.pre_final_residual);
    return out;
}

void ModelExecutor::Impl::ForwardTargetPass(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    NvtxScope forward_scope{"targetExecutorForward"};

    RunVisionPass(phase, env);

    const auto  stream = core::Context::stream().handle();
    const auto& batch  = *env.at("batch").data<BatchData*>()[0];

    PrefixSum(model_.status.SequenceLength().data(), batch.bsz, env.at("k_offsets").buffer().data<int>(), stream);

    TargetPass target = RunTarget(phase, env, nullptr);

    model_.output_processor.OutputHiddenStatesAndLogits(phase, env, 2);

    target.head_storage = model_.target->Logits(target.hidden_states, {}, env);
    env.produce("logits", target.head_storage);

    model_.output_processor.OutputHiddenStatesAndLogits(phase, env, 1);

    if (model_.status.GeneratingCount(phase)) {
        model_.generation.Run(BatchOp::kForward, phase, env);
        Copy(env.at("output_ids").buffer(), autoreg_ids_);
    }
}

DraftContext ModelExecutor::Impl::MakeDraftContext(const TargetPass& target, TensorMap& env)
{
    const auto& batch = *env.at("batch").data<BatchData*>()[0];

    DraftContext ctx;
    ctx.batch_size                = batch.bsz;
    ctx.target_q_offsets          = env.at("q_offsets").buffer();
    ctx.target_k_offsets          = env.at("k_offsets").buffer();
    ctx.accept_len                = env.at("accept_len").buffer();
    ctx.sequence_length           = model_.status.SequenceLength();
    ctx.finished_on_entry         = env.at("finished_on_entry").buffer();
    ctx.request_token_ids_ptrs    = env.at("request_token_ids_ptrs").data<int*>();
    ctx.target_pre_final_residual = target.pre_final_residual;
    ctx.embedding_storage         = target.embedding_storage;
    ctx.head_storage              = target.head_storage;
    ctx.input_processor           = &model_.input_processor;
    return ctx;
}

void ModelExecutor::Impl::ForwardSpeculativeRound(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    NvtxScope forward_scope{"speculativeExecutorForward"};

    RunVisionPass(phase, env);

    model_.input_processor.BuildTargetInputs(phase, env);

    const auto&  batch  = *env.at("batch").data<BatchData*>()[0];
    cudaStream_t stream = core::Context::stream().handle();
    PrefixSum(env.at("target_key_lengths").buffer().data<int>(), batch.bsz,
              env.at("k_offsets").buffer().data<int>(), stream);

    auto& verification = *model_.generation.Verification();
    env.produce("selected_normalized_hidden_buffer",
                verification.SelectedHiddenBuffer(phase, env.at("selected_token_pos").size()));

    HiddenStateTap* tap = model_.spec->Tap(phase);
    if (tap) {
        tap = tap->Arm(phase, batch.local_token_num, stream);
    }

    TargetPass target = RunTarget(phase, env, tap);

    const int positions = model_.status.VerificationPositions(phase);

    verification.InitializeTargetVerification(phase, positions, env);

    target.head_storage = model_.target->Logits(target.hidden_states, {}, env);
    verification.ProcessTargetBlock(phase, positions, target.head_storage, env);
    verification.ClampSelectedSpan(phase, env);

    model_.target->CommitAcceptedState(phase, env.at("accept_len").buffer());

    {
        NvtxScope draft_scope{"SpeculativeModel::RunDraft"};
        model_.spec->RunDraft(phase, MakeDraftContext(target, env), env);
    }

    verification.CommitAcceptedSpan(phase, model_.status.SequenceLength());
}

void ModelExecutor::Impl::Start()
{
    internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
}

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor()                         = default;
ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept = default;
ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

ModelExecutor::ModelExecutor(Model&                             model,
                             const EngineParam&                 param,
                             Context&                           context,
                             int                                device_id,
                             Queue<std::unique_ptr<BatchData>>& inbound,
                             Queue<std::unique_ptr<BatchData>>& outbound):
    impl_{std::make_unique<Impl>(model, param, context, device_id, inbound, outbound)}
{
}

void ModelExecutor::Start()
{
    return impl_->Start();
}

}  // namespace turbomind
