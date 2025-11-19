// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <curand_kernel.h>

#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

namespace turbomind {

struct MropeRope {
    int          stride{};
    Tensor_<int> position_ids;
    Buffer_<int> position_delta;
    Buffer_<int> length;
};

struct BatchState {

    Buffer_<int>  h_prompt_length;  // history + input, ignore generated
    Buffer_<int>  h_context_length;
    Buffer_<bool> h_finished;

    MropeRope mrope;

    Tensor_<uint8_t> curand_state;  // [n, sizeof(curandState_t)]

    Tensor_<int> output_ids;  // output ids in [B, S]

    Buffer_<float> h_rope_theta;

    std::vector<int> seq_len_limit;

    std::vector<const Sequence*>          sequences;
    std::vector<std::shared_ptr<Request>> requests;

    std::vector<int> errors;

    // |<-- existing -->|<-- swap-in -->|
    // |<----------- active ----------->|<-- inactive -->|
    int active_size;
    int size;
};

class LlamaV2;

struct GenerationState {
    int max_init_ctx_len;
    int step;

    int partial;
    int partial_context_legnth;

    std::vector<uint64_t> unique_ids;

    bool skip_init_sampling;

    // min tokens per iter for satisfying `max_prefill_iters` constraint
    std::deque<int> min_input_count;

    int finished_count;
};

class LlamaBatch {
public:
    void AllocateBuffer(ssize_t batch_size, ssize_t session_len, int cache_block_seq_len);

    void AllocSymmBuffers();
    void FreeSymmBuffers();

    void FreeBuffer();

    using Requests = std::vector<std::shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    void DisableInvalidRequests(Requests& infer_reqs, Requests& kill_reqs);

    void ProcessKillRequests(const Requests& reqs, std::vector<Signal>& signals);

    void ProcessInferRequests(const Requests& reqs, std::vector<Signal>& signals);

    int AdjustMaxInputCount(GenerationState&                    g,
                            const std::vector<const Sequence*>& sequences,
                            const std::vector<int>&             context_length);

    void Initialize(GenerationState& g);

    void InitializeSampling(const GenerationState& g);

    bool Forward(GenerationState& g);

    void Finish(GenerationState& g, std::vector<Signal>& signals);

    [[nodiscard]] Signal Interrupt(int index, bool force_stop = false);

    void ComputeAndOutputLogits(const Tensor& hidden_states, int first, int last);

    void OutputLogits(const Tensor& logits, int first, int last, GenerationConfig::OutType out_type);

    void OutputLastHiddenState(const Tensor& hidden_states, int first, int last);

    explicit LlamaBatch(DataType                 data_type,
                        const EngineParam&       param,
                        std::unique_ptr<LlamaV2> model,
                        std::shared_ptr<Context> ctx,
                        std::shared_ptr<Gateway> gateway,
                        int                      device_id,
                        int                      dp_rank);

    ~LlamaBatch();

    void InitializeBufferAndKVCache();

    void FreeBufferAndKVCache();

    void Start();

    LlamaV2& model() noexcept
    {
        return *model_;
    }

    int session_len() const noexcept
    {
        return session_len_;
    }

    void Warmup();

    ScheduleMetrics getScheduleMetrics()
    {
        const std::lock_guard<std::mutex> lock(metrics_mutex_);
        return schedule_metrics_;
    }

private:
    void FindCanceledIndices(std::vector<int>& indices);

    void ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals);

    void InternalThreadEntry();

    void OutputThreadEntry();

    void CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc);

    template<class... Ts>
    void IndexedCopyImpl(const int* src_idx, const int* dst_idx, int count, const std::tuple<Ts*, Ts*, int>&... cpys)
    {
        if (!count) {
            return;
        }
        constexpr int N = sizeof...(Ts);
        static_assert((!std::is_same_v<Ts, void> && ...));
        std::array<void*, N> src_ptr{std::get<0>(cpys)...};
        std::array<void*, N> dst_ptr{std::get<1>(cpys)...};
        std::array<int, N>   elem_sz{int(sizeof(Ts) * std::get<2>(cpys))...};
        invokeIndexedCopy(src_ptr.data(),  //
                          dst_ptr.data(),
                          elem_sz.data(),
                          src_idx,
                          dst_idx,
                          count,
                          N,
                          stream_);
        sync_check_cuda_error();
    }

    template<class... Ts>
    void IndexedCopy(const std::vector<int>& src_idx,
                     const std::vector<int>& dst_idx,
                     const std::tuple<Ts*, Ts*, int>&... cpys)
    {
        // has the same size, or one is empty
        FT_CHECK(src_idx.size() == dst_idx.size() || (src_idx.empty() ^ dst_idx.empty()));
        IndexedCopyImpl(src_idx.empty() ? nullptr : src_idx.data(),
                        dst_idx.empty() ? nullptr : dst_idx.data(),
                        std::max(src_idx.size(), dst_idx.size()),
                        cpys...);
    }

    template<class... Ts>
    void IndexedCopy(int count, const std::tuple<Ts*, Ts*, int>&... cpys)
    {
        IndexedCopyImpl(nullptr, nullptr, count, cpys...);
    }

    void* SymmAlloc(size_t size, bool register_);

    void SymmFree(void* ptr, size_t size, bool deregister);

    void DestroyCommunicators();

    void UpdateMetrics();

private:
    const EngineParam param_;

    const std::shared_ptr<Gateway> gateway_;

    const int      max_batch_size_;
    const int      max_forward_token_num_;
    const int      max_context_token_num_;
    const int      num_tokens_per_iter_;
    const int      max_prefill_iters_;
    const int      device_id_;
    const int      dp_rank_;
    const int      tp_size_;
    const int      tp_rank_;
    const DataType data_type_;
    const bool     debug_;

    // Refs into `Context<T>`
    cudaStream_t const stream_{};

    int session_len_;  // May be truncated in ctor

    std::shared_ptr<Context>         context_;
    std::unique_ptr<LlamaV2>         model_;
    std::unique_ptr<SequenceManager> sequence_manager_;

    Communicators& comm_;

    Allocator symm_alloc_;

    ///////////////////////////////////////////////////////////////////
    // k/v cache block buffers
    Buffer_<int>       cu_block_counts_;
    Buffer_<uintptr_t> block_ptrs_;

    ////////////////////////////////////////////////////////////////////
    // context decoding temp buffers
    Tensor symm_hidden_states_buf_;
    Tensor symm_logits_buf_;

    // context parallel
    Tensor_<float> symm_partial_ML_;

    Tensor decoder_output_buf_;

    Tensor_<float> sampling_logits_;

    Buffer_<int> input_ids_buf_;

    // lengths
    Buffer_<int> input_length_buf_;    // input + cache missed length
    Buffer_<int> context_length_buf_;  // history length + input_length
    Buffer_<int> init_context_length_;

    Buffer_<int> sequence_lengths_;  // current sequence length
    Buffer_<int> init_ctx_lens_;
    Buffer_<int> lora_mask_buf_;  // lora

    Buffer_<float>    sampled_logprobs_;
    Buffer_<uint32_t> sampled_indexes_;
    Buffer_<uint32_t> sampled_nums_;
    Buffer_<float>    h_sampled_logprobs_;
    Buffer_<uint32_t> h_sampled_indexes_;
    Buffer_<uint32_t> h_sampled_nums_;

    Buffer_<float> rope_theta_;

    // used by dynamic decoder
    Buffer_<int>  token_ids_buf_;  // all token IDs in [S, B], indexed using `step`
    Buffer_<bool> finished_buf_;
    Buffer_<int>  seq_limit_len_;

    // pinned buffers
    Buffer_<int> h_output_ids_;
    Buffer_<int> h_input_length_buf_;
    Buffer_<int> h_seq_limit_len_;

    Buffer_<int>       h_cu_block_counts_;
    Buffer_<uintptr_t> h_block_ptrs_;

    Buffer_<uint64_t> h_random_seed_;
    Buffer_<uint64_t> d_random_seed_;

    Tensor_<uint8_t> h_curand_state_;  // [n, sizeof(curandState_t)]
    Tensor_<uint8_t> d_curand_state_;

    std::array<BatchState, 3> states_{};

    BatchState* state_{};
    BatchState* back_{};
    BatchState* incoming_{};

    // hard limits for persistent buffers
    static constexpr int kMaxStopBadWordsLen = 32;
    static constexpr int kMaxEndIdsSize      = 32;

    std::thread internal_thread_;

    bool            enable_metrics_;
    ScheduleMetrics schedule_metrics_;
    std::mutex      metrics_mutex_;
};

using Engine = LlamaBatch;

}  // namespace turbomind
