// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <curand_kernel.h>
#include <deque>
#include <queue>
#include <vector>

#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

struct BatchState {

    Buffer_<int>  h_prompt_length;  // history + input, ignore generated
    Buffer_<int>  h_context_length;
    Buffer_<bool> h_finished;

    Tensor_<uint8_t> curand_state;  // [n, sizeof(curandState_t)]

    Tensor_<int> output_ids;  // output ids in [B, S]

    Buffer_<float> h_rope_theta;

    std::vector<int> seq_len_limit;

    std::vector<const Sequence*>          sequences;
    std::vector<std::shared_ptr<Request>> requests;

    std::vector<int> errors;

    Buffer_<int> input_ids_buf;

    // lengths
    Buffer_<int> context_length_buf;  // history length + input_length
    Buffer_<int> sequence_lengths;    // current sequence length, updated by sampling
    Buffer_<int> init_context_length;
    Buffer_<int> h_input_length_buf;

    bool copy_init{true};         // extra flag for pipeline parallel to control whether to copy init_context_length
    bool pp_init_sampling{true};  // flag to control whether to copy required buffers when initializing sampling

    // rope theta
    Buffer_<float> rope_theta;

    // used by dynamic decoder
    Buffer_<int>  token_ids_buf;  // all token IDs in [S, B], indexed using `step`
    Buffer_<bool> finished_buf;
    Buffer_<int>  h_seq_limit_len;
    Buffer_<int>  seq_limit_len;

    // value set when model forward, no need to copy
    int              dc_batch_size;
    int              pf_batch_size;
    std::vector<int> local_token_nums;
    int              global_token_num;
    Tensor           hidden_states;
    Tensor           residual;

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

// struct for pipeline parallel
struct IntermediateData {
    bool abort{false};

    // cpu
    std::vector<BlockIds> blocks;
    Buffer_<int>          h_cu_block_counts;
    Buffer_<int>          h_input_length_buf;
    Buffer_<int>          h_context_length;
    Buffer_<float>        h_rope_theta;
    Buffer_<bool>         h_finished;
    int                   dc_batch_size;
    int                   pf_batch_size;

    std::vector<int> local_token_nums;
    int              global_token_num;

    // gpu
    // hidden, residual, logits
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

    bool Forward(GenerationState*& g);

    void Finish(GenerationState& g, std::vector<Signal>& signals);

    [[nodiscard]] Signal Interrupt(int index, bool force_stop = false, bool force_end = false);

    void ComputeAndOutputLogits(const Tensor& hidden_states, int first, int last);

    void OutputLogits(const Tensor& logits, int first, int last, GenerationConfig::OutType out_type);

    void OutputLastHiddenState(const Tensor& hidden_states, int first, int last);

    explicit LlamaBatch(DataType                 data_type,
                        const EngineParam&       param,
                        std::unique_ptr<LlamaV2> model,
                        std::unique_ptr<Context> ctx,
                        std::shared_ptr<Gateway> gateway,
                        int                      device_id,
                        int                      dp_rank);

    ~LlamaBatch();

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

private:
    void FindCanceledIndices(std::vector<int>& indices);

    void ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals);

    void InternalThreadEntry();

    void OutputThreadEntry();

    void CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc);

    void SwapState(BatchState*& a, BatchState*& b);

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

    void SendIntermediateData(IntermediateData& inter);

    void RecvIntermediateData(IntermediateData& inter);

    void PreProcessIntermediateData(IntermediateData& inter);

    void PostProcessIntermediateData(IntermediateData& inter);

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

    std::unique_ptr<Context>         context_;
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
    Tensor symm_residual_buf_;

    Tensor decoder_output_buf_;

    Tensor_<float> sampling_logits_;

    Buffer_<int> lora_mask_buf_;  // lora

    Buffer_<float>    sampled_logprobs_;
    Buffer_<uint32_t> sampled_indexes_;
    Buffer_<uint32_t> sampled_nums_;
    Buffer_<float>    h_sampled_logprobs_;
    Buffer_<uint32_t> h_sampled_indexes_;
    Buffer_<uint32_t> h_sampled_nums_;

    // pinned buffers
    Buffer_<int> h_output_ids_;

    Buffer_<int>       h_cu_block_counts_;
    Buffer_<uintptr_t> h_block_ptrs_;

    Buffer_<uint64_t> h_random_seed_;
    Buffer_<uint64_t> d_random_seed_;

    Tensor_<uint8_t> h_curand_state_;  // [n, sizeof(curandState_t)]
    Tensor_<uint8_t> d_curand_state_;

    std::vector<BatchState> states_;

    BatchState* state_{};
    BatchState* back_{};
    BatchState* incoming_{};

    // pipeline parallel
    std::deque<std::pair<BatchState*, GenerationState*>> slots_;
    std::queue<std::pair<BatchState*, GenerationState*>> batch_que_;
    std::vector<GenerationState>                         gs_;
    bool                                                 pp_abort_{false};

    // hard limits for persistent buffers
    static constexpr int kMaxStopBadWordsLen = 32;
    static constexpr int kMaxEndIdsSize      = 32;

    std::thread internal_thread_;
};

using Engine = LlamaBatch;

}  // namespace turbomind
