// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/Barrier.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/instance_comm.h"
#include <condition_variable>
#include <curand_kernel.h>
#include <mutex>
#include <type_traits>

using ffi_api_lock_ctrl_t = std::function<void(int)>;

namespace turbomind {

struct SharedState {
    std::vector<std::shared_ptr<Request>> infer_requests;
    std::vector<std::shared_ptr<Request>> stop_requests;
    RequestQueue                          request_queue;
    std::shared_ptr<Barrier>              barrier;
    bool                                  abort;
    std::atomic<size_t>                   free_size{std::numeric_limits<size_t>::max()};
};

struct Control {
    AbstractInstanceComm* comm;
    Request::Callback     callback;
};

struct BatchState {
    int*  h_prompt_length;  // history + input, ignore generated
    int*  h_context_length;
    bool* h_finished;

    curandState_t* curand_state;
    int*           output_ids;  // output ids in [B, S]

    float* h_rope_theta;

    std::vector<int> seq_len_limit;

    std::vector<const Sequence*>          sequences;
    std::vector<std::shared_ptr<Request>> requests;

    std::vector<int> errors;

    // |<-- existing -->|<-- swap-in -->|
    // |<----------- active ----------->|<-- inactive -->|
    int active_size;
    int size;
};

template<typename T>
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

template<typename T>
class LlamaBatch {
public:
    void AllocateBuffer(size_t batch_size, size_t session_len, int cache_block_seq_len);
    void AllocatePersistantBuffer(size_t max_batch_size, int cache_block_seq_len);
    void FreeBuffer();

    using Requests = std::vector<std::shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    void RejectInvalidRequests(Requests& stop_reqs, Requests& infer_reqs);

    [[nodiscard]] auto ProcessStopRequests(const Requests& requests) -> std::vector<Signal>;

    void ProcessInferRequests(const Requests& requests);

    int AdjustMaxInputCount(GenerationState&                    g,
                            const std::vector<const Sequence*>& sequences,
                            const std::vector<int>&             context_length);

    void Initialize(GenerationState& g);

    void InitializeSampling(const GenerationState& g);

    [[nodiscard]] bool Forward(GenerationState& g);

    [[nodiscard]] auto Finish(GenerationState& g) -> std::vector<Signal>;

    [[nodiscard]] Signal Interrupt(int index, bool force_stop = false, bool force_end = false);

    void OutputContextLogits(T*                                  context_decoder_output,
                             const std::vector<int>&             indices,
                             const std::vector<int>&             lengths,
                             const std::vector<const Sequence*>& sequences);

    explicit LlamaBatch(const EngineParam&           param,
                        std::unique_ptr<LlamaV2<T>>  model,
                        std::unique_ptr<Context<T>>  ctx,
                        std::shared_ptr<SharedState> state,
                        int                          device_id);

    ~LlamaBatch();

    void Start();

    void Submit(std::unordered_map<std::string, Tensor>*       outputs,
                const std::unordered_map<std::string, Tensor>* inputs,
                Control                                        control);

    void set_ffi_lock(ffi_api_lock_ctrl_t func)
    {
        ffi_lock_ = func;
    }

    LlamaV2<T>& model() noexcept
    {
        return *model_;
    }

    int session_len() const noexcept
    {
        return session_len_;
    }

private:
    void InternalThreadEntry();

    void OutputThreadEntry();

    void CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc);

    void SendSignals(std::vector<Signal> signals);

    // analogs to `std::copy_n`
    template<typename U>
    U* Copy(const U* src, size_t count, U* dst)
    {
        check_cuda_error(cudaMemcpyAsync(dst, src, sizeof(U) * count, cudaMemcpyDefault, stream_));
        return dst += count;
    }

    template<typename U>
    U* Clear(U* data, size_t count)
    {
        check_cuda_error(cudaMemsetAsync(data, 0, sizeof(U) * count, stream_));
        return data += count;
    }

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

private:
    const EngineParam param_;

    const std::shared_ptr<SharedState> shared_state_;

    const int      max_batch_size_;
    const int      max_forward_token_num_;
    const int      max_context_token_num_;
    const int      num_tokens_per_iter_;
    const int      max_prefill_iters_;
    const int      device_id_;
    const int      rank_;
    const DataType data_type_;
    const bool     debug_;

    // Refs into `Context<T>`
    cudaStream_t const     stream_{};
    cublasMMWrapper* const cublas_wrapper_{};
    IAllocator* const      allocator_{};
    IAllocator* const      peer_allocator_{};

    int session_len_;  // May be truncated in ctor

    std::unique_ptr<Context<T>>      context_;
    std::unique_ptr<LlamaV2<T>>      model_;
    std::unique_ptr<SequenceManager> sequence_manager_;

    ///////////////////////////////////////////////////////////////////
    // k/v cache block buffers
    int*       cu_block_counts_{};
    uintptr_t* block_ptrs_{};

    ////////////////////////////////////////////////////////////////////
    // context decoding temp buffers
    T*   context_decoder_input_buf_{};
    T*   context_decoder_output_buf_{};
    int* context_decoder_ids_buf_{};
    int* input_ids_buf_{};
    // lengths
    int* input_length_buf_{};    // input + cache missed length
    int* context_length_buf_{};  // history length + input_length
    int* init_context_length_{};

    T*   decoder_input_buf_{};
    T*   decoder_output_buf_{};
    int* sequence_lengths_{};  // current sequence length
    int* init_ctx_lens_{};
    int* lora_mask_buf_{};  // lora

    float* logits_buf_{};        // combined logits
    float* local_logits_buf_{};  // tensor parallel local logits
    float* context_logits_buf_{};
    float* local_context_logits_buf_{};

    float*    sampled_logprobs_{};
    uint32_t* sampled_indexes_{};
    uint32_t* sampled_nums_{};
    float*    h_sampled_logprobs_{};
    uint32_t* h_sampled_indexes_{};
    uint32_t* h_sampled_nums_{};

    float* rope_theta_{};

    // used by dynamic decoder
    int*      token_ids_buf_{};  // all token IDs in [S, B], indexed using `step`
    bool*     finished_buf_{};
    uint32_t* seq_limit_len_{};
    int*      h_end_ids_buf_{};
    int*      d_end_ids_buf_{};

    // pinned buffers
    int*       h_input_ids_buf_{};
    int*       h_input_length_buf_{};
    uint32_t*  h_seq_limit_len_{};
    int*       h_cu_block_counts_{};
    uintptr_t* h_block_ptrs_{};

    int*   h_min_length_{};
    int*   h_runtime_top_k_{};
    float* h_runtime_top_p_{};
    float* h_runtime_min_p_{};
    float* h_temperature_{};
    float* h_repetition_penalty_{};
    int*   h_stop_words_{};  // [batch_size, 2, kMaxStopWordsLen]
    int*   h_bad_words_{};
    int*   d_stop_words_{};  // [batch_size, 2, kMaxStopWordsLen]
    int*   d_bad_words_{};

    unsigned long long* h_random_seed_{};
    unsigned long long* d_random_seed_{};

    curandState_t* h_curand_state_{};
    curandState_t* d_curand_state_{};

    std::array<BatchState, 3> states_{};

    BatchState* state_{};
    BatchState* back_{};
    BatchState* incoming_{};

    uint64_t request_count_{0};

    // hard limits for persistent buffers
    static constexpr int kMaxStopBadWordsLen = 32;

    bool is_allocate_persistant_buffer_ = false;
    bool is_allocate_buffer_            = false;

    TensorMap inputs_;
    TensorMap outputs_;

    std::vector<std::tuple<std::string, std::byte*, std::byte*>> sampling_params_;

    std::thread internal_thread_;

    // async stream callback utils
    std::thread             output_thread_;
    std::mutex              output_mutex_;
    std::condition_variable output_cv_;
    std::vector<Signal>     output_signals_;
    bool                    output_stop_token_{false};
    ffi_api_lock_ctrl_t     ffi_lock_;

    int* h_output_ids_{};
};

template<class T>
using Engine = LlamaBatch<T>;

}  // namespace turbomind
