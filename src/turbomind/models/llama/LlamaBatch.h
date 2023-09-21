// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

// #include "src/turbomind/models/llama/LlamaCacheManager.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"

namespace turbomind {

struct BatchState {
    int*  h_context_length;
    bool* h_finished;

    void* top_k_curand_state;
    void* top_p_curand_state;
    int*  output_ids;  // output ids in [B, S]

    std::vector<int> seq_len_limit;

    std::vector<const Sequence*>          sequences;
    std::vector<std::shared_ptr<Request>> requests;

    // |<-- existing -->|<-- swap-in -->|<-- inactive -->|
    int size;
    int active_size;
};

template<typename T>
class LlamaV2;

template<typename T>
class LlamaBatch {
public:
    void AllocateBuffer(size_t batch_size, size_t session_len);
    void AllocatePersistantBuffer(size_t max_batch_size);
    void FreeBuffer();

    using Requests = std::vector<std::shared_ptr<Request>>;

    void RejectInvalidRequests(Requests& stop_reqs, Requests& infer_reqs);

    void ProcessStopRequests(const Requests& requests);

    void ProcessInferRequests(const Requests& requests);

    bool Initialize();

    void ContextDecode();

    void InitializeSampling();
    void InitializeGeneration();
    bool Generate();

    int  Finish();
    void FinishRequest(int index, bool force_end);

    void SetOutputTensors(int max_gen_step);

    void
    OutputContextLogits(T* context_decoder_output, const std::vector<int>& indices, const std::vector<int>& lengths);

    explicit LlamaBatch(int                              max_batch_size,
                        int                              max_context_token_num,
                        int                              session_len,
                        std::unique_ptr<SequenceManager> sequence_manager,
                        LlamaV2<T>*                      llama);

    ~LlamaBatch()
    {
        llama_->shared_state_->request_queue.Abort();

        internal_thread_.join();

        FreeBuffer();
    }

    void Start();

private:
    void InternalThreadEntry(int device_id);

    void UpdateSequenceStates(BatchState& state, int index);

    void CopyState(const std::pair<BatchState*, int> _src, const std::pair<BatchState*, int>& _dst);

    void SaveRandomState(BatchState& state, int idx);

    void LoadRandomState(BatchState& state, int idx);

    // analogs to `std::copy_n`
    template<typename U>
    U* Copy(const U* src, size_t count, U* dst)
    {
        check_cuda_error(cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDefault, stream_));
        return dst += count;
    }

    template<typename U>
    U* Clear(U* data, size_t count)
    {
        check_cuda_error(cudaMemsetAsync(data, 0, sizeof(U) * count, stream_));
        return data += count;
    }

private:
    const int  max_batch_size_;
    const int  max_context_token_num_;
    const int  session_len_;
    const int  rank_;
    const bool debug_;
    const int  step_length_;

    LlamaV2<T>* const llama_;

    std::unique_ptr<SequenceManager> sequence_manager_;

    T*   context_decoder_input_buf_{};   // CTXDEC
    T*   context_decoder_output_buf_{};  // CTXDEC
    int* context_decoder_ids_buf_{};

    T* decoder_input_buf_{};   // CTXDEC, GENERATE
    T* decoder_output_buf_{};  // CTXDEC, GENERATE

    int*       input_ids_buf_{};       // input token ids + cache missed token ids, CTXDEC
    int*       input_length_buf_{};    // input + cache missed length, CTXDEC, GENERATE
    int*       history_length_buf_{};  // history length, CTXDEC
    int*       context_length_buf_{};  // history length + input_length, CTXDEC, GENERATE
    int*       sequence_lengths_{};    // current sequence length
    int*       cu_block_counts_{};
    uintptr_t* k_block_ptrs_{};
    uintptr_t* v_block_ptrs_{};

    float* logits_buf_{};        // combined logits
    float* local_logits_buf_{};  // tensor parallel local logits
    float* context_logits_buf_{};
    float* local_context_logits_buf_{};

    // used by dynamic decoder
    int*      token_ids_buf_{};  // all token IDs in [S, B], indexed using `step`
    int*      end_ids_buf_{};
    bool*     finished_buf_{};
    uint32_t* seq_limit_len_{};

    // pinned buffers
    int*       h_input_ids_buf_{};
    int*       h_input_length_buf_{};
    int*       h_history_length_buf_{};
    int*       h_sequence_lengths_{};
    uint32_t*  h_seq_limit_len_{};
    int*       h_cu_block_counts_{};
    uintptr_t* h_k_block_ptrs_{};
    uintptr_t* h_v_block_ptrs_{};

    int*      stop_words_buf_{};  // [batch_size, 2, kMaxStopWordsLen]
    int*      bad_words_buf_{};
    int*      h_runtime_top_k_{};
    float*    h_runtime_top_p_{};
    float*    h_temperature_{};
    float*    h_repetition_penalty_{};
    uint64_t* h_random_seed_{};

    BatchState states_[3];

    BatchState* state_{};
    BatchState* back_{};
    BatchState* incoming_{};

    uint64_t request_count_{0};

    // hard limits for persistent buffers
    static constexpr int kMaxStopBadWordsLen = 32;

    const DataType data_type_{};

    int max_context_len_{};
    int step_{};

    bool is_allocate_persistant_buffer_ = false;
    bool is_allocate_buffer_            = false;

    TensorMap inputs_;
    TensorMap outputs_;

    std::unordered_map<std::string, void*> sampling_params_;

    cudaStream_t     stream_{};
    cublasMMWrapper* cublas_wrapper_{};
    IAllocator*      allocator_{};

    std::thread internal_thread_;
};

}  // namespace turbomind
