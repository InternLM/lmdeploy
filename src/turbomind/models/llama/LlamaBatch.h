// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaCacheManager.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"

namespace turbomind {

template<typename T>
class LlamaV2;

template<typename T>
class LlamaBatch {
public:
    int size() const noexcept
    {
        return batch_size_;
    };

    int maxSize() const noexcept
    {
        return max_batch_size_;
    }

    int finishedCount() const noexcept
    {
        return finished_count_;
    }

    void verifyRequests(std::vector<std::shared_ptr<Request>>& stop_reqs,
                        std::vector<std::shared_ptr<Request>>& infer_reqs);
    void handleStopRequests(const std::vector<std::shared_ptr<Request>>& requests);

    void allocateBuffer(size_t batch_size, size_t session_len);
    void allocatePersistantBuffer(size_t max_batch_size);
    void freeBuffer();

    void initializeSampling(int infer_request_count);

    void initialize(const std::vector<std::shared_ptr<Request>>& infer_requests);
    void contextDecode();

    void initializeGeneration();
    bool generate();

    void finish();
    void finishRequest(int index, bool force_end);

    void synchronize();

    void setOutputTensors(int max_gen_step);

    void
    outputContextLogits(T* context_decoder_output, const std::vector<int>& indices, const std::vector<int>& lengths);

    explicit LlamaBatch(int max_batch_size, int max_context_token_num, int session_len, LlamaV2<T>* llama);

    ~LlamaBatch()
    {
        freeBuffer();
    }

private:
    const int  max_batch_size_;
    const int  max_context_token_num_;
    const int  session_len_;
    const int  rank_;
    const bool debug_;

    LlamaV2<T>* const llama_;

    // active requests
    std::vector<std::shared_ptr<Request>> requests_;

    T*   context_decoder_input_buf_{};   // CTXDEC
    T*   context_decoder_output_buf_{};  // CTXDEC
    int* context_decoder_ids_buf_{};

    T* decoder_input_buf_{};   // CTXDEC, GENERATE
    T* decoder_output_buf_{};  // CTXDEC, GENERATE

    int* input_ids_buf_{};       // input token ids + cache missed token ids, CTXDEC
    int* input_length_buf_{};    // input + cache missed length, CTXDEC, GENERATE
    int* history_length_buf_{};  // history length, CTXDEC
    int* context_length_buf_{};  // history length + input_length, CTXDEC, GENERATE

    int* total_padding_count_{};  // GENERATE
    int* sequence_lengths_{};     // current sequence length

    uint64_t* k_cache_ptr_buf_{};
    uint64_t* v_cache_ptr_buf_{};

    float* logits_buf_{};        // combined logits
    float* local_logits_buf_{};  // tensor parallel local logits
    float* context_logits_buf_{};
    float* local_context_logits_buf_{};

    // used by dynamic decoder
    int*      token_ids_buf_{};   // all token IDs in [S, B], indexed using `step`
    int*      output_ids_buf_{};  // output ids in [B, S]
    int*      end_ids_buf_{};
    bool*     finished_buf_{};
    uint32_t* seq_limit_len_{};

    // pinned buffers
    int*       h_input_ids_buf_{};
    int*       h_input_length_buf_{};
    int*       h_history_length_buf_{};
    int*       h_context_length_buf_{};
    int*       h_sequence_lengths_{};
    bool*      h_finished_buf_{};
    uintptr_t* h_k_cache_ptr_buf_{};
    uintptr_t* h_v_cache_ptr_buf_{};
    uint32_t*  h_seq_limit_len_{};

    int*      stop_words_buf_{};  // [batch_size, 2, kMaxStopWordsLen]
    int*      bad_words_buf_{};
    int*      h_runtime_top_k_{};
    float*    h_runtime_top_p_{};
    float*    h_temperature_{};
    float*    h_repetition_penalty_{};
    uint64_t* h_random_seed_{};

    void* topk_curandstate_buf_{};
    void* topp_curandstate_buf_{};

    // hard limits for persistent buffers
    static constexpr int kMaxStopBadWordsLen = 32;

    using CachedSeq = LlamaCacheManager::Sequence;

    std::vector<CachedSeq> cached_seq_;
    std::vector<int>       request_seq_len_limit_;

    const DataType data_type_{};

    int batch_size_{};
    int max_context_len_{};
    int step_{};
    int finished_count_{};

    bool is_allocate_persistant_buffer_ = false;
    bool is_allocate_buffer_            = false;

    TensorMap inputs_;
    TensorMap outputs_;

    std::unordered_map<std::string, void*> sampling_params_;

    cudaStream_t     stream_{};
    cublasMMWrapper* cublas_wrapper_{};
    IAllocator*      allocator_{};
};

}  // namespace turbomind
