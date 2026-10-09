// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/generation/generation.h"
#include "src/turbomind/generation/target_verification.h"

namespace turbomind {

class LogitsProcessor;
class Sampling;
class StopCriteria;
class GuidedDecoding;

// Per-phase shared generation state: rows, random states, and the request to
// generation-row topology used by both the ordinary sampling lifecycle and the
// verification component.
struct GenerationData {
    Buffer_<uint64_t> random_seed;
    Buffer_<bool>     random_init;
    Buffer_<int>      random_state_indices;
    Buffer_<int*>     token_ids_ptrs;
    Buffer_<int>      request_to_generation_row_offsets;
    Buffer_<int>      output_ids;

    bool random_init_needed;
    int  request_count;
    int  generation_size;
};

// The generation-module state the verification component borrows. Reference
// bundle bound once at construction; the referenced members outlive it.
struct GenerationShared {
    std::unique_ptr<LogitsProcessor>& logits_processor;
    std::unique_ptr<Sampling>&        sampling;
    std::shared_ptr<StopCriteria>&    stop_criteria;

    Tensor_<uint8_t>& random_states;
    Tensor_<int>&     token_ids;
    const int&        max_batch_size;
    const int&        token_row_width;

    Buffer_<int*>& token_ids_ptrs_buf;

    std::vector<std::unique_ptr<GenerationData>>& data;

    int* RowPtr(int row) const
    {
        TM_CHECK_GE(row, 0);
        TM_CHECK_LT(row, max_batch_size);
        return token_ids.data() + row * token_ids.stride(0);
    }
};

struct Generation::Impl {

    // child modules
    std::unique_ptr<LogitsProcessor> logits_processor_;
    std::unique_ptr<Sampling>        sampling_;
    std::shared_ptr<StopCriteria>    stop_criteria_;
    std::unique_ptr<GuidedDecoding>  guided_decoding_;

    // persistent
    Tensor_<int>     token_ids_;
    Tensor_<uint8_t> random_states_;

    // scheduling states
    std::vector<int> free_token_rows_;
    std::vector<int> free_random_state_rows_;

    // immutable states
    Buffer_<int> output_ids_;

    std::vector<std::unique_ptr<GenerationData>> data_;

    // staging buffers
    Buffer_<uint64_t> random_seed_buf_;
    Buffer_<bool>     random_init_buf_;
    Buffer_<int>      random_state_indices_buf_;
    Buffer_<int*>     token_ids_ptrs_buf_;
    Buffer_<int>      token_ids_buf_;
    Buffer_<int>      output_ids_buf_;
    Buffer_<int>      request_to_generation_row_offsets_buf_;

    const int                     max_batch_size_;
    const int                     session_len_;
    const int                     token_row_width_;
    const SpeculativePolicy* const policy_;

    GenerationShared shared_;

    // Present only when a speculative policy is (the composed engine's
    // verification half); the ordinary lifecycle below never enters it.
    std::unique_ptr<TargetVerification> verification_;

    Impl(DataType              dtype,
         int                   max_batch_size,
         int                   session_len,
         int                   vocab_size,
         int                   vocab_size_padded,
         int                   hidden_units,
         DataType              hidden_dtype,
         const comm::HostComm& tp_group,
         int                   phases,
         const SpeculativePolicy* policy,
         bool                     enable_metrics);

    void Setup(int phase, TensorMap& env);
    void Del(TensorMap& env);

    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);
    void Update(int phase, TensorMap& env);

    // The ordinary forward: logits processing, sampling, and stop criteria.
    void Forward(int phase, TensorMap& env);
};

}  // namespace turbomind
