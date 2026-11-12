#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace turbomind {

struct VerifyTargetBlockParams {
    const float* probabilities;
    int          probability_stride;
    const int*   probability_token_ids;
    int          token_id_stride;
    const int*   kept_count;
    const int*   verification_draft_ids;
    int          draft_row_stride;
    const bool*  greedy;
    const bool*  logits_active;

    curandState_t* random_states;
    const int*     random_state_indices;

    int* const* request_token_ids_ptrs;
    const int* entry_sequence_length;
    const int* request_to_generation_offsets;
    const bool* speculative_row;

    int* selected_span_ids;
    int  selected_span_stride;
    int* accept_len;
    int* accepted_draft_count;

    int request_count;
    int generation_count;
    int position_count;
};

void invokeVerifyTargetBlock(const VerifyTargetBlockParams& params, cudaStream_t stream);

}  // namespace turbomind
