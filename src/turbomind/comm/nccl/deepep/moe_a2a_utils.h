#pragma once

#include "src/turbomind/core/core.h"

#include <cuda_runtime.h>

#include <vector>

namespace turbomind {

struct MoeA2AInputPartition {
    int begin;
    int size;
    int max_tokens_per_rank;
};

MoeA2AInputPartition GetMoeA2AInputPartition(const std::vector<int>& local_token_nums,  //
                                             int                     ep_size,
                                             int                     ep_rank,
                                             int                     mlp_tp_size);

int CeilPowerOfTwo(int x);

// Build the expert-major mappings consumed by the grouped MoE GEMMs.
//
// recv_topk_idx:      [recv_capacity, num_topk], token-major local expert ids; non-local entries are -1
// actual_recv_tokens: device scalar, number of valid rows in recv_topk_idx
// offsets:            [num_local_experts + 1], exclusive prefix sum of assignments per local expert
// expert_counters:    [num_local_experts], assignment counts; consumed to zero by CTA-level atomic reservations
//
// Only [0, offsets[num_local_experts]) is valid in f2n/f2E:
//   f2n[flat] = recv token row
//   f2E[flat] = local expert id
//
// en2f is fully initialized as [num_topk, recv_capacity]. Its first dimension is
// the top-k slot (not the expert id), and invalid entries are -1.
void invokeMoeA2AMapping(int*         f2n,
                         int*         f2E,
                         int*         en2f,
                         const int*   recv_topk_idx,
                         const int*   actual_recv_tokens,
                         const int*   offsets,
                         int*         expert_counters,
                         int          recv_capacity,
                         int          num_topk,
                         int          num_local_experts,
                         cudaStream_t stream);

// Merge the routed result with the shared FFN rows and write back to output.
// This call is ALSO the routed write-back: when `shared` is empty (nullptr)
// it degrades to output = routed, so callers must keep it unconditional.
//
//   output = routed + sigmoid(shared_scales) * shared        (shared present)
//   output = routed                                          (shared == nullptr)
//
// shared_scales is optional (gate-less shared experts); when null, the shared
// term carries weight 1.
void invokeMoeA2ASharedCombine(Tensor&       output,  //
                               const Tensor& routed,
                               const Tensor& shared,
                               const float*  shared_scales,
                               cudaStream_t  stream);

}  // namespace turbomind
