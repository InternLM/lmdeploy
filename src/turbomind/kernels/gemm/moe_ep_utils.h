// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeMoeGateEp(float*       topk_weights,
                     int64_t*     topk_idx,
                     const float* logits,
                     int          tokens,
                     int          experts,
                     int          experts_per_token,
                     bool         softmax,
                     bool         norm_topk,
                     float        routed_scale,
                     cudaStream_t stream);

// Compute f2n, f2E, en2f mappings from recv_topk_idx after EP dispatch.
void invokeMoeRoutingMapEp(int*           f2n,
                           int*           f2E,
                           int*           en2f,
                           int*           offsets,
                           const int64_t* recv_topk_idx,
                           const int*     total_tokens_ptr,
                           int            num_tokens,
                           int            topk,
                           int            num_local_experts,
                           cudaStream_t   stream);

// Add expert-specific bias to received expert outputs in-place for low latency combine.
void invokeMoeAddBias(
    Ref<Tensor> out, const Tensor& bias, const int* f2E, const int* total_tokens_ptr, cudaStream_t st);

// Local reduce experts outputs before combine in EP mode(High throughput).
// `out.shape(0)` is an upper bound on received tokens; when `total_tokens_ptr` is
// non-null the kernel caps its grid-stride loop at the real total, so padding
// rows cost no per-token work.
void invokeMoeLocalCombineEp(Ref<Tensor>   out,
                             const Tensor& src,
                             const Tensor& bias,
                             const float*  topk_weights,
                             const int*    en2f,
                             const int*    f2E,
                             int           experts_per_token,
                             const int*    total_tokens_ptr,
                             cudaStream_t  st);

// Combine EP expert reduce result with shared expert output.
// output = output * shared_scale + src
// where shared_scale = sigmoid(shared_scales[ti]) if not null, else = scale.
void invokeMoeCombineOutputEp(
    Ref<Tensor> output, const Tensor& src, const float* shared_scales, float scale, cudaStream_t st);

// Build `f2n` and `f2E` mappings from device-side `offsets`.
void invokeMoeLLDispatchPostprocess(
    int* f2n, int* f2E, const int* offsets, const Tensor& packed_recv_x, cudaStream_t st);

// Reorder sparse LL dispatch scales from [E, H/128, max_T] contiguous (deep_ep
// layout) to [H/128, E*max_T] contiguous (the layout expected by
// invokeMoeDispatchScales). Only the valid [0, count_e) prefix of each expert
// is written; gap slots are untouched.
void invokeMoeLLDispatchScalesLayoutConvert(Tensor&       target,
                                            const Tensor& packed_recv_x_scales,
                                            const Tensor& packed_recv_count,
                                            cudaStream_t  st);

}  // namespace turbomind
