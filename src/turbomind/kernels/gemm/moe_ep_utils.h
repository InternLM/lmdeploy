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
                           int            num_tokens,
                           int            topk,
                           int            num_local_experts,
                           cudaStream_t   stream);

// Add expert-specific bias to received expert outputs in-place for low latency combine.
void invokeMoeAddBias(Ref<Tensor> out, const Tensor& bias, const int* f2E, cudaStream_t st);

// Local reduce experts outputs before combine in EP mode(High throughput).
void invokeMoeLocalCombineEp(Ref<Tensor>   out,
                             const Tensor& src,
                             const Tensor& bias,
                             const float*  topk_weights,
                             const int*    en2f,
                             const int*    f2E,
                             int           experts_per_token,
                             cudaStream_t  st);

// Combine EP expert reduce result with shared expert output.
// output = output * shared_scale + src
// where shared_scale = sigmoid(shared_scales[ti]) if not null, else = scale.
void invokeMoeCombineOutputEp(
    Ref<Tensor> output, const Tensor& src, const float* shared_scales, float scale, cudaStream_t st);

void invokeMoeLLDispatchPostprocess(Tensor&       out,
                                    int*          f2n,
                                    int*          f2E,
                                    const int*    offsets,
                                    volatile int* moe_recv_counter,
                                    int*          moe_recv_counter_mapped,
                                    Tensor&       packed_recv_x,
                                    cudaStream_t  st);

void invokeMoeLLCombinePreprocess(Tensor& out, const Tensor& src, const int* offsets, cudaStream_t st);

}  // namespace turbomind
