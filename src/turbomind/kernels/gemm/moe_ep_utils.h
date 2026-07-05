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

void invokeMoeBuildAgRsRoutingMap(int*           f2n,
                                  int*           f2E,
                                  int*           en2f,
                                  int*           offsets,
                                  int8_t*        masks,
                                  int*           accum,
                                  const int64_t* topk_idx,
                                  int            tokens,
                                  int            topk,
                                  int            local_expert_offset,
                                  int            num_local_experts,
                                  cudaStream_t   stream);

void invokeMoeLocalCombineEp(Ref<Tensor>   out,
                             const Tensor& src,
                             const Tensor& bias,
                             const float*  topk_weights,
                             const int*    en2f,
                             const int*    f2E,
                             int           experts_per_token,
                             cudaStream_t  st);

void invokeMoeCombineOutputEp(
    Ref<Tensor> output, const Tensor& shared, const float* shared_scales, float scale, cudaStream_t st);

}  // namespace turbomind
