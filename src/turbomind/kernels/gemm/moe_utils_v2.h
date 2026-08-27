// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

constexpr int kMoeGateMaxTiles = 16;
constexpr int kMoeGateVecSize  = 4;

// Select top-k experts and write token-major outputs for communication backends.
//
// logits:        [tokens, experts]
// token_mask:    [tokens]; invalid tokens route nowhere
// topk_weights:  [tokens, exp_per_tok]
// topk_indices:  [tokens, exp_per_tok], global expert ids
//
// The selected experts are written in ascending expert-id order. Each weight
// remains paired with the index at the same position.
void invokeMoeGate_V2(float*       topk_weights,
                      int*         topk_indices,
                      const float* logits,
                      const bool*  token_mask,
                      int          tokens,
                      int          experts,
                      int          exp_per_tok,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t st);

void invokeMoeGate_V2(int*         f2n,
                      int*         f2E,
                      int*         en2f,
                      int*         offsets,
                      float*       scales,
                      void*        masks,
                      int*         accum,
                      const float* logits,
                      const bool*  token_mask,  // [tokens]; invalid tokens route nowhere
                      int          tokens,
                      int          tokens_padded,
                      int          experts,
                      int          exp_per_tok,
                      int          local_expert_offset,
                      int          local_expert_num,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t st);

// num_worst_tokens is the output capacity / launch upper bound for f2n/out.
// If num_valid_tokens is set, it points to a device-side count of valid rows and
// rows >= *num_valid_tokens return before reading f2n.
void invokeMoeDispatch(Ref<Tensor>   out_,  //
                       const Tensor& src,
                       const int*    f2n,
                       int           num_worst_tokens,
                       const int*    num_valid_tokens,
                       cudaStream_t  st);

// Same num_worst_tokens / num_valid_tokens contract as invokeMoeDispatch
void invokeMoeDispatchScales(Ref<Tensor>   out_,  //
                             const Tensor& src,
                             const int*    f2n,
                             int           num_worst_tokens,
                             const int*    num_valid_tokens,
                             cudaStream_t  st);

void invokeMoeCombine(Ref<Tensor>   out_,
                      const Tensor& src,
                      const Tensor& bias,
                      const float*  scales,
                      const int*    en2f,
                      const int*    f2E,
                      const float*  dst_scales,
                      int           experts_per_token,
                      float         bscale,
                      float         dst_scale,
                      cudaStream_t  st);

void invokeMoeSoftmaxMaskTopKGroups(
    float* logits, int token_num, int expert_num, int group_size, int top_k, cudaStream_t st);

// Write token-major NoAuxTC outputs for communication backends.
//
// topk_weights:      [tokens, experts_per_token], float32
// topk_indices:      [tokens, experts_per_token], int32 global expert ids
// correction_bias:  [experts], float32, optional
//
// This interface performs global top-k selection (n_group == topk_group == 1).
// Selected experts are written in ascending expert-id order; weights and
// indices remain paired at the same position.
void invokeMoeGate_NoAuxTC(float*       topk_weights,
                           int*         topk_indices,
                           const float* logits,
                           const bool*  token_mask,
                           const float* correction_bias,
                           int          tokens,
                           int          experts,
                           int          experts_per_token,
                           bool         norm_topk,
                           float        routed_scale,
                           bool         use_sigmoid,
                           cudaStream_t stream);

/// noaux_tc routing: scores = scoring_func(logits), scores_for_choice = scores + correction_bias,
/// top-k on scores_for_choice, weights from scores; renormalize if norm_topk_prob; always apply routed_scale.
/// correction_bias may be nullptr (then treated as 0).
/// use_sigmoid: if true, scores = sigmoid(logits); if false, scores = softmax(logits).
void invokeMoeGate_NoAuxTC(int*         f2n,
                           int*         f2E,
                           int*         en2f,
                           int*         offsets,
                           float*       scales,
                           void*        masks,
                           int*         accum,
                           const float* logits,
                           const bool*  token_mask,  // [tokens]; invalid tokens route nowhere
                           const float* correction_bias,
                           int          tokens,
                           int          tokens_padded,
                           int          experts,
                           int          exp_per_tok,
                           int          local_expert_offset,
                           int          local_expert_num,
                           bool         norm_topk_prob,
                           float        routed_scale,
                           bool         use_sigmoid,
                           cudaStream_t st);

// Sample `e` from `E` experts uniformly for every token
std::vector<int> SampleUniform(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

std::vector<int> SampleBalanced(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

}  // namespace turbomind
