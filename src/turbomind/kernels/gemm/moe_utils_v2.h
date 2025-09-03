// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

constexpr int kMoeGateMaxTiles = 16;
constexpr int kMoeGateVecSize  = 4;

void invokeMoeGate_V2(int*         f2n,
                      int*         f2E,
                      int*         en2f,
                      int*         offsets,
                      float*       scales,
                      void*        masks,
                      int*         accum,
                      const float* logits,
                      int          tokens,
                      int          tokens_padded,
                      int          experts,
                      int          exp_per_tok,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t st);

void invokeMoeDispatch(Ref<Tensor>   out_,  //
                       const Tensor& src,
                       const int*    f2n,
                       int           expert_per_token,
                       cudaStream_t  st);

void invokeMoeDispatchScales(Ref<Tensor>   out_,  //
                             const Tensor& src,
                             const int*    f2n,
                             int           expert_per_token,
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

// Sample `e` from `E` experts uniformly for every token
std::vector<int> SampleUniform(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

std::vector<int> SampleBalanced(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

}  // namespace turbomind
