// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <cuda_runtime.h>
#include <random>
#include <vector>

namespace turbomind {

constexpr int kMoeGateMaxTiles = 16;
constexpr int kMoeGateVecSize  = 4;

void invokeMoeGate_V2(int*         f2n,
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

template<class T>
void invokeMoeGather(
    T* dst, const T* src, const int* f2n, int tokens, int experts_per_token, int dims, cudaStream_t st);

template<class T>
inline void
dispatchMoeGather(T* dst, const T* src, const int* f2n, int tokens, int experts_per_token, int dims, cudaStream_t st)
{
    const auto invoke = [&](auto type) {
        using V = decltype(type);
        invokeMoeGather((V*)dst, (const V*)src, f2n, tokens, experts_per_token, dims, st);
    };

    if constexpr (sizeof(T) == 2) {
        invoke(uint16_t{});
    }
    else {  /// TODO: dispatch for more types
        static_assert(sizeof(T) != sizeof(T), "Not implemented");
    }
}

template<class T>
void invokeMoeReduce(T*           dst,
                     const T*     src,
                     const float* scales,
                     const int*   en2f,
                     const float* dst_scales,
                     int          tokens,
                     int          experts_per_token,
                     int          dims,
                     float        dst_scale,
                     cudaStream_t st);

void invokeMoeSoftmaxMaskTopKGroups(
    float* logits, int token_num, int expert_num, int group_size, int top_k, cudaStream_t st);

// Sample `e` from `E` experts uniformly for every token
std::vector<int> SampleUniform(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

std::vector<int> SampleBalanced(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

}  // namespace turbomind
