// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"

#include <cstdint>

#include <cuda_runtime.h>
namespace turbomind {

[[nodiscard]] cudaError_t invokeGatherOutput(int*         output_ids,
                                             const int*   ids,
                                             const int*   context_length,
                                             int          max_context_len,
                                             int          max_gen_step,
                                             int          max_output_len,
                                             int          batch_size,
                                             cudaStream_t stream);

[[nodiscard]] cudaError_t invokeUpdateOutput(int**        request_output_ids_ptrs,
                                             int**        request_seqlen_ptrs,
                                             const int*   output_ids,
                                             const int*   sequence_lengths,
                                             const int*   request_output_ids_lens,
                                             int          max_session_len,
                                             bool         token_generated,
                                             int          batch_size,
                                             cudaStream_t stream);

// [aaa, bbbb, cc, ddd] -> [aaabbbbccddd]
[[nodiscard]] cudaError_t invokeCompactOutputIds(int*         cu_output_ids,
                                                 const int*   output_ids,
                                                 const int*   sequence_lengths,
                                                 int          max_session_len,
                                                 bool         token_generated,
                                                 int          batch_size,
                                                 cudaStream_t stream);

[[nodiscard]] cudaError_t invokeIndexedCopy(void**       h_src_ptr,
                                            void**       h_dst_ptr,
                                            const int*   h_elem_sz,
                                            const int*   h_src_idx,
                                            const int*   h_dst_idx,
                                            int          count,
                                            int          n_copys,
                                            cudaStream_t st);

[[nodiscard]] cudaError_t invokeBatchedCopy(void** src_ptr, void** dst_ptr, int* size, int count, cudaStream_t st);

// ABCDe            ABCDe     e
// ABCDEFGHIJk      ABCDEFGHIJk
// ABCDEFGHi    ->  ABCDEFGHi i
// ABCDEFGh         ABCDEFGh  h
// ABCd             ABCd      d
[[nodiscard]] cudaError_t invokePadLastTokenIds(
    int* token_ids, const int* context_length, int max_context_len, int batch_size, cudaStream_t stream);

[[nodiscard]] cudaError_t invokeGetFeatureOfLastToken(
    uint16_t* output, const uint16_t* input, const int* cu_seqlens, int dims, int batch_size, cudaStream_t stream);

template<typename T>
[[nodiscard]] cudaError_t invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream);

[[nodiscard]] cudaError_t invokeCastFloat2D(const core::Tensor& src, core::Tensor& dst, cudaStream_t stream);

void CollectHiddenStates(const Tensor& src, const Buffer_<int>& idxs, Ref<Tensor> dst, cudaStream_t st);

void BatchPrefixSum(const int** srcs, const int* ns, int** dsts, int count, cudaStream_t st);

inline void PrefixSum(const int* src, int n, int* dst, cudaStream_t st)
{
    return BatchPrefixSum(&src, &n, &dst, 1, st);
}

void AppendTokenIds(int**        token_ids_ptrs,  //
                    const int*   output_ids,
                    const int*   positions,
                    int          batch_size,
                    cudaStream_t stream);

// Apply sigmoid gating: attn[i] *= sigmoid(gate[i])
// attn:        [num_tokens, dim], contiguous
// gate_base:   pointer to first gate element in QKV buffer
// gate_stride: stride between tokens in QKV buffer (elements)
[[nodiscard]] cudaError_t invokeSigmoidGateMultiply(
    void* attn, const void* gate_base, int dim, int gate_stride, int num_tokens, DataType dtype, cudaStream_t stream);

}  // namespace turbomind
