#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

// Fused Conv1d + SiLU — unified batched launcher (row-major layout).
//
// Processes all requests in a single kernel launch.  Decode (seq_len == 1)
// and prefill (seq_len > 1) requests may be mixed freely within the batch.
//
// out:             (total_tokens, conv_dim)       row-major output
// in:              (total_tokens, in_stride)      non-contiguous slice of all_proj
// weight:          (d_conv, conv_dim)
// bias:            (conv_dim) or empty Tensor
// conv_state_ptrs: device array[batch_size] of per-request state pointers
// q_offsets:       device int[batch_size+1] cumulative token offsets
// k_offsets:       device int[batch_size+1] cumulative key (history+input) offsets
void invokeFusedConv1dSiLU(Ref<Tensor>           out,
                           const Tensor&         in,
                           const Tensor&         weight,
                           const Tensor&         bias,
                           const Buffer_<void*>& conv_state_ptrs,
                           const Buffer_<int>&   q_offsets,
                           const Buffer_<int>&   k_offsets,
                           const Buffer_<bool>&  finished,
                           int                   batch_size,
                           int                   state_layer_offset,
                           int                   sm_count,
                           int*                  work_counter,
                           cudaStream_t          stream);

inline void invokeFusedConv1dSiLU(Ref<Tensor>           out,
                                  const Tensor&         in,
                                  const Tensor&         weight,
                                  const Tensor&         bias,
                                  const Buffer_<void*>& conv_state_ptrs,
                                  const Buffer_<int>&   q_offsets,
                                  const Buffer_<int>&   k_offsets,
                                  int                   batch_size,
                                  int                   state_layer_offset,
                                  int                   sm_count,
                                  int*                  work_counter,
                                  cudaStream_t          stream)
{
    invokeFusedConv1dSiLU(out,
                          in,
                          weight,
                          bias,
                          conv_state_ptrs,
                          q_offsets,
                          k_offsets,
                          Buffer_<bool>{},
                          batch_size,
                          state_layer_offset,
                          sm_count,
                          work_counter,
                          stream);
}

// =============================================================================
// Helper kernels
// =============================================================================

void ComputeBetaG(core::Tensor&       beta,
                  core::Tensor&       g,
                  const core::Tensor& b,
                  const core::Tensor& a,
                  const core::Tensor& A_log,
                  const core::Tensor& dt_bias,
                  cudaStream_t        stream);

void invokeL2NormalizeQK(core::Tensor& q, core::Tensor& k, float epsilon, cudaStream_t stream);

// RMSNorm * SiLU-gate (fused output normalization)
void invokeRMSNormGated(Ref<Tensor> hidden, const Tensor& gate, const Tensor& weight, float eps, cudaStream_t stream);

}  // namespace turbomind
