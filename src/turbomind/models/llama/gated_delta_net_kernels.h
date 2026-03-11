#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

// =============================================================================
// Causal Conv1d (channel-first layout, for backward compat)
// =============================================================================
template<typename T>
void invokeCausalConv1d(T*           out,
                        const T*     in,
                        const T*     weight,
                        const T*     bias,
                        T*           conv_states,
                        int          batch_size,
                        int          conv_dim,
                        int          seq_len,
                        int          d_conv,
                        cudaStream_t stream);

// Fused Conv1d + SiLU — unified batched launcher (row-major layout).
//
// Processes all requests in a single kernel launch.  Decode (seq_len == 1)
// and prefill (seq_len > 1) requests may be mixed freely within the batch.
//
// out:             (total_tokens, conv_dim)       row-major output
// in:              (total_tokens, in_stride)      non-contiguous slice of all_proj
// weight:          (conv_dim, d_conv)
// bias:            (conv_dim) or empty Tensor
// conv_state_ptrs: device array[batch_size] of per-request state pointers
// q_offsets:       device int[batch_size+1] cumulative token offsets
void invokeFusedConv1dSiLU(Ref<Tensor>           out,
                            const Tensor&         in,
                            const Tensor&         weight,
                            const Tensor&         bias,
                            const Buffer_<void*>& conv_state_ptrs,
                            const Buffer_<int>&   q_offsets,
                            int                   batch_size,
                            int                   state_layer_offset,
                            cudaStream_t          stream);

// =============================================================================
// Gated Delta Rule — Recurrent decode (seq_len == 1)
//
// Reads Q/K/V directly from the packed qkv_in buffer (stride = conv_dim),
// L2-normalizes Q and K in-kernel, and handles GQA natively (kh = h / ratio).
// Eliminates separate invokeL2Norm and invokeRepeatInterleave passes.
//
// qkv_in layout per token (row-major):
//   [Q: (num_k_heads, key_head_dim) | K: (num_k_heads, key_head_dim)
//    | V: (num_v_heads, value_head_dim)]
// where k_dim_total = num_k_heads * key_head_dim.
// =============================================================================
template<typename T>
void invokeRecurrentGatedDeltaRule(T*           v_out,
                                   const T*     qkv_in,
                                   const T*     beta,
                                   const T*     g,
                                   T*           recurrent_state,
                                   int          batch_size,
                                   int          num_v_heads,
                                   int          num_k_heads,
                                   int          key_head_dim,
                                   int          value_head_dim,
                                   int          k_dim_total,
                                   cudaStream_t stream);

// =============================================================================
// Gated Delta Rule — Single-launch Prefill (seq_len > 1)
//
// Processes the ENTIRE sequence inside a single kernel launch.
// The original invokeSerialGatedDeltaRule called the decode kernel in a
// host-side for-loop (one CUDA launch per timestep = O(seq_len) dispatches).
// This kernel eliminates that overhead by looping over seq_len on the GPU.
// Also fuses L2Norm and GQA (same as the recurrent decode variant).
//
// v_out layout: (seq_len, num_v_heads, value_head_dim).
// qkv_in layout: (seq_len, conv_dim) packed as described above.
// state  layout: (num_v_heads, key_head_dim, value_head_dim) — updated in-place.
// =============================================================================
template<typename T>
void invokeGatedDeltaRulePrefill(T*           v_out,
                                 const T*     qkv_in,
                                 const T*     beta,
                                 const T*     g,
                                 T*           recurrent_state,
                                 int          seq_len,
                                 int          num_v_heads,
                                 int          num_k_heads,
                                 int          key_head_dim,
                                 int          value_head_dim,
                                 int          k_dim_total,
                                 cudaStream_t stream);

// =============================================================================
// Gated Delta Rule — Unified batched launcher (decode + prefill)
//
// Replaces per-request invokeRecurrentGatedDeltaRule / invokeGatedDeltaRulePrefill
// calls with a single launch across all requests in the batch.
//
// v_out:       (total_tokens, num_v_heads * value_head_dim)
// qkv_in:      (total_tokens, conv_dim)  packed conv output
// beta, g:     (total_tokens, num_v_heads)
// state_ptrs:  device array[batch_size] of per-request recurrent state pointers
// q_offsets:   device int[batch_size+1] cumulative token offsets
void invokeGatedDeltaRuleBatched(Ref<Tensor>           v_out,
                                  const Tensor&         qkv_in,
                                  const Tensor&         beta,
                                  const Tensor&         g,
                                  const Buffer_<void*>& state_ptrs,
                                  const Buffer_<int>&   q_offsets,
                                  int                   batch_size,
                                  int                   num_k_heads,
                                  int                   key_head_dim,
                                  int                   state_layer_offset,
                                  cudaStream_t          stream);

// =============================================================================
// Helper kernels
// =============================================================================

// Compute beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias)
template<typename T>
void invokeComputeBetaG(T*           beta_out,
                        T*           g_out,
                        const T*     b_in,
                        const T*     a_in,
                        const T*     A_log,
                        const T*     dt_bias,
                        int          total,
                        int          num_v_heads,
                        cudaStream_t stream);

void ComputeBetaG_v2(Ref<Tensor>   beta_out_,
                     Ref<Tensor>   g_out_,
                     const Tensor& b_in,
                     const Tensor& a_in,
                     const Tensor& A_log,
                     const Tensor& dt_bias,
                     cudaStream_t  stream);

// RMSNorm * SiLU-gate (fused output normalization)
template<typename T>
void invokeRMSNormGated(T*           hidden,
                        const T*     gate,
                        const T*     weight,
                        float        eps,
                        int          N,
                        int          head_dim,
                        int          gate_stride,
                        int          num_heads,
                        cudaStream_t stream);

// Element-wise SiLU activation in-place
template<typename T>
void invokeSiLU(T* data, int n, cudaStream_t stream);

}  // namespace turbomind
