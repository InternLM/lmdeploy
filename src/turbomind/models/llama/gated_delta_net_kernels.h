#pragma once

#include "src/turbomind/core/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

// Fused Conv1d + SiLU for row-major (token_num, conv_dim) layout.
// in:  (token_num, conv_dim)  — row-major
// out: (token_num, conv_dim)  — row-major
// weight: (conv_dim, d_conv)
// conv_states: (conv_dim, d_conv) per-request rolling state, may be NULL
template<typename T>
void invokeFusedConv1dSiLU(T*           out,
                           const T*     in,
                           const T*     weight,
                           const T*     bias,
                           T*           conv_states,
                           int          batch_size,
                           int          conv_dim,
                           int          seq_len,
                           int          d_conv,
                           int          in_stride,
                           cudaStream_t stream);

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

// Fused variant: computes beta/g from raw b/a logits + A_log/dt_bias
// inline, caches state in shared memory, eliminates 3 kernel launches.
// b_raw/a_raw are strided with ba_stride columns between tokens.
template<typename T>
void invokeRecurrentGatedDeltaRuleFused(T*           v_out,
                                         const T*     qkv_in,
                                         const T*     b_raw,
                                         const T*     a_raw,
                                         const T*     A_log,
                                         const T*     dt_bias,
                                         T*           recurrent_state,
                                         int          batch_size,
                                         int          num_v_heads,
                                         int          num_k_heads,
                                         int          key_head_dim,
                                         int          value_head_dim,
                                         int          k_dim_total,
                                         int          ba_stride,
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
