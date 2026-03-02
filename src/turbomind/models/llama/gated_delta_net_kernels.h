#pragma once

#include "src/turbomind/core/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// =============================================================================
// Causal Conv1d
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
// Avoids explicit transpose to channel-first format.
// For decode (seq_len==1): updates conv_states, applies depthwise conv + SiLU
// For prefill (seq_len>1): causal conv + SiLU, saves last d_conv inputs to conv_states
// in:  (token_num, conv_dim)  — row-major
// out: (token_num, conv_dim)  — row-major
// weight: (conv_dim, d_conv)  — depthwise weights
// conv_states: (batch_size, conv_dim, d_conv) — per-request rolling state, may be NULL
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
                           cudaStream_t stream);

// =============================================================================
// Gated Delta Rule — Recurrent (decode, seq_len == 1)
// =============================================================================

template<typename T>
void invokeRecurrentGatedDeltaRule(T*           v_out,
                                   const T*     q,
                                   const T*     k,
                                   const T*     v,
                                   const T*     beta,
                                   const T*     g,
                                   T*           recurrent_state,
                                   int          batch_size,
                                   int          num_v_heads,
                                   int          key_head_dim,
                                   int          value_head_dim,
                                   cudaStream_t stream);

// =============================================================================
// Gated Delta Rule — Serial prefill (seq_len > 1)
// =============================================================================

template<typename T>
void invokeSerialGatedDeltaRule(T*           v_out,
                                const T*     q,
                                const T*     k,
                                const T*     v,
                                const T*     beta,
                                const T*     g,
                                T*           recurrent_state,
                                int          batch_size,
                                int          seq_len,
                                int          num_v_heads,
                                int          key_head_dim,
                                int          value_head_dim,
                                cudaStream_t stream);

// =============================================================================
// Helper kernels for GatedDeltaNetLayer forward pass
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

// L2-normalize along the last dimension
template<typename T>
void invokeL2Norm(T* x, int outer, int head_dim, cudaStream_t stream);

// Repeat-interleave from num_k_heads to num_v_heads
template<typename T>
void invokeRepeatInterleave(
    T* dst, const T* src, int total, int num_k_heads, int num_v_heads, int head_dim, cudaStream_t stream);

// RMSNorm * SiLU-gate (fused output normalization)
template<typename T>
void invokeRMSNormGated(T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim, cudaStream_t stream);

// Element-wise SiLU activation in-place
template<typename T>
void invokeSiLU(T* data, int n, cudaStream_t stream);

}  // namespace turbomind
