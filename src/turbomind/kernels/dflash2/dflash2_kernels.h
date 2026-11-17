// Copyright (c) OpenMMLab. All rights reserved.
//
// DFlash2 draft kernels: grouped conv and candidate selector.
#pragma once

#include <cuda_runtime.h>

namespace turbomind {

/// Content-generated grouped depthwise conv (DFlash2 attention_conv / mlp_conv).
///
///   out[t, h] = sum_tap coeff[t, tap, g] * hidden[t - tap, h]
///   coeff[t, tap, g] = base[tap, h] + delta[t, tap, g],  g = h / group_size
///   term included only for tap == 0, or t - tap >= 0 and (t % block_size) >= tap
///
/// `base` is [taps, hidden], `delta` is [rows, taps, num_groups].
void invokeGroupedConv(void*        out,
                       const void*  hidden,
                       const void*  delta,
                       const void*  base,
                       int          rows,
                       int          hidden_size,
                       int          num_groups,
                       int          group_size,
                       int          taps,
                       int          block_size,
                       int          input_bits,
                       cudaStream_t stream);

/// Candidate-selector edge scores (DFlash2 `_score_edges`).
///
///   out[b, l, p, c] = unary[b, l, c]
///       + sum_r pred_cb[pred_id, r] * hidden[b, l, r] * succ_cb[cand_id, r]
///
/// where pred_id = anchor[b] if l == 0 else candidate_ids[b, l-1, p],
///       cand_id = candidate_ids[b, l, c].
/// `predecessor_codebook`/`successor_codebook` are [vocab, rank]; `out` is fp32
/// [batch, len, top_k, top_k].
void invokeCandidateSelector(float*       out,
                             const void*  hidden,
                             const void*  unary_logits,
                             const void*  predecessor_codebook,
                             const void*  successor_codebook,
                             const int*   candidate_ids,
                             const int*   anchor_token_ids,
                             int          batch,
                             int          len,
                             int          top_k,
                             int          vocab,
                             int          rank,
                             int          input_bits,
                             cudaStream_t stream);

/// DFlash2 draft block attention: non-causal GQA over [context | block].
///
///   q        [tq, nh,  hd]
///   k_ctx/v_ctx [tc, nkv, hd]   (context rows, RoPE already applied)
///   k_blk/v_blk [tq, nkv, hd]   (block rows, RoPE already applied)
///   out      [tq, nh,  hd]
///
/// Block queries attend to every block key (non-causal) and to context keys
/// within the sliding window before the query.
void invokeDflash2Attention(void*        out,
                            const void*  q,
                            const void*  k_ctx,
                            const void*  v_ctx,
                            const void*  k_blk,
                            const void*  v_blk,
                            int          tq,
                            int          tc,
                            int          nh,
                            int          nkv,
                            int          hd,
                            float        scale,
                            int          window,
                            int          input_bits,
                            cudaStream_t stream);

/// Quantized DFlash2 draft block attention: context K/V are packed int4 with
/// per-(token,head) scale/zero; block K/V stay in the input dtype.
void invokeDflash2AttentionInt4(void*        out,
                                const void*  q,
                                const void*  k_ctx_data,
                                const void*  k_ctx_param,
                                const void*  v_ctx_data,
                                const void*  v_ctx_param,
                                const void*  k_blk,
                                const void*  v_blk,
                                int          tq,
                                int          tc,
                                int          nh,
                                int          nkv,
                                int          hd,
                                float        scale,
                                int          window,
                                int          input_bits,
                                cudaStream_t stream);

/// NeoX-style RoPE applied in place to a [rows, heads, head_dim] tensor.
void invokeDflash2Rope(void*        x,
                       const int*   positions,
                       int          rows,
                       int          heads,
                       int          head_dim,
                       float        base,
                       int          input_bits,
                       cudaStream_t stream);

/// Elementwise `out = a + b` over `n` elements (fp16/bf16/fp32).
void invokeDflash2Add(void*        out,
                      const void*  a,
                      const void*  b,
                      long long    n,
                      int          input_bits,
                      cudaStream_t stream);

/// Per-row top-k over the vocabulary: fills `candidate_ids` [rows, k] (int) and
/// `values` [rows, k] (fp32, descending) from `logits` [rows, vocab] (fp16/fp32).
void invokeDflash2TopK(int*         candidate_ids,
                       float*       values,
                       const void*  logits,
                       int          rows,
                       int          vocab,
                       int          k,
                       int          input_bits,
                       cudaStream_t stream);

/// Build the DFlash2 block ids `[next_token, mask x (block-1)]` on device.
/// `ids[0] = token_ids[seq_len-1]` when `seq_len > 0`, else `mask`.
void invokeDflash2BuildBlockIds(int*               ids,
                                const int* const*  token_ids_ptrs,  // device array of device pointers
                                const int*         seq_len,         // device pointer to one int
                                int                block,
                                int                mask_token_id,
                                cudaStream_t       stream);

/// Greedy chain through the candidate lattice: `props[l] = cand[l, argmax_c edge[l, prev, c]]`,
/// `prev` starts at 0 and follows the chosen candidate index.
void invokeDflash2GreedyChain(int*         props,
                              const float* edge,
                              const int*   cand,
                              int          len,
                              int          top_k,
                              cudaStream_t stream);

/// Write `props[0..n)` into `token_ids_ptrs[0][base[0] + i]` (all device-side).
void invokeDflash2StoreTokens(int* const*  token_ids_ptrs,
                              const int*   props,
                              const int*   base,  // device pointer to one int
                              int          n,
                              cudaStream_t stream);

/// Range-preserving residual + RMSNorm: `residual[f32] += x` (when `has_x`),
/// then `out = fp16(rmsnorm(residual) * w)`. Keeps the residual stream in FP32
/// so it does not overflow fp16 (DFlash2 SM70/75 requirement).
void invokeDflash2ResidualRMSNorm(void*        out,
                                  float*       residual,
                                  const void*  x,
                                  const void*  w,
                                  int          rows,
                                  int          cols,
                                  float        eps,
                                  int          has_x,
                                  float        x_scale,  // residual += x * x_scale
                                  int          input_bits,
                                  cudaStream_t stream);

/// `out[0] = entry[0] + (has_accept ? accept[0] : 0)` (device-side draft length).
void invokeDflash2CommitLen(int* out, const int* entry, const int* accept, int has_accept, cudaStream_t stream);

/// `out[i] = base[0] + i` for i in [0, n) (device-side position generation).
void invokeDflash2CountUp(int* out, const int* base, int n, cudaStream_t stream);

/// Scatter context K/V rows: dst[base[0] + i] = src[i] for i < committed[0] - base[0].
/// `src`/`dst` are [rows, nkv, hd] / [max_ctx, nkv, hd] flattened; `per_row = nkv*hd`.
void invokeDflash2KvScatter(void*        dst,
                            const void*  src,
                            const int*   base,
                            const int*   committed,
                            int          rows,
                            int          per_row,
                            int          input_bits,
                            cudaStream_t stream);

/// Quantize and scatter context K/V rows: dst_data[base[0] + i] = int4(src[i]),
/// dst_param[...] = (scale, zero), for i < committed[0] - base[0].
/// `src` is [rows, nkv, hd]; `dst_data` is [max_ctx, nkv, hd/2] (packed);
/// `dst_param` is [max_ctx, nkv, 2] in the input dtype.
void invokeDflash2KvScatterInt4(void*        dst_data,
                                void*        dst_param,
                                const void*  src,
                                const int*   base,
                                const int*   committed,
                                int          rows,
                                int          nkv,
                                int          hd,
                                int          input_bits,
                                cudaStream_t stream);

/// In-place `t *= factor` (elementwise, fp16/fp32).
void invokeDflash2Scale(void*        t,
                        long long    n,
                        float        factor,
                        int          input_bits,
                        cudaStream_t stream);

}  // namespace turbomind
