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

}  // namespace turbomind
