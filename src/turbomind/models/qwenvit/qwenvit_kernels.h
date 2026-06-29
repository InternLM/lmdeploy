// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"

#include <cuda_runtime.h>

namespace turbomind {

// =====================================================================================
// QKV preprocessing (qkv_preprocess)
// =====================================================================================

// Prepare the Qwen ViT attention inputs after the fused QKV projection.
//
// qkv layout:
//   [token, local_q_heads + 2 * local_kv_heads, head_dim]
// Q is updated in place with bias + RoPE. K/V are written to `kv` as:
//   [local_kv_heads, 2, token, head_dim]
//
// `rope_head_dim` is the per-head dim of the rotary_pos_emb buffer and is
// also the cutoff below which RoPE is applied. When the model's real head_dim
// is not natively supported by the attention kernel, Q/K/V are zero-padded
// per-head to a kernel-supported `head_dim` >= `rope_head_dim`; the padded
// `[rope_head_dim, head_dim)` slice has zero Q/K so RoPE is skipped there.
void invokeQwenVitPrepareQKV(void*        qkv,
                             void*        kv,
                             const void*  qkv_bias,
                             const void*  rotary_pos_emb,
                             const int*   mapped_idx,
                             DataType     dtype,
                             int          token_num,
                             int          local_head_num,
                             int          head_dim,
                             int          rope_head_dim,
                             cudaStream_t stream);

// =====================================================================================
// Spatial-merge index mapping (grid_mapping)
// =====================================================================================

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 int          token_offset,
                                 int          natural_offset,
                                 int          t,
                                 int          h,
                                 int          w,
                                 int          spatial_merge_size,
                                 cudaStream_t stream);

void invokeQwenVitBuildMappedIdx(int*         mapped_idx,
                                 const int*   grid_thws,
                                 const int*   grid_offsets,
                                 int          num_grids,
                                 int          spatial_merge_size,
                                 cudaStream_t stream);

// =====================================================================================
// Learned positional-embedding bilinear interpolation (fast_pos_embed) — Qwen3.5
// =====================================================================================

// Precomputes the 4 bilinear-interpolation gather indices and weights
// used by the subsequent pos-embed merge step in Qwen3-VL.
void invokeFastPosEmbedIdxWeight(int*         idx_out,     // [total_n * 4]
                                 void*        weight_out,  // [total_n * 4]
                                 DataType     dtype,
                                 const int*   grid_thws,     // [num_grids * 3], (t, h, w)
                                 const int*   grid_offsets,  // [num_grids * 2], (t*h*w, h*w)
                                 int          num_grids,
                                 int          total_n,
                                 int          num_grid_per_side,
                                 cudaStream_t stream);

// Fuses the spatial-merge permutation, the bilinear-weighted sum, and the
// t-expansion of Qwen3-VL ViT pos_embed interpolation into a single pass on
// top of the patch_embed linear output. (fused_embed_merge)
void invokeFusedPosEmbedMerge(void*        hidden_states,      // [batch, hidden]
                              const void*  pos_embeds,         // [total_hw * 4, hidden]
                              const void*  pos_embed_weights,  // [total_hw * 4]
                              const int*   mapped_idx,         // [batch]
                              const void*  bias,               // [hidden] or nullptr
                              int          batch,
                              int          hidden,
                              DataType     dtype,
                              cudaStream_t stream);

// =====================================================================================
// 2D rotary position embedding table (fast_rotary_pos_emb)
// =====================================================================================

// Precomputes the (cos, sin) rotary-embedding table for Qwen-VL vision tokens.
// Layout per natural flat position (keyed by the same index `mapped_idx` carries):
//   [c_0, s_0, c_1, s_1, ..., c_{head_dim/2-1}, s_{head_dim/2-1}]
// The pair index `k` uses `h_coord` for k < head_dim/4 and `w_coord` otherwise,
// with inv_freq = theta^(-2*(k % (head_dim/4)) / (head_dim/2)).
void invokeQwenVitRotaryPosEmb(void*        cos_sin,  // [total_hw, head_dim]
                               DataType     dtype,
                               const int*   grid_thws,     // [num_grids * 3], (t, h, w)
                               const int*   grid_offsets,  // [num_grids * 2], (t*h*w, h*w)
                               int          num_grids,
                               int          total_hw,
                               int          head_dim,
                               float        theta,
                               cudaStream_t stream);

// =====================================================================================
// mrope position ids (mrope_position_ids)
// =====================================================================================

// One descriptor per text / image run, clipped to a prefill chunk's active window.
// `h2 == 0` flags a text run (real image grids always have h2 > 0).
struct MropeSegment {
    int dst_offset;  // flat forward-token index of the first token written by this segment
    int n_tok;       // tokens to write (already clipped to the active range)
    int base_pos;    // text: position id at local_k = 0; image: image's mm_start
    int h2;          // image grid h after spatial merge (0 => text)
    int w2;          // image grid w after spatial merge (ignored when h2 == 0)
    int k_offset;    // starting "k" for image grid math (clip-offset within the run); unused for text
};

// Scatter `num_segments` segments into `pos_ids` of shape (max_fwd_tokens, 3).
// `pos_ids` may be null when num_segments == 0.
void invokeMropePositionIds(int*                pos_ids,
                            const MropeSegment* segments,  // device
                            int                 num_segments,
                            int                 max_seg_len,
                            cudaStream_t        stream);

// =====================================================================================
// Window attention reordering (window kernels) — Qwen2.5
// =====================================================================================

void invokeQwenVitWindowReorder(
    Tensor& out, const Tensor& in, const int* window_idx, int merge_unit, int group_count, cudaStream_t stream);

void invokeQwenVitReverseWindow(
    Tensor& out, const Tensor& in, const int* window_idx, int group_count, cudaStream_t stream);

void invokeQwenVitBuildWindowMappedIdx(int*         window_mapped_idx,
                                       const int*   mapped_idx,
                                       const int*   window_idx,
                                       int          merge_unit,
                                       int          group_count,
                                       cudaStream_t stream);

}  // namespace turbomind
