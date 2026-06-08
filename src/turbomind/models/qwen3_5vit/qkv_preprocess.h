// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"

#include <cuda_runtime.h>

namespace turbomind {

// Prepare the Qwen3.5 ViT attention inputs after the fused QKV projection.
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
void invokeQwen3_5VitPrepareQKV(void*        qkv,
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

}  // namespace turbomind
