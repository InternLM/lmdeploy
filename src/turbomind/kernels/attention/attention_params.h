// Copyright (c) OpenMMLab. All rights reserved.

#pragma once
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
struct AttentionParams {
    // token-level buffers, [B, qH + 2kvH, D] or [B, kvH, D]
    T*  out;
    T*  q;
    T*  k;
    T*  v;
    int stride;

    // bias, [qH, D] or [kvH, D]
    T* q_bias;
    T* k_bias;
    T* v_bias;

    const void* kv;  // tmp kv cache buffer

    // sequence-level buffers
    const int*   cu_q_len;
    const int*   cu_k_len;
    const bool*  finished;
    const float* rope_theta;

    int key_offset;
    int val_offset;

    void** k_cache_block_ptrs;  // S/s,[L,2,H,s,D]
    int*   cu_block_cnts;       // [B+1]
    int    kv_cache_block_size;

    // batch-level params
    int token_num;
    int batch_size;
    int max_q_len;
    int max_k_len;

    // instance-level params
    int   num_heads;
    int   num_kv_heads;
    int   size_per_head;
    float inv_sqrt_dh;

    // rotary embedding
    int   rotary_embedding_dim;
    float rotary_embedding_base;
    int   max_position_embeddings;
    float rope_ti_scale;  // used for linear RoPE scaling

    // log(n) attention
    bool use_logn_attn;

    int   quant_policy;
    float kv_quant_params[4];

    int    max_split_k;
    int*   split_cnt;
    float* partial_O;
    float* partial_M;
    float* partial_L;
    int*   locks;

    int          arch;
    cudaStream_t stream;

    // debug
    float* qk;
    T*     pr;
};

}  // namespace turbomind
