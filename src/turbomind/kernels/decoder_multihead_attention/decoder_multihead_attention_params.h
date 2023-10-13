#pragma once
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
struct DecoderMultiHeadAttentionParams {
    // token-level buffers, [B, qH + 2kvH, D] or [B, kvH, D]
    T* __restrict__ out;
    T* __restrict__ q;
    T* __restrict__ k;
    T* __restrict__ v;
    int stride;

    // bias, [qH, D] or [kvH, D]
    T* __restrict__ q_bias;
    T* __restrict__ k_bias;
    T* __restrict__ v_bias;

    // sequence-level buffers
    const int* __restrict__ per_sample_length;
    const bool* __restrict__ finished;

    // kv cache
    void** __restrict__ per_sample_k_cache;  // [H, S, D]
    void** __restrict__ per_sample_v_cache;  // [H, S, D]
    size_t layer_offset;

    /// cache layout M,[N,H,x,D]
    /// S: [s0/x, s1/x, s2/x, ..., sn-1/x], si <- block
    /// 1. [L,sum(S),H,x,D]
    void** __restrict__ k_cache_block_ptrs;  // X,[H,x,D]
    void** __restrict__ v_cache_block_ptrs;  // X,[H,x,D]
    int* __restrict__ cu_block_cnts;         // [B+1]
    int kv_cache_block_size;

    // batch-level params
    int batch_size;
    int max_seq_len;

    // instance-level params
    int   num_heads;
    int   num_kv_heads;
    int   size_per_head;
    float inv_sqrt_dh;

    // rotary embedding
    int   rotary_embedding_dim;
    float rotary_embedding_base;
    int   max_position_embeddings;
    bool  use_dynamic_ntk;

    // log(n) attention
    bool use_logn_attn;

    int   quant_policy;
    float kv_quant_params[4];

    cudaStream_t stream;
};

}  // namespace turbomind