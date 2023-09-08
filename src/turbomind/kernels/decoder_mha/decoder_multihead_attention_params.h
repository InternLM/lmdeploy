#pragma once

namespace turbomind {

template<typename T>
struct DecoderMultiHeadAttentionParams {
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

    // sequence-level buffers
    int*  per_sample_length;
    bool* finished;

    // kv cache
    void** per_sample_k_cache;  // [H, S, D]
    void** per_sample_v_cache;  // [H, S, D]
    size_t per_sample_kv_cache_offset;

    // batch-level params
    int batch_size;
    int max_seq_len;
    int max_timestep;  // max_timestep in the batch, used to compute smem sizes

    // instance-level params
    int num_heads;
    int num_kv_heads;
    int size_per_head;
    float inv_sqrt_dh;

    // rotary embedding
    int   rotary_embedding_dim;
    float rotary_embedding_base;
    int   max_position_embeddings;
    bool  use_dynamic_ntk;

    // log(n) attention
    bool use_logn_attn;
};

}  // namespace turbomind