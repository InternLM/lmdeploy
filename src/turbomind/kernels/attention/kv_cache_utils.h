// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"

namespace turbomind {

template<class T>
void invokeProcessKV(void**       blocks,
                     const T*     k,
                     const T*     v,
                     const T*     k_bias,
                     const T*     v_bias,
                     const int*   cu_q_len,
                     const int*   cu_k_len,
                     const int*   cu_block_num,
                     const float* rope_base,
                     int          stride_b,
                     int          stride_c,  // cumulative len
                     int          stride_h,
                     int          stride_s,
                     int          block_seq_len,
                     int          block_k_offset,
                     int          block_v_offset,
                     int          max_q_len,
                     int          kv_head_num,
                     int          batch_size,
                     int          quant_policy,
                     const float* quant_params_kv,
                     cudaStream_t stream = {});

template<class T>
void invokeProcessKV_(const AttentionParams<T>& params)
{
    invokeProcessKV(params.k_cache_block_ptrs,
                    params.k,
                    params.v,
                    params.k_bias,
                    params.v_bias,
                    params.cu_q_len,
                    params.cu_k_len,
                    params.cu_block_cnts,
                    params.rope_theta,
                    0,                                     // stride b
                    params.stride / params.size_per_head,  // stride c
                    1,                                     // stride h
                    params.stride / params.size_per_head,  // stride s
                    params.kv_cache_block_size,
                    params.key_offset,
                    params.val_offset,
                    params.max_q_len,
                    params.num_kv_heads,
                    params.batch_size,
                    params.quant_policy,
                    params.kv_quant_params,
                    params.stream);
}

template<class T>
void invokeFlattenKV(T*           k,
                     T*           v,
                     const void** blocks,
                     const int*   cu_k_len,
                     const int*   cu_block_num,
                     const float* rope_base,
                     int          stride_b,
                     int          stride_c,  // cumulative len
                     int          stride_h,
                     int          stride_s,
                     int          block_seq_len,
                     int          block_k_offset,
                     int          block_v_offset,
                     int          max_seq_len,
                     int          head_num,
                     int          batch_size,
                     int          quant_policy,
                     const float* quant_params,
                     cudaStream_t stream = {});

template<class T>
void invokeFlattenKV_(const AttentionParams<T>& params, int sum_k_len)
{
    // blocks -> [H, 2, sum_k_len, D]
    invokeFlattenKV((T*)params.kv,
                    (T*)params.kv + sum_k_len * params.size_per_head,
                    (const void**)params.k_cache_block_ptrs,
                    params.cu_k_len,
                    params.cu_block_cnts,
                    nullptr,  // params.rope_theta,
                    0,
                    1,
                    2 * sum_k_len,
                    1,
                    params.kv_cache_block_size,
                    params.key_offset,
                    params.val_offset,
                    params.max_k_len,
                    params.num_kv_heads,
                    params.batch_size,
                    params.quant_policy,
                    params.kv_quant_params,
                    params.stream);
}

}  // namespace turbomind