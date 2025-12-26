// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/attention/attention_params.h"

namespace turbomind {

template<class T>
void invokeProcessKV_v2(char**                 blocks,
                        const T*               k,
                        const T*               v,
                        const T*               k_bias,
                        const T*               v_bias,
                        const int*             cu_q_len,
                        const int*             cu_k_len,
                        const int*             cu_block_num,
                        const RopeKernelParam& rope_param,
                        int64_t                stride_b,
                        int64_t                stride_c,
                        int64_t                stride_h,
                        int64_t                stride_s,
                        int                    block_seq_len,
                        int                    layer_id,
                        int                    cp_rank,
                        cutlass::FastDivmod    cp_size,
                        int                    max_q_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        cudaStream_t           stream = {});

template<class T>
void invokeProcessKV_v2_(const AttentionParams<T>& params)
{
    invokeProcessKV_v2((char**)params.block_iter_params.block_ptrs,
                       params.k,
                       params.v,
                       params.k_bias,
                       params.v_bias,
                       params.cu_q_len,
                       params.cu_k_len,
                       params.block_iter_params.cu_block_nums,
                       params.rope_param,
                       0,                                     // stride b
                       params.stride / params.size_per_head,  // stride c
                       1,                                     // stride h
                       params.stride / params.size_per_head,  // stride s
                       params.block_iter_params.block_len,
                       params.block_iter_params.layer_id,
                       params.cp_rank,
                       params.cp_size,
                       params.max_q_len,
                       params.num_kv_heads,
                       params.size_per_head,
                       params.batch_size,
                       params.quant_policy,
                       params.stream);
}

template<class T>
void invokeFlattenKV_v2(T*                     k,
                        T*                     v,
                        char**                 blocks,
                        const int*             cu_k_len,
                        const int*             cu_block_num,
                        const RopeKernelParam& rope_param,
                        int64_t                stride_b,
                        int64_t                stride_c,
                        int64_t                stride_h,
                        int64_t                stride_s,
                        int                    block_seq_len,
                        int                    layer_id,
                        int                    cp_rank,
                        cutlass::FastDivmod    cp_size,
                        int                    max_seq_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        cudaStream_t           stream = {});

/// TODO: remove `sum_k_len`
template<class T>
void invokeFlattenKV_v2_(const AttentionParams<T>& params, int sum_k_len)
{
    // blocks -> [H, 2, sum_k_len, D]
    invokeFlattenKV_v2((T*)params.linear_iter_params.kv_cache,
                       (T*)params.linear_iter_params.kv_cache + sum_k_len * params.size_per_head,
                       (char**)params.block_iter_params.block_ptrs,
                       params.cu_k_len,
                       params.block_iter_params.cu_block_nums,
                       RopeKernelParam{},
                       0,
                       1,
                       2 * sum_k_len,
                       1,
                       params.block_iter_params.block_len,
                       params.block_iter_params.layer_id,
                       params.cp_rank,
                       params.cp_size,
                       params.max_k_len,
                       params.num_kv_heads,
                       params.size_per_head,
                       params.batch_size,
                       params.quant_policy,
                       params.stream);
}

size_t
get_cache_block_size(DataType dtype, DataType kvtype, int layer_num, int head_num, int head_dim, int block_seq_len);

}  // namespace turbomind
