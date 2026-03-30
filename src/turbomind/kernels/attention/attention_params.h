// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "cutlass/fast_math.h"
#include <cstdint>
#include <cuda_runtime.h>

#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

// 64-bit offsets may be needed
struct LinearIteratorParams {
    const void* kv_cache;
    int         stride_h;
    int         key_to_val;
};

struct BlockIteratorParams {
    char**     block_ptrs;
    const int* cu_block_nums;
    int        layer_id;
    int        block_len;
};

typedef void (*cp_post_fn)(void* context);

/// TODO: Rename to attention::Param
template<typename T>
struct AttentionParams {
    // token-level buffers, [B, qH + 2kvH, D] or [B, kvH, D]
    T*      out;
    T*      q;
    T*      k;
    T*      v;
    int64_t stride;

    // bias, [qH, D] or [kvH, D]
    T* q_bias;
    T* k_bias;
    T* v_bias;

    // sequence-level buffers
    const int*   cu_q_len;
    const int*   cu_k_len;
    const bool*  finished;
    const float* rope_theta;

    const T* sinks;
    float    scale_sinks;

    LinearIteratorParams linear_iter_params;
    BlockIteratorParams  block_iter_params;

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
    int   window_size;
    int   layer_id;  // for debugging

    // rotary embedding
    RopeKernelParam rope_param;

    // log(n) attention
    bool use_logn_attn;
    int  max_position_embeddings;

    int quant_policy;

    int    max_split_k;
    int*   split_cnt;
    float* partial_O;
    float* partial_ML;

    // context parallel
    int                 cp_rank{0};
    cutlass::FastDivmod cp_size{1};
    int                 offset_q{0};  // decode offset
    cp_post_fn          cp_fn{nullptr};
    void*               cp_fn_ctx{nullptr};

    int          arch;
    cudaStream_t stream;

    // debug
    float* qk;
    T*     pr;
};

template<class CacheIterFactory, class SFINAE = void>
struct CreateCacheIterFactory {
    template<class Param>
    static CacheIterFactory apply(const Param& param)
    {
        using Tkv = typename CacheIterFactory::Tkv;
        return {(const Tkv*)param.linear_iter_params.kv_cache,
                param.cu_k_len,
                param.linear_iter_params.stride_h,
                param.linear_iter_params.key_to_val};
    }
};

}  // namespace turbomind
