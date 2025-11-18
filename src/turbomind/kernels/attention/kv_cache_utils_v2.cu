// Copyright (c) OpenMMLab. All rights reserved.

#include <type_traits>

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

using cutlass::FastDivmod;

template<class Tkv, int CTA_S, int HeadDim, int WarpCnt, class T, class BlockLayout>
__global__ void __launch_bounds__(128) ProcessKV_v2(char**          blocks,
                                                    const T*        k,
                                                    const T*        v,
                                                    const T*        k_bias,
                                                    const T*        v_bias,
                                                    const int*      cu_q_len,
                                                    const int*      cu_k_len,
                                                    const int*      cu_block_num,
                                                    RopeKernelParam rope_param,
                                                    int64_t         stride_b,
                                                    int64_t         stride_c,
                                                    int64_t         stride_h,
                                                    int64_t         stride_s,
                                                    int             layer_id,
                                                    int             cp_rank,
                                                    FastDivmod      cp_size,
                                                    BlockLayout     block_layout)
{

    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    const int token_idx = blockIdx.x * CTA_S;  // local offset into `input_length`
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int qi_beg = cu_q_len[batch_idx];
    const int qi_end = cu_q_len[batch_idx + 1];
    const int q_len  = qi_end - qi_beg;

    const int k_len       = cu_k_len[batch_idx + 1] - cu_k_len[batch_idx];
    const int history_len = k_len - q_len;

    if (qi_beg + token_idx >= qi_end) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Vec __align__(16) vec_K[ITER_S][ITER_C];
    Vec __align__(16) vec_V[ITER_S][ITER_C];

    Vec bias_V[ITER_C];
    Vec bias_K[ITER_C];

    if (k_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_K[c], &k_bias[head_idx * HeadDim + di]);
        }
    }
    if (v_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_V[c], &v_bias[head_idx * HeadDim + di]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int     qi = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
            const int     di = offset.x + c * Map::kDeltaC;
            const int64_t index =
                (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + head_idx * stride_h) * HeadDim + di;
            if (qi < q_len) {
                Ldg(vec_K[s][c], &k[index]);
                Ldg(vec_V[s][c], &v[index]);
            }
        }
    }

    if (k_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_K[s][c] = vec_K[s][c] + bias_K[c];
            }
        }
    }
    if (v_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_V[s][c] = vec_V[s][c] + bias_V[c];
            }
        }
    }

    if (rope_param.type != RopeType::kNull) {
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = history_len + offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(vec_K[s][c], ti);
            }
        }
    }

    Array<T, 2> param_K[ITER_S];
    Array<T, 2> param_V[ITER_S];

    if constexpr (!std::is_same_v<T, Tkv>) {
        warp_stats<Map::kWarpThreadC>(param_K, vec_K, bitsof<Tkv>);
        warp_stats<Map::kWarpThreadC>(param_V, vec_V, bitsof<Tkv>);
    }

    Array<Tkv, kVecSize> out_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> out_V[ITER_S][ITER_C];

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        ConvertKvCache<T, Tkv> conv_K{param_K[s][0], param_K[s][1]};
        ConvertKvCache<T, Tkv> conv_V{param_V[s][0], param_V[s][1]};
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            out_K[s][c] = conv_K(vec_K[s][c]);
            out_V[s][c] = conv_V(vec_V[s][c]);
        }
    }

    int local_ti, local_ti_rank;

    blocks += cu_block_num[batch_idx];

    block::Head<T, Tkv, BlockLayout> block_head{block_layout, layer_id, head_idx};

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`
        const int ti = history_len + qi;                         // timestep
        local_ti     = cp_size.divmod(local_ti_rank, ti);
        if (qi < q_len && local_ti_rank == cp_rank) {
            block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    int di = offset.x + c * Map::kDeltaC;
                    Store(&k_cache[di], out_K[s][c]);
                    Store(&v_cache[di], out_V[s][c]);
                }
                if constexpr (!std::is_same_v<T, Tkv>) {
                    if (offset.x == 0) {
                        StoreQuantParam<Tkv>(k_param, param_K[s]);
                        StoreQuantParam<Tkv>(v_param, param_V[s]);
                        // if (ti == history_len) {
                        // printf("src %d %f %f\n", ti, (float)param_K[s][0], (float)param_K[s][1]);
                        // }
                    }
                }
            });
        }
    }
}

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
                        FastDivmod             cp_size,
                        int                    max_q_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        cudaStream_t           stream)
{
    constexpr int WARPS = 4;
    constexpr int CTA_S = 64;

    int  block = WARPS * WARP_SIZE;
    dim3 grid((max_q_len + CTA_S - 1) / CTA_S, head_num, batch_size);

    auto invoke = [&](auto tkv, const auto dim) {
        using Tkv = decltype(tkv);

        constexpr int kHeadDim = dim;
        FT_CHECK(head_dim == kHeadDim);

        block::Layout block_layout{block::Config<T, Tkv, kHeadDim>{head_num, block_seq_len}};

        ProcessKV_v2<Tkv, CTA_S, kHeadDim, WARPS><<<grid, block, 0, stream>>>(blocks,
                                                                              k,
                                                                              v,
                                                                              k_bias,
                                                                              v_bias,
                                                                              cu_q_len,
                                                                              cu_k_len,
                                                                              cu_block_num,
                                                                              rope_param,
                                                                              stride_b,
                                                                              stride_c,
                                                                              stride_h,
                                                                              stride_s,
                                                                              layer_id,
                                                                              cp_rank,
                                                                              cp_size,
                                                                              block_layout);
    };

    auto dispatch = [&](auto tkv) {
        if (head_dim == 64) {
            return invoke(tkv, std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            return invoke(tkv, std::integral_constant<int, 128>{});
        }
        else if (head_dim == 192) {
            return invoke(tkv, std::integral_constant<int, 192>{});
        }
        FT_CHECK(0);
    };

    if (quant_policy & QuantPolicy::kCacheKVInt8) {
        dispatch(uint8_t{});
    }
    else if (quant_policy & QuantPolicy::kCacheKVInt4) {
        dispatch(uint4_t{});
    }
    else {
        dispatch(T{});
    }
}

#define INSTANTIATE_invokeProcessKV_v2(type)                                                                           \
    template void invokeProcessKV_v2(char**                 blocks,                                                    \
                                     const type*            k,                                                         \
                                     const type*            v,                                                         \
                                     const type*            k_bias,                                                    \
                                     const type*            v_bias,                                                    \
                                     const int*             cu_q_len,                                                  \
                                     const int*             cu_k_len,                                                  \
                                     const int*             cu_block_num,                                              \
                                     const RopeKernelParam& rope_param,                                                \
                                     int64_t                stride_b,                                                  \
                                     int64_t                stride_c,                                                  \
                                     int64_t                stride_h,                                                  \
                                     int64_t                stride_s,                                                  \
                                     int                    block_seq_len,                                             \
                                     int                    layer_id,                                                  \
                                     int                    cp_rank,                                                   \
                                     FastDivmod             cp_size,                                                   \
                                     int                    max_q_len,                                                 \
                                     int                    head_num,                                                  \
                                     int                    head_dim,                                                  \
                                     int                    batch_size,                                                \
                                     int                    quant_policy,                                              \
                                     cudaStream_t           stream);

INSTANTIATE_invokeProcessKV_v2(half);
#if ENABLE_BF16
INSTANTIATE_invokeProcessKV_v2(nv_bfloat16);
#endif

template<int CTA_S, int HeadDim, int WarpCnt, class T, class Tkv, class BlockLayout>
__global__ void __launch_bounds__(128) flattenKV_v2(T*              k,
                                                    T*              v,
                                                    const Tkv**     blocks,
                                                    const int*      cu_k_len,
                                                    const int*      cu_block_num,
                                                    RopeKernelParam rope_param,
                                                    int64_t         stride_b,
                                                    int64_t         stride_c,
                                                    int64_t         stride_h,
                                                    int64_t         stride_s,
                                                    int             layer_id,
                                                    int             cp_rank,
                                                    FastDivmod      cp_size,
                                                    BlockLayout     block_layout)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    const int token_idx = blockIdx.x * CTA_S;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int ti_0   = cu_k_len[0];
    const int ti_beg = cu_k_len[batch_idx] - ti_0;
    const int ti_end = cu_k_len[batch_idx + 1] - ti_0;

    const int seq_len = ti_end - ti_beg;

    if (ti_beg + token_idx >= ti_end) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Array<Tkv, kVecSize> __align__(16) vec_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> __align__(16) vec_V[ITER_S][ITER_C];

    Array<T, kVecSize> __align__(16) out_K[ITER_S][ITER_C];
    Array<T, kVecSize> __align__(16) out_V[ITER_S][ITER_C];

    blocks += cu_block_num[batch_idx];

    block::Head<T, Tkv, BlockLayout> block_head{block_layout, layer_id, head_idx};

    Array<T, 2> param_K[ITER_S];
    Array<T, 2> param_V[ITER_S];

    int local_ti, local_ti_rank;

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int si = offset.y + s * Map::kDeltaS + token_idx;
        local_ti     = cp_size.divmod(local_ti_rank, si);
        if (si < seq_len && local_ti_rank == cp_rank) {
            block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    int di = offset.x + c * Map::kDeltaC;
                    Ldg(vec_K[s][c], &k_cache[di]);
                    Ldg(vec_V[s][c], &v_cache[di]);
                }
                if constexpr (!std::is_same_v<T, Tkv>) {
                    Ldg(param_K[s], k_param);
                    Ldg(param_V[s], v_param);
                }
            });
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        ConvertKvCache<Tkv, T> conv_K{param_K[s][0], param_K[s][1]};
        ConvertKvCache<Tkv, T> conv_V{param_V[s][0], param_V[s][1]};
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            out_K[s][c] = conv_K(vec_K[s][c]);
            out_V[s][c] = conv_V(vec_V[s][c]);
        }
    }

    if (rope_param.type != RopeType::kNull) {
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(out_K[s][c], ti);
            }
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int si = offset.y + s * Map::kDeltaS + token_idx;
            const int di = offset.x + c * Map::kDeltaC;
            local_ti     = cp_size.divmod(local_ti_rank, si);
            if (si < seq_len && local_ti_rank == cp_rank) {
                const int64_t index =
                    (batch_idx * stride_b + ti_beg * stride_c + local_ti * stride_s + head_idx * stride_h) * HeadDim
                    + di;
                Store(&k[index], out_K[s][c]);
                Store(&v[index], out_V[s][c]);
            }
        }
    }
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
                        FastDivmod             cp_size,
                        int                    max_seq_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        cudaStream_t           stream)
{
    constexpr int kWarpCnt = 4;
    constexpr int CTA_S    = 64;

    constexpr int block = kWarpCnt * WARP_SIZE;
    const dim3    grid((max_seq_len + CTA_S - 1) / CTA_S, head_num, batch_size);

    auto invoke = [&](auto tkv, const auto dim) {
        using Tkv = decltype(tkv);

        constexpr int kHeadDim = dim;
        FT_CHECK(head_dim == kHeadDim);

        block::Layout block_layout{block::Config<T, Tkv, kHeadDim>{head_num, block_seq_len}};

        flattenKV_v2<CTA_S, kHeadDim, kWarpCnt><<<grid, block, 0, stream>>>(k,
                                                                            v,
                                                                            (const Tkv**)blocks,
                                                                            cu_k_len,
                                                                            cu_block_num,
                                                                            rope_param,
                                                                            stride_b,
                                                                            stride_c,
                                                                            stride_h,
                                                                            stride_s,
                                                                            layer_id,
                                                                            cp_rank,
                                                                            cp_size,
                                                                            block_layout);
    };

    auto dispatch = [&](auto tkv) {
        if (head_dim == 64) {
            return invoke(tkv, std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            return invoke(tkv, std::integral_constant<int, 128>{});
        }
        else if (head_dim == 192) {
            return invoke(tkv, std::integral_constant<int, 192>{});
        }
        FT_CHECK(0);
    };

    if (quant_policy & QuantPolicy::kCacheKVInt8) {
        dispatch(uint8_t{});
    }
    else if (quant_policy & QuantPolicy::kCacheKVInt4) {
        dispatch(uint4_t{});
    }
    else {
        dispatch(T{});
    }
}

#define INSTANTIATE_invokeFlattenKV_v2(type)                                                                           \
    template void invokeFlattenKV_v2(type*                  k,                                                         \
                                     type*                  v,                                                         \
                                     char**                 blocks,                                                    \
                                     const int*             cu_k_len,                                                  \
                                     const int*             cu_block_num,                                              \
                                     const RopeKernelParam& rope_param,                                                \
                                     int64_t                stride_b,                                                  \
                                     int64_t                stride_c,                                                  \
                                     int64_t                stride_h,                                                  \
                                     int64_t                stride_s,                                                  \
                                     int                    block_seq_len,                                             \
                                     int                    layer_id,                                                  \
                                     int                    cp_rank,                                                   \
                                     FastDivmod             cp_size,                                                   \
                                     int                    max_seq_len,                                               \
                                     int                    head_num,                                                  \
                                     int                    head_dim,                                                  \
                                     int                    batch_size,                                                \
                                     int                    quant_policy,                                              \
                                     cudaStream_t           stream);

INSTANTIATE_invokeFlattenKV_v2(half);
#if ENABLE_BF16
INSTANTIATE_invokeFlattenKV_v2(nv_bfloat16);
#endif

}  // namespace turbomind
