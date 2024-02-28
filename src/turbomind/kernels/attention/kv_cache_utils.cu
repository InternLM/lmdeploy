
#include "array_ops.h"
#include "kv_cache_utils.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "thread_map.h"
#include <type_traits>

namespace turbomind {

template<int CTA_S, int HeadDim, int WarpCnt, class T, class Tkv, class Offset, class TransformK, class TransformV>
__global__ void __launch_bounds__(128) ProcessKV(Tkv**        blocks,
                                                 const T*     k,
                                                 const T*     v,
                                                 const T*     k_bias,
                                                 const T*     v_bias,
                                                 const int*   cu_q_len,
                                                 const int*   cu_k_len,
                                                 const int*   cu_block_num,
                                                 const float* rope_base,
                                                 int          stride_b,
                                                 int          stride_c,
                                                 int          stride_h,
                                                 int          stride_s,
                                                 int          block_seq_len,
                                                 Offset       k_offset,
                                                 Offset       v_offset,
                                                 TransformK   transform_k,
                                                 TransformV   transform_v)
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
            const int qi = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
            const int di = offset.x + c * Map::kDeltaC;
            const int index =
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

    if (rope_base) {
        float base = rope_base[batch_idx];
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            FastRoPE  rope(di, std::integral_constant<int, HeadDim>{}, base, std::integral_constant<int, kVecSize>{});
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = history_len + offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(vec_K[s][c], ti);
            }
        }
    }

    Array<Tkv, kVecSize> out_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> out_V[ITER_S][ITER_C];

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            out_K[s][c] = transform_k(vec_K[s][c]);
            out_V[s][c] = transform_v(vec_V[s][c]);
        }
    }

    Tkv** k_cache_block_ptrs = blocks + cu_block_num[batch_idx];

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`

        if (qi < q_len) {
            const int ti = history_len + qi;  // timestep

            const int block_seqlen = block_seq_len;
            // block index and local offsets
            const int cache_block_index  = ti / block_seqlen;
            const int cache_block_offset = ti % block_seqlen;
            // [H, s, D]
            Tkv* k_cache = k_cache_block_ptrs[cache_block_index] + k_offset + head_idx * block_seqlen * HeadDim
                           + cache_block_offset * HeadDim;
            Tkv* v_cache = k_cache_block_ptrs[cache_block_index] + v_offset + head_idx * block_seqlen * HeadDim
                           + cache_block_offset * HeadDim;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                int di = offset.x + c * Map::kDeltaC;
                Store(&k_cache[di], out_K[s][c]);
                Store(&v_cache[di], out_V[s][c]);
            }
        }
    }
}

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
                     int          stride_c,
                     int          stride_h,
                     int          stride_s,
                     int          block_seq_len,
                     int          k_offset,
                     int          v_offset,
                     int          max_q_len,
                     int          kv_head_num,
                     int          batch_size,
                     int          quant_policy,
                     const float* quant_params,
                     cudaStream_t stream)
{
    constexpr int WARPS = 4;
    constexpr int DIMS  = 128;
    constexpr int CTA_S = 64;

    int  block = WARPS * WARP_SIZE;
    dim3 grid((max_q_len + CTA_S - 1) / CTA_S, kv_head_num, batch_size);

    auto invoke = [&](auto tkv) {
        using Tkv = decltype(tkv);
        ConvertKvCache<T, Tkv> transform_k{quant_params[0], quant_params[1]};
        ConvertKvCache<T, Tkv> transform_v{quant_params[2], quant_params[3]};
        ProcessKV<CTA_S, DIMS, WARPS><<<grid, block, 0, stream>>>((Tkv**)blocks,
                                                                  k,
                                                                  v,
                                                                  k_bias,
                                                                  v_bias,
                                                                  cu_q_len,
                                                                  cu_k_len,
                                                                  cu_block_num,
                                                                  rope_base,
                                                                  stride_b,
                                                                  stride_c,
                                                                  stride_h,
                                                                  stride_s,
                                                                  block_seq_len,
                                                                  k_offset,
                                                                  v_offset,
                                                                  transform_k,
                                                                  transform_v);
    };

    (quant_policy & QuantPolicy::kCacheKVInt8) ? invoke(int8_t{}) : invoke(T{});
}

template void invokeProcessKV(void**       blocks,
                              const half*  k,
                              const half*  v,
                              const half*  k_bias,
                              const half*  v_bias,
                              const int*   cu_q_len,
                              const int*   cu_k_len,
                              const int*   cu_block_num,
                              const float* rope_base,
                              int          stride_b,
                              int          stride_c,
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
                              cudaStream_t stream);
#if ENABLE_BF16
template void invokeProcessKV(void**             blocks,
                              const nv_bfloat16* k,
                              const nv_bfloat16* v,
                              const nv_bfloat16* k_bias,
                              const nv_bfloat16* v_bias,
                              const int*         cu_q_len,
                              const int*         cu_k_len,
                              const int*         cu_block_num,
                              const float*       rope_base,
                              int                stride_b,
                              int                stride_c,
                              int                stride_h,
                              int                stride_s,
                              int                block_seq_len,
                              int                block_k_offset,
                              int                block_v_offset,
                              int                max_q_len,
                              int                kv_head_num,
                              int                batch_size,
                              int                quant_policy,
                              const float*       quant_params_kv,
                              cudaStream_t       stream);
#endif

template<int CTA_S, int HeadDim, int WarpCnt, class T, class Tkv, class Offset, class TransformK, class TransformV>
__global__ void __launch_bounds__(128) flattenKV(T*           k,
                                                 T*           v,
                                                 const Tkv**  blocks,
                                                 const int*   cu_k_len,
                                                 const int*   cu_block_num,
                                                 const float* rope_base,
                                                 int          stride_b,
                                                 int          stride_c,
                                                 int          stride_h,
                                                 int          stride_s,
                                                 int          block_seq_len,
                                                 Offset       block_k_offset,
                                                 Offset       block_v_offset,
                                                 TransformK   transform_k,
                                                 TransformV   transform_v)
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

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int si = offset.y + s * Map::kDeltaS + token_idx;

        const int block_index  = si >> (31 - __clz(block_seq_len));
        const int block_offset = si & (block_seq_len - 1);

        const Tkv* block = blocks[block_index];

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            int       di    = offset.x + c * Map::kDeltaC;
            const int idx_k = block_k_offset + head_idx * block_seq_len * HeadDim + block_offset * HeadDim + di;
            const int idx_v = block_v_offset + head_idx * block_seq_len * HeadDim + block_offset * HeadDim + di;
            if (si < seq_len) {
                Ldg(vec_K[s][c], &block[idx_k]);
                Ldg(vec_V[s][c], &block[idx_v]);
            }
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            out_K[s][c] = transform_k(vec_K[s][c]);
            out_V[s][c] = transform_v(vec_V[s][c]);
        }
    }

    if (rope_base) {
        float base = rope_base[batch_idx];
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            FastRoPE  rope(di, std::integral_constant<int, HeadDim>{}, base, std::integral_constant<int, kVecSize>{});
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
            const int index =
                (batch_idx * stride_b + ti_beg * stride_c + si * stride_s + head_idx * stride_h) * HeadDim + di;
            if (si < seq_len) {
                Store(&k[index], out_K[s][c]);
                Store(&v[index], out_V[s][c]);
            }
        }
    }
}

template<class T>
void invokeFlattenKV(T*           k,
                     T*           v,
                     const void** blocks,
                     const int*   cu_k_len,
                     const int*   cu_block_num,
                     const float* rope_base,
                     int          stride_b,
                     int          stride_c,
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
                     cudaStream_t stream)
{
    constexpr int kWarpCnt = 4;
    constexpr int kHeadDim = 128;
    constexpr int CTA_S    = 64;

    constexpr int block = kWarpCnt * WARP_SIZE;
    const dim3    grid((max_seq_len + CTA_S - 1) / CTA_S, head_num, batch_size);

    auto invoke = [&](auto tkv) {
        using Tkv = decltype(tkv);

        ConvertKvCache<Tkv, T> transform_k{quant_params[0], quant_params[1]};
        ConvertKvCache<Tkv, T> transform_v{quant_params[2], quant_params[3]};
        flattenKV<CTA_S, kHeadDim, kWarpCnt><<<grid, block, 0, stream>>>(k,
                                                                         v,
                                                                         (const Tkv**)blocks,
                                                                         cu_k_len,
                                                                         cu_block_num,
                                                                         rope_base,
                                                                         stride_b,
                                                                         stride_c,
                                                                         stride_h,
                                                                         stride_s,
                                                                         block_seq_len,
                                                                         block_k_offset,
                                                                         block_v_offset,
                                                                         transform_k,
                                                                         transform_v);
    };

    (quant_policy & QuantPolicy::kCacheKVInt8) ? invoke(int8_t{}) : invoke(T{});
}

template void invokeFlattenKV(half*        k,
                              half*        v,
                              const void** blocks,
                              const int*   cu_k_len,
                              const int*   cu_block_num,
                              const float* rope_base,
                              int          stride_b,
                              int          stride_c,
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
                              cudaStream_t stream);

#if ENABLE_BF16
template void invokeFlattenKV(nv_bfloat16* k,
                              nv_bfloat16* v,
                              const void** blocks,
                              const int*   cu_k_len,
                              const int*   cu_block_num,
                              const float* rope_base,
                              int          stride_b,
                              int          stride_c,
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
                              cudaStream_t stream);
#endif

}  // namespace turbomind
