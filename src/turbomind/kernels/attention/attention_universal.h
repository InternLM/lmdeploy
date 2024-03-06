// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"

#include "iterator.h"
#include "reduce.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <limits>
#include <type_traits>

#include "attention_params.h"

namespace turbomind {

template<class Mainloop, class BlockSeqLen, class CtaMap_>
struct AttentionUniversal {

    using T   = typename Mainloop::T;
    using Tkv = typename Mainloop::Tkv;

    using Impl   = typename Mainloop::Impl;
    using CtaMap = CtaMap_;

    using Arch = typename Impl::Arch;

    static constexpr int kWarpCount = Impl::kWarpCount;

    using ParamType = AttentionParams<T>;

    static constexpr int kHeadDim = Impl::kHeadDim;

    using FragQ = typename Impl::FragQ;
    using FragO = typename Impl::FragO;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using GmemIterK = typename Mainloop::GmemIterK;
    using GmemIterV = typename Mainloop::GmemIterV;

    using TransformK = typename Impl::TransformK;
    using TransformV = typename Impl::TransformV;

    static constexpr int CTA_H = Impl::CTA_H;
    static constexpr int CTA_Q = Impl::CTA_Q;
    static constexpr int CTA_S = Impl::CTA_S;

    using SharedStorage = typename Mainloop::SharedStorage;

    __device__ __host__ static bool need_separate_reduce(int max_split_cnt)
    {
        if constexpr (CTA_Q > 1) {
            // using `max_split_cnt > 1` here make the kernel slightly slower
            return true;
        }
        else {
            return CTA_H * max_split_cnt > 32;
        }
    }

    __device__ void Prologue(const ParamType& params,
                             T*               smem_Q,
                             FragQ&           frag_Q,
                             int              qi_begin,
                             int              qi_end,
                             int              query_idx,
                             int              head_idx,
                             int              kv_head_idx,
                             int              batch_idx,
                             Tkv**            block_ptrs,
                             BlockSeqLen      block_seq_len,
                             int              local_k_offset,
                             int              local_v_offset,
                             int              history_len,
                             int              warp_id,
                             int              lane_id)
    {
        constexpr bool kProcessKV = CTA_Q == 1;

        using Map = typename Impl::ThreadMapQ;

        constexpr int kVecSize = Map::kAccessC;

        using Vec = Array<T, kVecSize>;

        constexpr int ITER_C = Map::kIterC;
        constexpr int ITER_S = Map::kIterS;

        Vec vec_Q[ITER_S][ITER_C]{};
        Vec vec_K[ITER_S][ITER_C];
        Vec vec_V[ITER_S][ITER_C];

        const int2 offset = Map::get_offset(warp_id, lane_id);

        // Load Q
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS;
            const int hi = si % CTA_H + head_idx;
            const int qi = si / CTA_H + qi_begin;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di    = offset.x + c * Map::kDeltaC;
                const int q_idx = qi * params.stride + hi * kHeadDim + di;
                const int k_idx = qi * params.stride + kv_head_idx * kHeadDim + di;
                if (qi < qi_end) {
                    Ldg(vec_Q[s][c], &params.q[q_idx]);
                    if constexpr (kProcessKV) {
                        Ldg(vec_K[s][c], &params.k[k_idx]);
                        Ldg(vec_V[s][c], &params.v[k_idx]);
                    }
                }
            }
        }

        Vec bias_Q[ITER_C];
        Vec bias_K[ITER_C];
        Vec bias_V[ITER_C];

        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di    = offset.x + c * Map::kDeltaC;
            const int q_idx = head_idx * kHeadDim + di;
            const int k_idx = kv_head_idx * kHeadDim + di;
            if (params.q_bias) {
                Ldg(bias_Q[c], &params.q_bias[q_idx]);
            }
            if constexpr (kProcessKV) {
                if (params.k_bias) {
                    Ldg(bias_K[c], &params.k_bias[k_idx]);
                }
                if (params.v_bias) {
                    Ldg(bias_V[c], &params.v_bias[k_idx]);
                }
            }
        }

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                using namespace ops;
                if (params.q_bias) {
                    vec_Q[s][c] = vec_Q[s][c] + bias_Q[c];
                }
                if constexpr (kProcessKV) {
                    if (params.k_bias) {
                        vec_K[s][c] = vec_K[s][c] + bias_K[c];
                    }
                    if (params.v_bias) {
                        vec_V[s][c] = vec_V[s][c] + bias_V[c];
                    }
                }
            }
        }

        const float rope_base = params.rope_theta ? params.rope_theta[batch_idx] : params.rotary_embedding_base;
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            FastRoPE  rope(di,
                          std::integral_constant<int, kHeadDim>{},
                          rope_base,
                          params.rope_ti_scale,
                          std::integral_constant<int, kVecSize>{});
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = (offset.y + s * Map::kDeltaS) / CTA_H + query_idx + history_len;
                rope.apply(vec_Q[s][c], ti);
                if constexpr (kProcessKV) {
                    static_assert(ITER_S == 1);
                    rope.apply(vec_K[0][c], ti);
                }
            }
        }

        if (params.use_logn_attn) {
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int   ti = (offset.y + s * Map::kDeltaS) / CTA_H + query_idx + history_len;
                LogNScaling logn_scaling(ti, params.max_position_embeddings);
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    logn_scaling.apply(vec_Q[c][s]);
                }
            }
        }

        if constexpr (kProcessKV) {
            static_assert(ITER_S == 1);
            const int              qi           = offset.y / CTA_H;
            const int              ti           = history_len;
            const int              block_index  = ti >> (31 - __clz(block_seq_len));
            const int              block_offset = ti & (block_seq_len - 1);
            Tkv*                   block        = block_ptrs[block_index];
            ConvertKvCache<T, Tkv> transform_K{params.kv_quant_params[0], params.kv_quant_params[1]};
            ConvertKvCache<T, Tkv> transform_V{params.kv_quant_params[2], params.kv_quant_params[3]};
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di    = offset.x + c * Map::kDeltaC;
                const int idx_k = local_k_offset + block_offset * kHeadDim + di;
                const int idx_v = local_v_offset + block_offset * kHeadDim + di;
                if (qi < CTA_Q) {
                    Store(&block[idx_k], transform_K(vec_K[0][c]));
                    Store(&block[idx_v], transform_V(vec_V[0][c]));
                }
            }
            __syncthreads();
        }

        using SmemLayoutQ = typename Impl::SmemLayoutQ;

        SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};

        // Store to shared memory
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS;
            const int hi = si % CTA_H;
            const int qi = si / CTA_H;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                if (qi < CTA_Q && hi < CTA_H) {
                    Store(&sQ(si, di), vec_Q[s][c]);
                }
            }
        }

        __syncthreads();

        Impl::TransformQ(smem_Q, frag_Q);
    }

    __device__ void operator()(const ParamType& params, const CtaMap& cta_map, char* smem_buf)
    {
        // [q, h, b]
        const int query_idx = cta_map.query_idx() * CTA_Q;
        const int head_idx  = cta_map.head_idx() * CTA_H;
        const int batch_idx = cta_map.batch_idx();
        const int split_idx = cta_map.split_idx();
        const int split_cnt = cta_map.split_count();

        // early exit if finished flag is set
        if (params.finished[batch_idx]) {
            return;
        }

        const int qi_begin = params.cu_q_len[batch_idx] + query_idx;  // global offset into `cu_seqlens`
        const int qi_end   = params.cu_q_len[batch_idx + 1];

        if (qi_begin >= qi_end) {
            return;
        }

        const int input_len = qi_end - (qi_begin - query_idx);

        const BlockSeqLen block_seq_len = [&]() -> BlockSeqLen {
            if constexpr (std::is_integral_v<BlockSeqLen>) {
                return params.kv_cache_block_size;
            }
            else {
                return {};
            }
        }();

        SharedStorage& storage = *(SharedStorage*)smem_buf;

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int kv_head_idx = head_idx * params.num_kv_heads / params.num_heads;

        // [L, 2, H, s, D]
        const int local_k_offset = params.key_offset + kv_head_idx * block_seq_len * kHeadDim;
        const int local_v_offset = params.val_offset + kv_head_idx * block_seq_len * kHeadDim;

        const auto k_cache_ptrs = (Tkv**)params.k_cache_block_ptrs + params.cu_block_cnts[batch_idx];

        const int context_len = params.cu_k_len[batch_idx + 1] - params.cu_k_len[batch_idx];
        const int history_len = context_len - input_len;

        const int tile_count = (history_len + min(query_idx + CTA_Q, input_len) + CTA_S - 1) / CTA_S;

        const int tile_per_split = (tile_count + split_cnt - 1) / split_cnt;
        const int iter_begin     = tile_per_split * split_idx;
        const int iter_end       = min(iter_begin + tile_per_split, tile_count);

        if (iter_begin >= tile_count) {
            return;
        }

        FragQ frag_Q;
        Prologue(params,
                 storage.Q,
                 frag_Q,
                 qi_begin,
                 qi_end,
                 query_idx,
                 head_idx,
                 kv_head_idx,
                 batch_idx,
                 k_cache_ptrs,
                 block_seq_len,
                 local_k_offset,
                 local_v_offset,
                 history_len,
                 warp_id,
                 lane_id);

        GmemIterK gmem_K{warp_id, lane_id};
        GmemIterV gmem_V{warp_id, lane_id};

        TransformK transform_K{params.kv_quant_params[0], params.kv_quant_params[1]};
        TransformV transform_V{params.kv_quant_params[2], params.kv_quant_params[3]};

        /// TODO: move the branching to template parameter
        auto block_iter = [&] {
            if constexpr (CTA_Q == 1) {
                // [H, s, D]
                auto block_ptrs      = (const Tkv**)k_cache_ptrs + iter_begin * CTA_S / block_seq_len;
                int  local_id_offset = iter_begin % (block_seq_len / CTA_S);
                return BlockTileIter<Tkv, CTA_S, kHeadDim, BlockSeqLen>{
                    block_ptrs, block_seq_len, {local_k_offset, local_v_offset}, local_id_offset};
            }
            else {
                // [H, 2, cuS, D]
                const int skip_k   = params.cu_k_len[0];
                const int sum_k    = params.cu_k_len[params.batch_size] - skip_k;
                const int begin    = params.cu_k_len[batch_idx] - skip_k + iter_begin * CTA_S;
                const int stride_h = 2 * sum_k * kHeadDim;
                return LinearTileIter<Tkv, CTA_S, kHeadDim>{
                    (const Tkv*)params.kv + kv_head_idx * stride_h + begin * kHeadDim, sum_k * kHeadDim};
            }
        }();

        __align__(16) FragO frag_O{};

        FragL frag_L{};
        FragM frag_M;
        fill(frag_M, -std::numeric_limits<float>::infinity());

        __syncthreads();

        const int offset_Q = history_len + query_idx - iter_begin * CTA_S;
        const int max_step = context_len - iter_begin * CTA_S;

        int tile_iter = iter_end - iter_begin - 1;
        int mask_iter = (CTA_Q + CTA_S - 1) / CTA_S + 1;

        Mainloop mainloop;
        mainloop(frag_Q,
                 gmem_K,
                 gmem_V,
                 transform_K,
                 transform_V,
                 block_iter,
                 frag_O,
                 frag_M,
                 frag_L,
                 offset_Q,
                 max_step,
                 tile_iter,
                 mask_iter,
                 params.inv_sqrt_dh,
                 storage,
                 StoreS(params, query_idx, head_idx, batch_idx, context_len));

        if constexpr (Impl::kWarpCntS > 1) {
            Impl::Merge(frag_O, frag_M, frag_L, params.inv_sqrt_dh, storage);
        }

        if (iter_begin == 0 && iter_end == tile_count) {
            StoreO(frag_O, frag_L, qi_begin, qi_end, head_idx, params, storage);
        }
        else {
            StorePartial(frag_O, frag_M, frag_L, qi_begin, qi_end, head_idx, split_idx, params, storage);

            if (iter_end == tile_count) {  // store actual split count
                for (int ti = qi_begin + threadIdx.x; ti < qi_end; ti += kWarpCount * WARP_SIZE) {
                    params.split_cnt[ti] = split_idx + 1;
                }
            }

            if (need_separate_reduce(cta_map.split_count())) {
                return;
            }
            else {
                Reduce(qi_begin, head_idx, split_idx, iter_end != tile_count, params, cta_map, smem_buf);
            }
        }
    }

    __device__ void Reduce(int              qi_begin,
                           int              head_idx,
                           int              split_idx,
                           bool             is_last,
                           const ParamType& params,
                           const CtaMap&    cta_map,
                           char*            smem_buf)
    {
        // Note: `head_idx` is cta_map.head_idx() * CTA_H
        const auto index = (cta_map.batch_idx() * params.num_heads + cta_map.head_idx()) * params.max_split_k;
        const auto locks = params.locks + index;

        if (is_last) {
            sem_post(&locks[split_idx], 1, threadIdx.x == 0);
        }
        else {
            const int split_count = split_idx + 1;

            sem_wait_many(&locks[threadIdx.x], split_count - 1, threadIdx.x < split_count - 1);

            using Reduce = attention::Reduce<T, CTA_H, 32, kHeadDim, kWarpCount>;

            Reduce reduce_op;
            reduce_op(params.out,
                      params.partial_M,
                      params.partial_L,
                      params.partial_O,
                      qi_begin,
                      head_idx,
                      params.num_heads,
                      split_idx + 1,
                      params.max_split_k,
                      params.inv_sqrt_dh,
                      1,
                      0,
                      *(typename Reduce::SharedStorage*)smem_buf,
                      std::true_type{});

            if (threadIdx.x < split_idx) {
                locks[threadIdx.x] = 0;
            }
        }
    }

    __device__ void StoreO(FragO&           frag_O,
                           FragL&           frag_L,
                           int              qi_begin,
                           int              qi_end,
                           int              head_idx,
                           const ParamType& params,
                           SharedStorage&   storage)
    {
        Impl::StoreO<true>(frag_O, frag_L, storage, [&](int hi, int qi, int di, const auto& vec) {
            if (qi_begin + qi < qi_end) {
                const int offset = (qi_begin + qi) * params.num_heads * kHeadDim + (head_idx + hi) * kHeadDim + di;
                Store(&params.out[offset], cast<T>(vec));
            }
        });
    }

    __device__ auto StoreS(const ParamType& params,
                           const int&       query_idx,
                           const int&       head_idx,
                           const int&       batch_idx,
                           const int&       max_context_len)
    {
        return [&](auto& frag_S, int offset_K) {
            Impl::ForeachS(frag_S, [&](int hi, int qi, int si, int ri, float score) {
                qi += query_idx;
                si += offset_K;
                if (qi < params.max_q_len && si < max_context_len) {
                    params.qk[batch_idx * params.num_heads * params.max_q_len * max_context_len
                              + (head_idx + hi) * params.max_q_len * max_context_len + qi * max_context_len + si] =
                        score;
                }
            });
        };
    }

    __device__ void StorePartial(FragO&           frag_O,
                                 FragM&           frag_M,
                                 FragL&           frag_L,
                                 int              qi_begin,
                                 int              qi_end,
                                 int              head_idx,
                                 int              split_idx,
                                 const ParamType& params,
                                 SharedStorage&   storage)
    {
        auto get_index = [&](int hi, int qi) {
            // [B, H, k, D]
            return (qi_begin + qi) * params.num_heads * params.max_split_k + (head_idx + hi) * params.max_split_k
                   + split_idx;
        };

        Impl::StoreO<false>(frag_O, frag_L, storage, [&](int hi, int qi, int di, const auto& vec) {
            if (qi_begin + qi < qi_end) {
                Store(&params.partial_O[get_index(hi, qi) * kHeadDim + di], vec);
            }
        });

        Impl::ForeachML(frag_M, frag_L, [&](int hi, int qi, int ri, float M, float L) {
            const int index = get_index(hi, qi);
            if (qi_begin + qi < qi_end && ri == 0) {
                // printf("ML %2d %2d %f %f\n", split_idx, head_idx + hi, M, L);
                params.partial_M[index] = M;
                params.partial_L[index] = L;
            }
        });
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern __shared__ char smem_buf[];

template<class Attention>
__global__ void attention_kernel(typename Attention::ParamType params, typename Attention::CtaMap cta_map)
{
#if __CUDA_ARCH__
    if constexpr (Attention::Arch::is_compatible(__CUDA_ARCH__ / 10)) {
        Attention{}(params, cta_map, smem_buf);
    }
#endif
}

}  // namespace turbomind
