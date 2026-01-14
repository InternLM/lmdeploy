// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <limits>
#include <type_traits>

#include "quantization.h"

#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/math.h"

#include "attention_params.h"

namespace turbomind {

template<class Arch_, class Mainloop, class CacheIteratorFactory_, class CtaMap_>
struct AttentionUniversal {

    using T   = typename Mainloop::T;
    using Tkv = typename Mainloop::Tkv;

    using Impl = typename Mainloop::Impl;

    using CacheIteratorFactory = CacheIteratorFactory_;
    using CtaMap               = CtaMap_;

    using Arch = Arch_;

    static constexpr int kWarpCount = Impl::kWarpCount;

    using ParamType = AttentionParams<T>;

    static constexpr int kHeadDim = Impl::kHeadDim;

    using FragQ = typename Impl::FragQ;
    using FragO = typename Impl::FragO;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using GmemIterK = typename Mainloop::GmemIterK;
    using GmemIterV = typename Mainloop::GmemIterV;

    static constexpr int CTA_H = Impl::CTA_H;
    static constexpr int CTA_Q = Impl::CTA_Q;
    static constexpr int CTA_S = Impl::CTA_S;

    using SharedStorage = typename Mainloop::SharedStorage;

    static constexpr bool kProcessKV = CTA_Q == 1;

    const int q_group_size_;
    const int q_head_per_cta_;
    const int cta_per_q_group_;

    // past-the-end hi of the CTA
    int hi_end_{1};

    __device__ bool check_h(int hi)
    {
        if constexpr (CTA_Q > 1) {
            // bypass the check for prefill kernels since `hi == 0` constantly
            return true;
        }
        else {
            return hi < hi_end_;
        }
    }

    template<class VecQ, class VecKV>
    __device__ void ApplyBias(
        VecQ& vec_Q, VecKV& vec_K, VecKV& vec_V, const ParamType& params, int head_idx, int kv_head_idx, int2 offset)
    {
        using Map              = typename Impl::ThreadMapQ;
        constexpr int kVecSize = Map::kAccessC;
        constexpr int ITER_C   = Map::kIterC;
        constexpr int ITER_S   = Map::kIterS;
        if constexpr (kProcessKV) {
            Array<T, kVecSize> bias_K[ITER_C];
            Array<T, kVecSize> bias_V[ITER_C];
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di    = offset.x + c * Map::kDeltaC;
                const int k_idx = kv_head_idx * kHeadDim + di;
                if (params.k_bias) {
                    Ldg(bias_K[c], &params.k_bias[k_idx]);
                }
                if (params.v_bias) {
                    Ldg(bias_V[c], &params.v_bias[k_idx]);
                }
            }
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                using namespace ops;
                if (params.k_bias) {
                    vec_K[0][c] = vec_K[0][c] + bias_K[c];
                }
                if (params.v_bias) {
                    vec_V[0][c] = vec_V[0][c] + bias_V[c];
                }
            }
        }

        if constexpr (CTA_H == 1) {
            Array<T, kVecSize> bias_Q[ITER_C];
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di    = offset.x + c * Map::kDeltaC;
                const int q_idx = head_idx * kHeadDim + di;
                if (params.q_bias) {
                    Ldg(bias_Q[c], &params.q_bias[q_idx]);
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
                }
            }
        }
        else if constexpr (CTA_Q == 1) {
            Array<T, kVecSize> bias_Q[ITER_S][ITER_C];
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int hi = offset.y + s * Map::kDeltaS;
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    const int di    = offset.x + c * Map::kDeltaC;
                    const int q_idx = (head_idx + hi) * kHeadDim + di;
                    if (params.q_bias && check_h(hi)) {
                        Ldg(bias_Q[s][c], &params.q_bias[q_idx]);
                    }
                }
            }
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    using namespace ops;
                    if (params.q_bias) {
                        vec_Q[s][c] = vec_Q[s][c] + bias_Q[s][c];
                    }
                }
            }
        }
        else {
            static_assert(CTA_Q == 1 || CTA_H == 1);
        }
    }

    template<class Iterator>
    __device__ void Prologue(const ParamType& params,
                             T*               smem_Q,
                             FragQ&           frag_Q,
                             int              qi_begin,
                             int              qi_end,
                             int              query_idx,
                             int              head_idx,
                             int              kv_head_idx,
                             int              batch_idx,
                             int              history_len,
                             Iterator&        iterator,
                             int              warp_id,
                             int              lane_id)
    {

        using Map = typename Impl::ThreadMapQ;

        constexpr int kVecSize = Map::kAccessC;

        using Vec = Array<T, kVecSize>;

        constexpr int ITER_C = Map::kIterC;
        constexpr int ITER_S = Map::kIterS;

        Vec vec_Q[ITER_S][ITER_C]{};  // [QxH, D]
        Vec vec_K[1][ITER_C];
        Vec vec_V[1][ITER_C];

        const int2 offset = Map::get_offset(warp_id, lane_id);

        // Load Q
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS;
            const int hi = si % CTA_H + head_idx;
            const int qi = si / CTA_H + qi_begin;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int     di    = offset.x + c * Map::kDeltaC;
                const int64_t q_idx = qi * params.stride + hi * kHeadDim + di;
                const int64_t k_idx = qi * params.stride + kv_head_idx * kHeadDim + di;
                if (qi < qi_end) {
                    if (check_h(si % CTA_H)) {
                        Ldg(vec_Q[s][c], &params.q[q_idx]);
                    }
                    if constexpr (kProcessKV) {  // duplicate loads in s
                        if (s == 0) {
                            Ldg(vec_K[0][c], &params.k[k_idx]);
                            Ldg(vec_V[0][c], &params.v[k_idx]);
                        }
                    }
                }
            }
        }

        ApplyBias(vec_Q, vec_K, vec_V, params, head_idx, kv_head_idx, offset);

        FastRoPE rope(params.rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = (offset.y + s * Map::kDeltaS) / CTA_H + query_idx + history_len;
                rope.apply(vec_Q[s][c], ti);
                if constexpr (kProcessKV) {
                    if (s == 0) {
                        rope.apply(vec_K[0][c], ti);
                    }
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
                    logn_scaling.apply(vec_Q[s][c]);
                }
            }
        }

        if constexpr (kProcessKV) {
            const int qi = offset.y / CTA_H;
            const int ti = history_len;

            int local_ti, local_ti_rank;
            local_ti = params.cp_size.divmod(local_ti_rank, ti);

            Array<T, 2> param_K[1];
            Array<T, 2> param_V[1];

            if constexpr (!std::is_same_v<T, Tkv>) {
                warp_stats<Map::kWarpThreadC>(param_K, vec_K, bitsof<Tkv>);
                warp_stats<Map::kWarpThreadC>(param_V, vec_V, bitsof<Tkv>);
            }

            Array<Tkv, kVecSize> out_K[1][ITER_C];
            Array<Tkv, kVecSize> out_V[1][ITER_C];

            ConvertKvCache<T, Tkv> conv_K{param_K[0][0], param_K[0][1]};
            ConvertKvCache<T, Tkv> conv_V{param_V[0][0], param_V[0][1]};
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                out_K[0][c] = conv_K(vec_K[0][c]);
                out_V[0][c] = conv_V(vec_V[0][c]);
            }

            iterator.block_head_.with(
                iterator.block_ptrs_, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                    if (local_ti_rank != params.cp_rank) {
                        return;
                    }
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        const int di = offset.x + c * Map::kDeltaC;
                        if (qi < CTA_Q) {
                            Store(&k_cache[di], out_K[0][c]);
                            Store(&v_cache[di], out_V[0][c]);
                        }
                    }
                    if constexpr (!std::is_same_v<T, Tkv>) {
                        if (qi < CTA_Q && offset.x == 0) {
                            StoreQuantParam<Tkv>(k_param, param_K[0]);
                            StoreQuantParam<Tkv>(v_param, param_V[0]);
                        }
                    }
                });

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

    __device__ AttentionUniversal(int q_group_size, int q_head_per_cta, int cta_per_q_group):
        q_group_size_{q_group_size}, q_head_per_cta_{q_head_per_cta}, cta_per_q_group_{cta_per_q_group}
    {
    }

    __device__ void
    operator()(const ParamType& params, CacheIteratorFactory& cache_iter_factory, const CtaMap& cta_map, char* smem_buf)
    {
        // [q, h, b]
        const int query_idx = cta_map.query_idx() * CTA_Q;  // Q offset of this sequence
        const int batch_idx = cta_map.batch_idx();
        const int split_idx = cta_map.split_idx();
        const int split_cnt = cta_map.split_count();

        int head_idx;
        int kv_head_idx;

        if constexpr (CTA_H == 1) {
            head_idx    = cta_map.head_idx();
            kv_head_idx = head_idx / q_group_size_;
        }
        else {
            int cta_h_idx = cta_map.head_idx();
            int local_idx = cta_h_idx % cta_per_q_group_ * q_head_per_cta_;
            kv_head_idx   = cta_h_idx / cta_per_q_group_;
            head_idx      = kv_head_idx * q_group_size_ + local_idx;
            hi_end_       = q_group_size_ - local_idx;
        }

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

        SharedStorage& storage = *(SharedStorage*)smem_buf;

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int context_len = params.cu_k_len[batch_idx + 1] - params.cu_k_len[batch_idx];
        const int history_len = context_len - input_len;

        auto get_cp_len = [&](int length, int rank) -> int {
            int local_ti, local_ti_rank;
            local_ti = params.cp_size.divmod(local_ti_rank, length);
            return (local_ti + (local_ti_rank > rank ? 1 : 0));
        };

        const int last_K = history_len + min(query_idx + CTA_Q, input_len);
        const int last_K_tile =
            (get_cp_len(last_K, 0) - 1) / CTA_S + 1;  // past-the-end index to past-the-end tile index conversion

        const int first_K      = max(history_len + query_idx - (params.window_size - 1), 0);
        const int first_K_tile = get_cp_len(first_K, 0) / CTA_S;

        const int tile_count = last_K_tile - first_K_tile;

        /// FIXME: This scheme produce splits less than expected
        const int tile_per_split = cdiv(tile_count, split_cnt);
        const int iter_begin     = tile_per_split * split_idx;
        const int iter_end       = min(iter_begin + tile_per_split, tile_count);

        if (iter_begin >= iter_end) {
            return;
        }

        auto cache_iter = cache_iter_factory.Create(batch_idx, kv_head_idx);

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
                 history_len,
                 cache_iter,
                 warp_id,
                 lane_id);

        __align__(16) FragO frag_O{};

        FragL frag_L{};
        FragM frag_M;
        fill(frag_M, -std::numeric_limits<float>::infinity());

        __syncthreads();

        const int offset_Q = history_len + query_idx;
        const int offset_K = (first_K_tile + iter_end - 1) * CTA_S;

        // This is for avoiding OOB access only
        const int max_K = min(get_cp_len(context_len, params.cp_rank), (first_K_tile + iter_end) * CTA_S);

        int tile_iter = iter_end - iter_begin;

        //    min(Q) >= max(K)
        // -> offset_Q >= offset_K + CTA_S - x * CTA_S
        // -> x * CTA_S >= offset_K - offset_Q + CTA_S
        int mask_iter_back = cdiv(max(0, offset_K - offset_Q + CTA_S), CTA_S);
        //    max(Q) < min(K) + w
        // -> offset_Q + CTA_Q - 1 < offset_K - tile_iter * CTA_S + x * CTA_S + w
        // -> x * CTA_S >= offset_Q + CTA_Q - offset_K + tile_iter * CTA_S - w
        int mask_iter_front = cdiv(max(0, offset_Q + CTA_Q - offset_K + tile_iter * CTA_S - params.window_size), CTA_S);

        if (params.cp_size > 1) {
            mask_iter_back =
                cdiv(max(0, params.cp_size * (offset_K + CTA_S) - offset_Q + params.cp_rank), params.cp_size * CTA_S);
            mask_iter_front = cdiv(max(0,
                                       offset_Q + CTA_Q - params.window_size - params.cp_rank
                                           - params.cp_size * (offset_K - tile_iter * CTA_S)),
                                   params.cp_size * CTA_S);
        }

#if 0
        if (threadIdx.x == 0) {
            printf(
                "tile count: %d, tile per iter: %d, range_Q: [%d, %d), offset_K: %d, max_K: %d, tile_iter: %d, range_K: [%d, %d), range_K_tiles: [%d, %d), mask_iter: %d, mask_iter_front: %d\n",
                tile_count,
                tile_per_split,
                offset_Q,
                offset_Q + min(query_idx + CTA_Q, input_len),
                offset_K,
                max_K,
                tile_iter,
                first_K,
                last_K,
                first_K_tile * CTA_S,
                last_K_tile * CTA_S,
                mask_iter_back,
                mask_iter_front);
        }
#endif

        cache_iter.SetTile(first_K_tile + iter_end - 1);

        Mainloop mainloop;
        mainloop.SetCpInfo(params.cp_size, params.cp_rank);
        mainloop(frag_Q,
                 cache_iter,
                 frag_O,
                 frag_M,
                 frag_L,
                 offset_Q,
                 offset_K,
                 max_K,
                 tile_iter,
                 mask_iter_back,
                 mask_iter_front,
                 params.window_size,
                 params.inv_sqrt_dh,
                 storage,
                 StoreS(params, query_idx, head_idx, batch_idx, context_len));

        Impl::Merge(frag_O, frag_M, frag_L, params.inv_sqrt_dh, storage);

        if (params.sinks && iter_end == tile_count && params.cp_rank == 0) {
            Impl::ForeachML(frag_M, frag_L, [&](int hi, int qi, int ri, float& M, float& L) {
                if (check_h(hi) && M != -std::numeric_limits<float>::infinity()) {
                    auto sink = (float)params.sinks[head_idx + hi];
                    L += expf(sink - M * params.scale_sinks);
                }
            });
        }

        if (split_cnt > 1 && iter_end == tile_count && head_idx == 0) {
            // Store actual split count, only used by separate reduction kernel
            for (int ti = threadIdx.x; ti < CTA_Q; ti += kWarpCount * WARP_SIZE) {
                if (qi_begin + ti < qi_end) {
                    params.split_cnt[qi_begin + ti] = split_idx ? split_idx + 1 : (params.cp_size > 1 ? 1 : 0);
                }
            }
        }

        if (iter_begin == 0 && iter_end == tile_count && params.cp_size == 1) {
            StoreO(frag_O, frag_L, qi_begin, qi_end, head_idx, params, storage);
        }
        else {
            StorePartial(frag_O, frag_M, frag_L, split_cnt, qi_begin, qi_end, head_idx, split_idx, params, storage);
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
            if (qi_begin + qi < qi_end && check_h(hi)) {
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
                if (qi < params.max_q_len && si < max_context_len && check_h(hi)) {
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
                                 int              split_cnt,
                                 int              qi_begin,
                                 int              qi_end,
                                 int              head_idx,
                                 int              split_idx,
                                 const ParamType& params,
                                 SharedStorage&   storage)
    {
        auto get_index = [&](int hi, int qi) {
            // [B, H, k, D]
            return (qi_begin + qi - params.offset_q) * params.num_heads * params.max_split_k
                   + (head_idx + hi) * params.max_split_k + split_idx;
        };

        Impl::StoreO<false>(frag_O, frag_L, storage, [&](int hi, int qi, int di, const auto& vec) {
            if (qi_begin + qi < qi_end && check_h(hi)) {
                Store(&params.partial_O[get_index(hi, qi) * kHeadDim + di], vec);
            }
        });

        Impl::ForeachML(frag_M, frag_L, [&](int hi, int qi, int ri, float M, float L) {
            const int index = get_index(hi, qi);
            if (qi_begin + qi < qi_end && ri == 0 && check_h(hi)) {
                Store(&params.partial_ML[index * 2], Array<float, 2>{M, L});
            }
        });
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void attention_kernel(typename Kernel::ParamType            params,
                                 typename Kernel::CacheIteratorFactory cache_iter_factory,
                                 typename Kernel::CtaMap               cta_map,
                                 int                                   q_group_size,
                                 int                                   q_head_per_cta,
                                 int                                   cta_per_q_group)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel{q_group_size, q_head_per_cta, cta_per_q_group}(params, cache_iter_factory, cta_map, smem_buf);
    }
#endif
}

}  // namespace turbomind
