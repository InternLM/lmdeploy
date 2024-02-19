
#pragma once

#include "cta_map.h"
#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cstddef>
#include <type_traits>

namespace turbomind::attention {

template<class T_, int CTA_H_, int CTA_K_, int HeadDim, int WarpCnt>
struct Reduce {
    using T = T_;

    static constexpr int CTA_H    = CTA_H_;
    static constexpr int CTA_K    = CTA_K_;
    static constexpr int kWarpCnt = WarpCnt;

    static_assert((CTA_K & (CTA_K - 1)) == 0, "must be pow of 2");

    struct SharedStorage {
        float scale[CTA_H][CTA_K];
        float O[CTA_H][WarpCnt][HeadDim];
    };

    template<bool IsFinal>
    __device__ void operator()(T*             out,
                               float*         partial_M,
                               float*         partial_L,
                               float*         partial_O,
                               int            query_idx,
                               int            head_idx,
                               int            head_num,
                               int            split_cnt,
                               int            max_split_cnt,
                               float          exp_scale,
                               int            stride_k,
                               int            offset_k,
                               SharedStorage& storage,
                               std::integral_constant<bool, IsFinal>)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        // iterations per warp, K > 1 when CTA_K is multiple of WARP_SIZE
        constexpr int K = (CTA_K + WARP_SIZE - 1) / WARP_SIZE;
        // heads per warp iteration, M > 1 when WARP_SIZE is multiple of CTA_K
        constexpr int M = (WARP_SIZE + CTA_K - 1) / CTA_K;
        // lanes per head, a warp is processing M heads in parallel
        constexpr int L = WARP_SIZE / M;

        PRAGMA_UNROLL
        for (int h = 0; h < CTA_H; h += WarpCnt * M) {

            const int hi = h + warp_id * M + lane_id / L;

            Array<float, K> frag_M;
            Array<float, K> frag_L;

            fill(frag_M, -std::numeric_limits<float>::infinity());
            fill(frag_L, 0.f);

            PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
                const int  si   = (lane_id % L + k * L) * stride_k + offset_k;
                const int  idx  = (query_idx * head_num + head_idx + hi) * max_split_cnt + si;
                const bool mask = hi < CTA_H && si < split_cnt;
                if (mask) {
                    frag_M[k] = partial_M[idx];
                    frag_L[k] = partial_L[idx];
                }
            }

            float block_M = frag_M[0];
            PRAGMA_UNROLL
            for (int k = 1; k < K; ++k) {
                block_M = fmaxf(block_M, frag_M[k]);
            }

            PRAGMA_UNROLL
            for (int mask = L / 2; mask >= 1; mask /= 2) {
                block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
            }

            Array<float, K> expdiff_M;
            PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
                expdiff_M[k] = exp2f((frag_M[k] - block_M) * exp_scale);
            }

            float block_L{};
            PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
                block_L += expdiff_M[k] * frag_L[k];
            }

            PRAGMA_UNROLL
            for (int mask = L / 2; mask >= 1; mask /= 2) {
                block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
            }

            Array<float, K> scale;
            PRAGMA_UNROLL
            for (int k = 0; k < K; ++k) {
                scale[k] = IsFinal ? expdiff_M[k] / block_L : expdiff_M[k];
            }

            if (hi < CTA_H) {
                PRAGMA_UNROLL
                for (int k = 0; k < K; ++k) {
                    storage.scale[hi][lane_id % L + k * L] = scale[k];
                }
            }

            if constexpr (!IsFinal) {
                PRAGMA_UNROLL
                for (int k = 0; k < K; ++k) {
                    const int  si   = (lane_id % L + k * L) * stride_k + offset_k;
                    const int  idx  = (query_idx * head_num + head_idx + hi) * max_split_cnt + si;
                    const bool mask = hi < CTA_H && si < split_cnt;
                    if (mask) {
                        partial_M[idx] = block_M;
                        partial_L[idx] = block_L;
                    }
                }
            }
        }

        __syncthreads();

        constexpr int kVecSize = HeadDim / WARP_SIZE;

        using Map = RakedThreadMap<HeadDim, WarpCnt * CTA_H, kVecSize, WarpCnt>;

        static_assert(Map::kIterS == CTA_H);

        constexpr int S = Map::kIterS;
        constexpr int C = Map::kIterC;

        using Vec = Array<float, kVecSize>;

        Vec accu_O[S][C]{};
        Vec frag_O[S][C];

        const int2 d = Map::get_offset(warp_id, lane_id);

        auto for_each = [&](auto fn) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                const int si = d.y + s * Map::kDeltaS;
                const int hi = si % CTA_H;
                const int ki = si / CTA_H;
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int di = d.x + c * Map::kDeltaC;
                    fn(s, c, ki, hi, di);
                }
            }
        };

        PRAGMA_UNROLL
        for (int k = 0; k < CTA_K; k += WarpCnt) {
            for_each([&](int s, int c, int ki, int hi, int di) {
                using namespace ops;
                ki += k;
                const int  split_idx = offset_k + stride_k * ki;
                const bool mask      = split_idx < split_cnt;
                const int  offset = ((query_idx * head_num + head_idx + hi) * max_split_cnt + split_idx) * HeadDim + di;
                if (mask) {
                    Load(frag_O[s][c], &partial_O[offset]);
                    accu_O[s][c] = accu_O[s][c] + frag_O[s][c] * storage.scale[hi][ki];
                }
            });
        }

        for_each([&](int s, int c, int ki, int hi, int di) {
            Store(&storage.O[hi][ki][di], accu_O[s][c]);  //
        });

        PRAGMA_UNROLL
        for (int w = WarpCnt / 2; w > 0; w /= 2) {
            __syncthreads();
            for_each([&](int s, int c, int ki, int hi, int di) {
                using namespace ops;
                if (ki < w) {
                    (Vec&)storage.O[hi][ki][di] = (Vec&)storage.O[hi][ki][di] + (Vec&)storage.O[hi][w + ki][di];
                }
            });
        }

        for_each([&](int s, int c, int ki, int hi, int di) {
            if (ki == 0) {
                if constexpr (IsFinal) {
                    const int offset = (query_idx * head_num + head_idx + hi) * HeadDim + di;
                    Store(&out[offset], cast<T>((Vec&)storage.O[hi][ki][di]));
                }
                else {
                    const int offset =
                        ((query_idx * head_num + head_idx + hi) * max_split_cnt + offset_k) * HeadDim + di;
                    Store(&partial_O[offset], (Vec&)storage.O[hi][ki][di]);
                }
            }
        });
    }
};

}  // namespace turbomind::attention