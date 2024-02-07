
#pragma once

#include "cta_map.h"
#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cstddef>
#include <type_traits>

namespace turbomind::attention {

template<class T_, int CTA_H_, int HeadDim, int WarpCnt>
struct Reduce {

    using T = T_;

    static constexpr int CTA_H    = CTA_H_;
    static constexpr int kWarpCnt = WarpCnt;

    static constexpr int kMaxSplitCount = WARP_SIZE;

    struct SharedStorage {
        float scale[CTA_H][kMaxSplitCount];
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

        PRAGMA_UNROLL
        for (int h = 0; h < CTA_H; h += WarpCnt) {
            const int hi = h + warp_id;

            const int split_idx = offset_k + lane_id * stride_k;

            const bool is_valid = hi < CTA_H && split_idx < split_cnt;
            const int  index    = (query_idx * head_num + head_idx + hi) * max_split_cnt + split_idx;

            const float frag_M = is_valid ? partial_M[index] : -std::numeric_limits<float>::infinity();

            float block_M = frag_M;
            PRAGMA_UNROLL
            for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
                block_M = fmaxf(block_M, __shfl_xor_sync((uint32_t)-1, block_M, mask));
            }

            const float expdiff_M = exp2f((frag_M - block_M) * exp_scale);

            const float frag_L = is_valid ? partial_L[index] : 0.f;

            float block_L = expdiff_M * frag_L;
            PRAGMA_UNROLL
            for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
                block_L += __shfl_xor_sync((uint32_t)-1, block_L, mask);
            }

            if (is_valid) {
                float scale = expdiff_M;
                if constexpr (IsFinal) {
                    scale = expdiff_M / block_L;
                }
                storage.scale[hi][lane_id] = scale;

                if constexpr (!IsFinal) {
                    partial_M[index] = block_M;
                    partial_L[index] = block_L;
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
                const int ki = si / CTA_H;
                const int hi = si % CTA_H;
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int di = d.x + c * Map::kDeltaC;
                    fn(s, c, ki, hi, di);
                }
            }
        };

        PRAGMA_UNROLL
        for (int k = 0; k < kMaxSplitCount; k += WarpCnt) {
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