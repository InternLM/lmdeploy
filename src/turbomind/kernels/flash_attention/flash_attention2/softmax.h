/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>

#include "utils.h"

namespace flash {

template<bool zero_init = true,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1,
         typename Operator>
__device__ inline void
thread_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0>& dst, Tensor<Engine1, Layout1>& src, Operator& op)
{
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init = true,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1,
         typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op)
{
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& max)
{
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& sum)
{
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template<bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void
scale_apply_exp2(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1> const& max, const float scale)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

using namespace cute;
template<typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout>& tensor, const int max_seqlen_k, const int col_idx_offset_ = 0)
{
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id        = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
// Without the "make_coord" we get wrong results
#pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template<typename Engine, typename Layout>
inline __device__ void apply_mask_causal(Tensor<Engine, Layout>& tensor,
                                         const uint32_t          col_idx_offset_,
                                         const uint32_t          max_seqlen_k,
                                         const uint32_t          row_idx_offset_,
                                         const uint32_t          warp_row_stride)
{
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const uint32_t lane_id = threadIdx.x % 32;
    // const uint32_t row_idx_offset = row_idx_offset_ + lane_id / 4;
    const uint32_t row_idx_offset = row_idx_offset_;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const uint32_t row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const uint32_t row_idx       = row_idx_base + i * 8;
            const uint32_t col_idx_limit = std::min(max_seqlen_k, row_idx + 1);
#pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const uint32_t col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const uint32_t col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}
}  // namespace flash
