// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "cta_map.h"

#include "../gemm_s_f16/common.h"

#include <iostream>
#include <type_traits>

namespace turbomind::attention {

template<class Kernel, bool IsFinal>
__global__ void reduce_kernel(typename Kernel::T* out,
                              float*              partial_M,
                              float*              partial_L,
                              float*              partial_O,
                              int*                signals,
                              const int*          split_cnt_,
                              int                 max_split_cnt,
                              int                 head_num,
                              float               exp_scale,
                              int                 stride_k)
{
    extern __shared__ char smem[];

    const int head_idx  = ReduceCtaMap::head_idx();
    const int query_idx = ReduceCtaMap::query_idx();
    const int chunk_idx = ReduceCtaMap::split_idx();

    const int split_cnt = split_cnt_[query_idx];

    const int chunk_cnt = (split_cnt + 31) / 32;

    if (chunk_idx >= chunk_cnt) {
        return;
    }

    Kernel reduce{};
    reduce(out,
           partial_M,
           partial_L,
           partial_O,
           query_idx,
           head_idx,
           head_num,
           split_cnt,
           max_split_cnt,
           exp_scale,
           stride_k,
           chunk_idx * 32,
           *(typename Kernel::SharedStorage*)smem,
           std::integral_constant<bool, IsFinal>{});
}

template<class Kernel>
void invokeReduce(typename Kernel::T* out,
                  float*              partial_M,
                  float*              partial_L,
                  float*              partial_O,
                  int*                signals,
                  const int*          split_cnt,
                  int                 max_split_cnt,
                  int                 dyn_split_cnt,
                  int                 query_num,
                  int                 head_num,
                  float               exp_scale,
                  cudaStream_t        stream)
{
    static constexpr size_t kDynamicSmemSize = sizeof(typename Kernel::SharedStorage);
    static_assert(kDynamicSmemSize < (48 << 10));

    auto invoke = [&](auto is_final, int d_split_cnt, int stride_k) {
        const dim3 block = Kernel::kWarpCnt * 32;
        const dim3 grid  = ReduceCtaMap::get_grid_shape(query_num, head_num, d_split_cnt);

        reduce_kernel<Kernel, is_final><<<grid, block, kDynamicSmemSize, stream>>>(
            out, partial_M, partial_L, partial_O, signals, split_cnt, max_split_cnt, head_num, exp_scale, stride_k);
    };

    invoke(std::false_type{}, dyn_split_cnt, 1);
    invoke(std::true_type{}, 32, 32);
}

}  // namespace turbomind::attention