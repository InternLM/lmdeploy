// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

namespace turbomind::gemm {

struct VoidGmemIter {
    static constexpr int ITER_S = 0;
    template<class P>
    __device__ VoidGmemIter(P, int, int2, int2, int2)
    {
    }
    __device__ void ClearSmem() {}
    __device__ void Prefetch(int, int, bool) {}
    __device__ void Prefetch(bool) {}
    __device__ void Advance() {}
    int*            smem_data_;
};

struct GetGmemIter {
    template<class Operand, class Iterator, class SmemLayout, int C, int S, int WARPS>
    static constexpr auto apply(basic_type<Operand>, basic_type<Iterator>, basic_type<SmemLayout>, pair<C, S>, constant<WARPS>)
    {
        using GmemIter = typename Iterator::template Type<typename Operand::Dtype,
                                                          gemm::ThreadMap<C, S, 8, WARPS>,
                                                          SmemLayout,
                                                          Operand::kPack,
                                                          Operand::kOrder,
                                                          0,
                                                          0>;
        return type_c<GmemIter>;
    }
};

}  // namespace turbomind::gemm