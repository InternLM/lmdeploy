// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

struct VoidGmemIter {
    static constexpr int ITER_S = 0;
    template<class P>
    __device__ VoidGmemIter(P, int, int2, int2)
    {
    }
    __device__ void ClearSmem() {}
    __device__ void Prefetch(int, int, bool) {}
    __device__ void Prefetch(bool) {}
    __device__ void Advance() {}
    int*            smem_data_;
    bool            g_mask{false};
};

struct GetGmemIter {
    template<class Operand, class Iterator, class SmemLayout, int M, int K, int WARPS>
    static constexpr auto
    apply(basic_type<Operand>, basic_type<Iterator>, basic_type<SmemLayout>, pair<M, K>, constant<WARPS>)
    {
        using Dtype = typename Operand::Dtype;

        constexpr int kAccessSize =
            std::min<int>(128 / bitsof<Dtype>, std::max<int>(32 / bitsof<Dtype>, M * K / (WARPS * WARP_SIZE)));

        constexpr int2 kCS = mk2cs<Operand::kOrder>(M, K);

        using GmemIter = typename Iterator::template Type<Dtype,
                                                          gemm::ThreadMap_V2<kCS.x, kCS.y, kAccessSize, Blocked, WARPS>,
                                                          SmemLayout,
                                                          Operand::kPack,
                                                          Operand::kOrder,
                                                          0,
                                                          0>;
        return type_c<GmemIter>;
    }
};

}  // namespace turbomind::gemm