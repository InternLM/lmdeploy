// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm80_hmma_16816_pack {

template<Order order>
struct GetSmemLayout {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        if constexpr (order == kRowMajor) {
            return SmemLayoutV2<M, K>{};
        }
        else {
            return SmemLayoutV2<K, M>{};
        }
    }
};

template<class T>
struct Operand_A_N {
    using Dtype = T;

    static constexpr int Pack_M = 2;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_A | Pack_M;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, kOrder, 16 * Pack_M, 16, 8, Pack_M>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_B_T {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_B | Pack_M;
    static constexpr Order kOrder = Order::kRowMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, kOrder, 16 * Pack_M, 16, 8, Pack_M>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_U {
    using Dtype = T;

    static constexpr int Pack_M = 2;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_U | Pack_M;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, kOrder, 16 * Pack_M, 16, 2, Pack_M, 4>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

}  // namespace sm80_hmma_16816_pack

template<class T>
struct GetOperand<HMMA_16816, OPERAND_A, T, kColMajor, true>: std::true_type {
    using Operand = sm80_hmma_16816_pack::Operand_A_N<T>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_B, T, kRowMajor, true>: std::true_type {
    using Operand = sm80_hmma_16816_pack::Operand_B_T<T>;
};

template<>
struct GetOperand<HMMA_16816, OPERAND_U, uint32_t, kColMajor, true>: std::true_type {
    using Operand = sm80_hmma_16816_pack::Operand_U<uint32_t>;
};

template<>
struct GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>: std::true_type {
    using Operand = sm80_hmma_16816_pack::Operand_U<uint32_t>;
};

}  // namespace turbomind::gemm