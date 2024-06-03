// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"

namespace turbomind::gemm {

namespace sm80_hmma_16816_pack {

struct GetSmemLayout {
    template<int C, int S>
    static constexpr auto apply(pair<C, S>)
    {
        return SmemLayoutV2<S, C>{};
    }
};

template<class T>
struct Operand_A_N {
    using Dtype = T;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_A | 1;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, 8, 1>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_B_T {
    using Dtype = T;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_B | 1;
    static constexpr Order kOrder = Order::kRowMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, 8, 1>;

    using GetSmemLayout = GetSmemLayout;
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

}  // namespace turbomind::gemm