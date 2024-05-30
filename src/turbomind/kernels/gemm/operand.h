// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct VoidOperand {
    using Dtype = int;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemLayout = SmemLayoutV2<1, 1>;
    using SmemCopy   = VoidSmemCopy<1, 1>;
    using GmemIter   = VoidIterator;
};

// CPO for getting specific operand templates
template<MMA_Tag mma, Op_Tag op, Order order>
struct GetOperand: std::false_type {};

}  // namespace turbomind::gemm