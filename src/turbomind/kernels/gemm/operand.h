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

struct VoidOperandConst {
    template<class, int, int, int, int, bool>
    using type = VoidOperand;
};

// CPO for getting specific operand templates
template<MMA_Tag mma, Op_Tag optag, Order order, bool is_pack>
struct GetOperand: std::false_type {};

}  // namespace turbomind::gemm