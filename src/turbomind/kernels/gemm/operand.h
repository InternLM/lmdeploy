// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct VoidOperand {
    using Dtype                   = int;
    static constexpr Order kOrder = Order::kColMajor;
    static constexpr Pack  kPack  = Pack::kNone;
    using SmemLayout              = SmemLayoutV2<1, 1>;
    using GmemIter                = VoidIterator;
    using SmemCopy                = VoidSmemCopy<1, 1>;
};

}  // namespace turbomind::gemm