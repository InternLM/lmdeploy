// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

struct VoidOperand {
    using Dtype = int;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kColMajor;

    struct GetSmemLayout {
        static constexpr SmemLayoutV2<1, 1> apply(...)
        {
            return {};
        }
    };

    using SmemCopyAtom = VoidSmemCopyAtom;

    struct GetGmemIter {
        static constexpr auto apply(...)
        {
            return type_c<VoidGmemIter>;
        }
    };
};

/// TODO: fix AlignC, AlignS
/// TODO: fix GroupSize
template<class Operand, class Iterator, int M, int K, int WARPS, int GroupSize = 1>
struct MakeOperand {

    using Dtype = typename Operand::Dtype;

    static constexpr Pack  kPack      = Operand::kPack;
    static constexpr Order kOrder     = Operand::kOrder;
    static constexpr int   kGroupSize = GroupSize;

    static constexpr int2 kPackMK = Packing_v2<kPack, kOrder>::apply({M, ceil_div(K, kGroupSize)});

    static constexpr pair<kPackMK.x, kPackMK.y> kShapeMK{};

    using SmemLayout   = decltype(Operand::GetSmemLayout::apply(kShapeMK));
    using SmemAccessor = SmemAccessorV2<Dtype, SmemLayout, kOrder>;

    using GmemIter = typename decltype(Operand::GetGmemIter::apply(
        type_c<Operand>, type_c<Iterator>, type_c<SmemLayout>, kShapeMK, constant<WARPS>{}))::type;

    using SmemCopyAtom = typename Operand::SmemCopyAtom;
};

// CPO for getting specific operand templates
template<MMA_Tag mma, Op_Tag optag, class T, Order order, bool is_pack, class SFINAE = void>
struct GetOperand: std::false_type {
};

}  // namespace turbomind::gemm
