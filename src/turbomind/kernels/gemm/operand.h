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

    using SmemCopy = VoidSmemCopy;

    struct GetGmemIter {
        static constexpr auto apply(...)
        {
            return type_c<VoidGmemIter>;
        }
    };
};

/// TODO: fix AlignC, AlignS
/// TODO: fix GroupSize
template<class Operand, class Iterator, int M, int K, int WARP_M, int WARP_K, int WARPS>
struct MakeOperand {

    static constexpr Pack  kPack      = Operand::kPack;
    static constexpr Order kOrder     = Operand::kOrder;
    static constexpr int   kGroupSize = 0;

    static constexpr int2 CS = mk2cs<kOrder>(M, K);
    /// TODO: fix `16`, we dont want WARP_X here in the first place
    static constexpr int2 WARP_CS = mk2cs<kOrder>(WARP_M, 16);

    static constexpr pair<CS.x, CS.y> kShapeCS{};

    using Dtype = typename Operand::Dtype;

    using SmemLayout = decltype(Operand::GetSmemLayout::apply(kShapeCS));

    using GmemIter = typename decltype(Operand::GetGmemIter::apply(
        type_c<Operand>, type_c<Iterator>, type_c<SmemLayout>, kShapeCS, constant<WARPS>{}))::type;

    using SmemCopy = typename Operand::SmemCopy::template Type<WARP_CS.y, WARP_CS.x>;
};

// CPO for getting specific operand templates
template<MMA_Tag mma, Op_Tag optag, class T, Order order, bool is_pack>
struct GetOperand: std::false_type {};

}  // namespace turbomind::gemm