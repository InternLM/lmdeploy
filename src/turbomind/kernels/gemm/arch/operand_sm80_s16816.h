// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/arch/smem_copy_sm80.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <type_traits>

namespace turbomind::gemm {

namespace sm80_s16816 {

namespace detail {

struct GetSmemLayout {
    template<int S, int C>
    static constexpr auto apply(pair<S, C>)
    {
        // constexpr int S0 = S >= 16 ? 16 : 8;
        constexpr int S0 = 8;
        constexpr int C0 = C >= 64 ? 64 : (C >= 32 ? 32 : 16);
        using _Small     = std::conditional_t<C0 == 32, Swizzle<2, 3, 3>, Swizzle<1, 3, 3>>;
        using Swizzle    = std::conditional_t<C0 == 64, Swizzle<3, 3, 3>, _Small>;
        return SmemLayoutV2<S, C, S0, C0, Swizzle>{};
    }
};

}  // namespace detail

template<Order order>
struct GetSmemLayoutV2 {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        constexpr int2 cs = mk2cs<order>(M, K);
        return detail::GetSmemLayout::apply(pair<cs.y, cs.x>{});
    }
};

// (m, k)
template<class T, Order order>
struct Operand_A {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = order;

    // using SmemCopyAtom =
    //     std::conditional_t<order == kRowMajor, SmemCopy_MMA_16816_A<T, false>, SmemCopy_MMA_16816_B<T, true>>;

    // using SmemCopyAtom = std::conditional_t<order == kRowMajor,
    //                                         LDSM_SM75_8x8<T, 16, 16, kColMajor, kRowMajor>,
    //                                         LDSM_SM75_8x8<T, 16, 16, kRowMajor, kColMajor>>;

    using SmemCopyAtom = LDSM_SM75_8x8<T, 16, 16, ~order, order>;

    using GetSmemLayout = GetSmemLayoutV2<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

// (n, k)
template<class T, Order order, int N>
struct Operand_B {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = order;

    // using SmemCopyAtom =
    //     std::conditional_t<order == kRowMajor, SmemCopy_MMA_16816_B<T, false>, SmemCopy_MMA_16816_A<T, true>>;
    // using SmemCopyAtom = std::conditional_t<order == kRowMajor,  //
    //                                         LDSM_SM75_8x8<T, 16, 16, kRowMajor, kRowMajor>,
    //                                         LDSM_SM75_8x8<T, 16, 16, kColMajor, kColMajor>>;

    using SmemCopyAtom = LDSM_SM75_8x8<T, N, 16, order, order>;

    using GetSmemLayout = GetSmemLayoutV2<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<Order order>
struct _GetSmemLayoutC {
    template<int M, int N>
    static constexpr auto apply(pair<M, N>)
    {
        if constexpr (order == kRowMajor) {
            // x01  23
            // cccccss
            //                                    bits base shift
            return SmemLayoutV2<M, N, 8, 32, Swizzle<2, 3, 2>>{};
        }
        else {
            // 234  x01
            // 23401x
            // cccccsss
            // so that x is not part of swizzling
            return SmemLayoutV2<N, M, 8, 32, Swizzle<2, 3, 3>>{};
        }
    }
};

template<Order order>
struct _GetThreadMapC {
    template<int M, int N, int THREADS>
    static constexpr auto apply(pair<M, N>, constant<THREADS>)
    {
        constexpr auto cs    = mk2cs<order>(M, N);
        constexpr int  WARPS = THREADS / WARP_SIZE;

        return ThreadMap_V2<cs.x, cs.y, 4, Raked, WARPS>{};
    }
};

template<class T, Order order>
struct Operand_C {
    using Dtype = T;

    static constexpr Order kOrder = order;

    using GetSmemLayout = _GetSmemLayoutC<order>;
    using GetThreadMap  = _GetThreadMapC<order>;
};

template<class T>
struct Operand_UV {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_U<T>;

    struct GetSmemLayout {
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };
    using GetGmemIter = GetGmemIter;
};

template<Order order>
struct GetSmemLayout_Pack {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        constexpr int2 CS = mk2cs<order>(M, K);
        return SmemLayoutV2<CS.y, CS.x, 1, 1>{};
    }
};

template<class T, Order order>
struct Operand_A_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 2;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_A | Pack_M;
    static constexpr Order kOrder = order;

    // using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, kOrder, 16 * Pack_M, 16, 8, Pack_M>;
    using _SCp         = typename Operand_A<T, order>::SmemCopyAtom;
    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, _SCp, order, Pack_M>;

    using GetSmemLayout = GetSmemLayout_Pack<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T, Order order>
struct Operand_B_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 2;

    static constexpr Pack  kPack  = HMMA_16816 | OPERAND_B | Pack_M;
    static constexpr Order kOrder = order;

    using SmemCopyAtom = SmemCopyAtom_Pack_v2<T, kOrder, 16 * Pack_M, 16, 8, Pack_M>;

    using GetSmemLayout = GetSmemLayout_Pack<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T, bool is_V>
struct Operand_UV_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_16816 | (is_V ? OPERAND_V : OPERAND_U) | Pack_M;
    static constexpr Order kOrder = Order::kColMajor;

    using _SCp         = typename Operand_UV<T>::SmemCopyAtom;
    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, _SCp, kOrder, Pack_M>;

    using GetSmemLayout = GetSmemLayout_Pack<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

}  // namespace sm80_s16816

template<class T, Order order>
struct GetOperand<HMMA_16816, OPERAND_A, T, order, false>: std::true_type {
    using Operand = sm80_s16816::Operand_A<T, order>;
};

template<class T, Order order>
struct GetOperand<HMMA_16816, OPERAND_B, T, order, false>: std::true_type {
    using Operand = sm80_s16816::Operand_B<T, order, 16>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_U, T, kColMajor, false>: std::true_type {
    using Operand = sm80_s16816::Operand_UV<T>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_V, T, kColMajor, false>: std::true_type {
    using Operand = sm80_s16816::Operand_UV<T>;
};

// template<class T>
// struct GetOperand<HMMA_16816, OPERAND_A, T, kColMajor, true>: std::true_type {
//     using Operand = sm80_s16816::Operand_A_Pack<T, kColMajor>;
// };

// template<class T>
// struct GetOperand<HMMA_16816, OPERAND_B, T, kColMajor, true>: std::true_type {
//     using Operand = sm80_s16816::Operand_B_Pack<T, kColMajor>;
// };

// template<>
// struct GetOperand<HMMA_16816, OPERAND_U, uint32_t, kColMajor, true>: std::true_type {
//     using Operand = sm80_s16816::Operand_U_Pack<uint32_t>;
// };

// template<>
// struct GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>: std::true_type {
//     using Operand = sm80_s16816::Operand_U_Pack<uint32_t>;
// };

}  // namespace turbomind::gemm
