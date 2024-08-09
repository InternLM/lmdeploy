// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/arch/smem_copy_sm70.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm70_s884 {

template<Order order>
struct GetSmemLayout {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        constexpr int2 cs = mk2cs<order>(M, K);
        return SmemLayoutV2<cs.y, cs.x, 1, 1>{};
    }
};

template<class T>
struct Operand_A {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_884_A<T>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_B {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;  // (n,k)

    using SmemCopyAtom = SmemCopy_MMA_884_B<T>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_V {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kColMajor;  // (n,k)

    using SmemCopyAtom = SmemCopy_MMA_884_V<T, 1>;

    struct GetSmemLayout {  // m-major
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };

    using GetGmemIter = GetGmemIter;
};

template<Order order>
struct _GetSmemLayoutC {
    template<int M, int N>
    static constexpr auto apply(pair<M, N>)
    {
        constexpr auto cs = mk2cs<order>(M, N);
        return SmemLayoutV2<cs.y, cs.x, 1, 1>{};
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
struct Operand_B_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_884 | OPERAND_B | Pack_M;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, SmemCopy_MMA_884_B<T>, kOrder, Pack_M>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_V_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_884 | OPERAND_V | Pack_M;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, SmemCopy_MMA_884_V<T, 8>, kColMajor, Pack_M>;

    struct GetSmemLayout {  // m-major
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };

    using GetGmemIter = GetGmemIter;
};

}  // namespace sm70_s884

template<class T>
struct GetOperand<HMMA_884, OPERAND_A, T, kRowMajor, false>: std::true_type {
    using Operand = sm70_s884::Operand_A<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_B, T, kRowMajor, false>: std::true_type {
    using Operand = sm70_s884::Operand_B<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_V, T, kColMajor, false>: std::true_type {
    using Operand = sm70_s884::Operand_V<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_B, T, kRowMajor, true>: std::true_type {
    using Operand = sm70_s884::Operand_B_Pack<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_V, T, kColMajor, true>: std::true_type {
    using Operand = sm70_s884::Operand_V_Pack<T>;
};

}  // namespace turbomind::gemm
