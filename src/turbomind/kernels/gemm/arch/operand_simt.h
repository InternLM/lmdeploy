// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/arch/smem_copy_simt.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/simt.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace simt {

struct GetSmemLayout {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        return SmemLayoutV2<M, K>{};
    }
};

template<class T, int K>
struct Operand_A {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_SIMT_A<T, K>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
};

template<class T, int K>
struct Operand_B {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_SIMT_B<T, K>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
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
struct Operand_V {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopy_MMA_SIMT_V<T, 1>;

    struct GetSmemLayout {  // m-major
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };

    using GetGmemIter = GetGmemIter;
};

struct GetSmemLayout_Pack {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        return SmemLayoutV2<M, K>{};
    }
};

template<class T, int K>
struct Operand_B_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_SIMT | OPERAND_B | Pack_M;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom  = SmemCopyAtom_Pack_v3<T, typename Operand_B<T, K>::SmemCopyAtom, kRowMajor, Pack_M>;
    using GetSmemLayout = GetSmemLayout_Pack;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_V_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_SIMT | OPERAND_V | Pack_M;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, SmemCopy_MMA_SIMT_V<T, OP_K>, kColMajor, Pack_M>;

    struct GetSmemLayout {  // m-major
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };

    using GetGmemIter = GetGmemIter;
};

}  // namespace simt

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_A, T, kRowMajor, false>: std::true_type {
    using Operand = simt::Operand_A<T, simt::OP_K>;
};

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_B, T, kRowMajor, false>: std::true_type {
    using Operand = simt::Operand_B<T, simt::OP_K>;
};

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_V, T, kColMajor, false>: std::true_type {
    using Operand = simt::Operand_V<T>;
};

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_B, T, kRowMajor, true>: std::true_type {
    using Operand = simt::Operand_B_Pack<T, simt::OP_K>;
};

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_V, T, kColMajor, true>: std::true_type {
    using Operand = simt::Operand_V_Pack<T>;
};

}  // namespace turbomind::gemm
