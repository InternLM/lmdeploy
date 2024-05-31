// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/registry.h"

namespace turbomind::gemm {

namespace sm80_hmma_16816_pack {

struct Operand_A_N {
    template<class T, int CTA_M, int CTA_K, int WARP_M, int WARP_CNT, bool Align_M>
    struct type {
        using Dtype = T;

        static constexpr Pack  kPack  = HMMA_16816 | OPERAND_A | 1;
        static constexpr Order kOrder = Order::kColMajor;

        static constexpr int2 _PACK_CS = Packing<kPack>::apply(mk2cs<kOrder>(CTA_M, CTA_K));
        static constexpr int  _PACK_C  = _PACK_CS.x;
        static constexpr int  _PACK_S  = _PACK_CS.y;
        //                                    S     C
        using SmemCopy   = SmemCopy_Packed<T, 16, WARP_M, 1, 1>;
        using SmemLayout = SmemLayoutV2<_PACK_S, _PACK_C>;

        using _ThreadMap = ThreadMap<_PACK_C, _PACK_S, 8, WARP_CNT>;
        using GmemIter   = GmemIteratorSm80<T, _ThreadMap, SmemLayout, kPack, kOrder, Align_M, true>;
    };
};

struct Operand_B_T {
    template<class T, int CTA_N, int CTA_K, int WARP_N, int WARP_CNT, bool Align_N>
    struct type {
        using Dtype = T;

        static constexpr Pack  kPack  = HMMA_16816 | OPERAND_B | 1;
        static constexpr Order kOrder = Order::kRowMajor;

        static constexpr int2 _PACK_CS = Packing<kPack>::apply(mk2cs<kOrder>(CTA_N, CTA_K));
        static constexpr int  _PACK_C  = _PACK_CS.x;
        static constexpr int  _PACK_S  = _PACK_CS.y;

        //                                      S     C
        using SmemCopy   = SmemCopy_Packed<T, WARP_N, 16, 1, 1>;
        using SmemLayout = SmemLayoutV2<_PACK_S, _PACK_C>;

        using _ThreadMap = gemm::ThreadMap<_PACK_C, _PACK_S, 8, WARP_CNT>;
        using GmemIter   = GmemIteratorSm80<T, _ThreadMap, SmemLayout, kPack, kOrder, true, Align_N>;
    };
};

}  // namespace sm80_hmma_16816_pack

template<>
struct GetOperand<HMMA_16816, OPERAND_A, Order::kColMajor, true>: std::true_type {
    using Operand = sm80_hmma_16816_pack::Operand_A_N;
};

template<>
struct GetOperand<HMMA_16816, OPERAND_B, Order::kRowMajor, true>: std::true_type {
    using Operand = sm80_hmma_16816_pack::Operand_B_T;
};

}  // namespace turbomind::gemm