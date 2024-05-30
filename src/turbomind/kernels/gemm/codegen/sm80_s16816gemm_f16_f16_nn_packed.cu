// Copyright (c) OpenMMLab. All rights reserved.

#include "sm80_s16816gemm_f16_f16_nn.h"
#include "src/turbomind/kernels/gemm/registry.h"

namespace turbomind::gemm {

namespace sm80_s16816gemm_f16_f16_nn_packed {

template<class T, int CTA_M, int CTA_K, int WARP_M, int WARP_CNT, bool Align_M>
struct OperandA {
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

template<class T, int CTA_N, int CTA_K, int WARP_N, int WARP_CNT, bool Align_N>
struct OperandB {
    using Dtype = T;

    static constexpr Pack  kPack  =  HMMA_16816 | OPERAND_B | 1;
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

template<class T,
         int  CTA_M,
         int  CTA_N,
         int  CTA_K,
         int  WARP_M,
         int  WARP_N,
         int  WARP_K,
         int  Stages,
         bool SplitK,
         bool AlignedM,
         bool AlignedN>
struct Config {

    using TiledMma = TiledMMA<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, WARP_M, WARP_N, WARP_K>;

    static constexpr int WARP_CNT = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);

    using OperandA = OperandA<T, CTA_M, CTA_K, WARP_M, WARP_CNT, AlignedM>;
    using OperandB = OperandB<T, CTA_N, CTA_K, WARP_N, WARP_CNT, AlignedN>;
    // using OperandB = sm80_s16816gemm_f16_f16_nn::OperandB<T, CTA_N, CTA_K, WARP_N, WARP_CNT, AlignedN>;

    using Void = VoidOperand;

    using Mainloop = MainloopSm80_v2<CTA_M,
                                     CTA_N,
                                     CTA_K,  //
                                     TiledMma,
                                     OperandA,
                                     OperandB,
                                     Void,
                                     Void,
                                     Transform,
                                     Stages>;

    using Kernel = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
};

}  // namespace sm80_s16816gemm_f16_f16_nn_packed

void Registry::reigster_sm80_s16816gemm_f16_f16_nn_packed()
{
    using sm80_s16816gemm_f16_f16_nn_packed::Config;

    Add(std::make_unique<KernelImpl<typename Config<half, 256, 128, 64, 64, 64, 64, 3, false, 0, 0>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<half, 32, 32, 32, 32, 32, 32, 3, false, 0, 0>::Kernel>>());
}

}  // namespace turbomind::gemm