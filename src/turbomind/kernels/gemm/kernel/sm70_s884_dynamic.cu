// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm70_s884_dynamic()
{
    if constexpr (1) {
        using C = Sm70_s884<Operand_A<half>,       // A
                            Transform_Default,     // tarnsform A
                            VoidOperand,           // U
                            Operand_B_Pack<half>,  // B
                            Transform_Default,     // transform B
                            VoidOperand,           // V
                            kRowMajor,             // order_C
                            half,                  // Tc
                            Striding::kIndexed,    // indexed input
                            Striding::kBlocked,
                            Striding::kBlocked,
                            kColMajor,
                            0>;

        // clang-format off
        Add<C::Type<256, 128,  16, 4, 2, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 1,  64, 128>>();
        Add<C::Type< 96,  64,  32, 2, 2, 1, D, D, 2, true, 1, 1>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        // clang-format on
    }

    if constexpr (1) {
        using C = Sm70_s884<Operand_A<half>,           // A
                            Transform_Default,         // tarnsform A
                            VoidOperand,               // U
                            Operand_B_Pack<uint4_t>,   // B
                            Transform_HMMA_SIMT_B,     // transform B,
                            Operand_V_Pack<uint32_t>,  // V
                            kRowMajor,                 // order_C
                            half,                      // Tc
                            Striding::kIndexed,        // indexed input
                            Striding::kBlocked,
                            Striding::kBlocked,
                            kColMajor,
                            0>;

        // clang-format off
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        using C = Sm70_s884<Operand_A<half>,             // A
                            Transform_Default,           // tarnsform A
                            VoidOperand,                 // U
                            Operand_B_Pack<fp4_e2m1_t>,  // B
                            Transform_HMMA_SIMT_B,       // transform B,
                            Operand_V_Pack<uint8_t>,     // V
                            kRowMajor,                   // order_C
                            half,                        // Tc
                            Striding::kIndexed,          // indexed input
                            Striding::kBlocked,
                            Striding::kBlocked,
                            kColMajor,
                            0>;

        // clang-format off
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        // clang-format on
    }

    if constexpr (1) {
        using C = Sm70_s884<Operand_A<half>,             // A
                            Transform_Default,           // tarnsform A
                            VoidOperand,                 // U
                            Operand_B_Pack<fp8_e4m3_t>,  // B
                            Transform_HMMA_SIMT_B,       // transform B,
                            Operand_V_Pack<uint16_t>,    // V
                            kRowMajor,                   // order_C
                            half,                        // Tc
                            Striding::kIndexed,          // indexed input
                            Striding::kBlocked,
                            Striding::kBlocked,
                            kColMajor,
                            0>;

        // clang-format off
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
