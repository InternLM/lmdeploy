// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm75_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm75_s16816_dynamic()
{
    if constexpr (1) {
        using C = Sm75_s16816<Operand_A<half, kRowMajor>,          // A
                              Transform_Default,                   // tarnsform A
                              VoidOperand,                         // U
                              Operand_B_Pack<half, kRowMajor, 1>,  // B
                              Transform_Default,                   // transform B
                              VoidOperand,                         // V
                              kRowMajor,                           // order_C
                              half,                                // Tc
                              Striding::kIndexed,
                              Striding::kBlocked,
                              Striding::kBlocked,
                              kColMajor,
                              0>;

        // clang-format off
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 2,    0, 1, 1, 128, 128>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 2, true, 1, 1,  64, 128>>();
        Add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 2, true, 1, 1>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 64,  64, 128, 1, 2, 2, D, S, 2, true, 1, 1>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 2, true, 1, 1>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 2, true, 1, 1>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        // clang-format on
    }

    if constexpr (1) {
        using C = Sm75_s16816<Operand_A<half, kRowMajor>,             // A
                              Transform_Default,                      // tarnsform A
                              VoidOperand,                            // U
                              Operand_B_Pack<uint4_t, kRowMajor, 2>,  // B
                              Transform_HMMA_16816<1, 0>,             // transform B,
                              Operand_UV_Pack<uint32_t, true>,        // V
                              kRowMajor,                              // order_C
                              half,                                   // Tc
                              Striding::kIndexed,
                              Striding::kBlocked,
                              Striding::kBlocked,
                              kColMajor,
                              0>;

        // clang-format off
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 2,    0, 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 256>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16,  64,  64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        using C = Sm75_s16816<Operand_A_Pack<fp4_e2m1_t, kColMajor, 1>,  // A
                              Transform_HMMA_16816<0, 1>,                // tarnsform A
                              Operand_UV_Pack<uint8_t, false>,           // U
                              Operand_B<half_t, kRowMajor>,              // B
                              Transform_Default,                         // transform B
                              VoidOperand,                               // V
                              kColMajor,                                 // order_C
                              half_t,                                    // Tc
                              Striding::kBlocked,
                              Striding::kIndexed,  // indexed input
                              Striding::kBlocked,
                              kRowMajor,
                              1>;

        Add<C::Type<128, 128, 32, 4, 1, 1, D, D, 2, false, 32, 1, 128, 64>>();
        Add<C::Type<128, 64, 32, 4, 1, 1, D, D, 2, false, 32, 1>>();
        Add<C::Type<128, 32, 32, 4, 1, 1, D, D, 2, false, 32, 1>>();
        Add<C::Type<128, 16, 32, 4, 1, 1, D, D, 2, false, 32, 1>>();
        Add<C::Type<128, 16, 64, 4, 1, 1, D, D, 2, false, 32, 1>>();
        Add<C::Type<64, 16, 64, 4, 1, 1, D, D, 2, false, 32, 1>>();
    }
}

}  // namespace turbomind::gemm
