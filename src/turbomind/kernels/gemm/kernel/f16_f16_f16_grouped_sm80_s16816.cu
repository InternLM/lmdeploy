// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::f16_f16_f16_grouped_sm80_s16816()
{

    if constexpr (1) {
        using C = Sm80_s16816<Sm80,
                              Operand_A<half, kRowMajor>,          // A
                              Transform_Default,                   // tarnsform A
                              VoidOperand,                         // U
                              Operand_B_Pack<half, kRowMajor, 1>,  // B
                              Transform_Default,                   // transform B
                              VoidOperand,                         // V
                              kRowMajor,                           // order_C
                              half,                                // Tc
                              Striding::kIndexed,                  // indexed input
                              Striding::kBlocked,
                              Striding::kBlocked,
                              pair<false, false>,
                              pair<Striding::kFlat, Striding::kFlat>,
                              DynamicScheduler<kColMajor>>;

        // clang-format off
        Add<C::Type<256, 128,  64, 4, 2, 1, D, D, 3, false, 1, 1>>();
        Add<C::Type<128, 256,  64, 2, 4, 1, D, D, 3, false, 1, 1>>(); // 10
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3, false, 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 6
        Add<C::Type<128, 128,  64, 2, 2, 1, D, D, 3, true, 1, 1>>(); 
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 5, true, 1, 1>>(); 
        Add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 2
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 3, true, 1, 1>>(); // *
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 5, true, 1, 1>>();
        Add<C::Type< 64,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 4
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 10
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        // clang-format on
    }

    if constexpr (1) {
        using C = Sm80_s16816<Sm80,
                              Operand_A<half, kRowMajor>,             // A
                              Transform_Default,                      // tarnsform A
                              VoidOperand,                            // U
                              Operand_B_Pack<uint4_t, kRowMajor, 2>,  // B
                              Transform_HMMA_16816<1, 0>,             // transform B,
                              Operand_UV_Pack<uint32_t, true>,        // V
                              kRowMajor,                              // order_C
                              half,                                   // Tc
                              Striding::kIndexed,                     // indexed input
                              Striding::kBlocked,
                              Striding::kBlocked,
                              pair<false, false>,
                              pair<Striding::kFlat, Striding::kFlat>,
                              DynamicScheduler<kColMajor>>;

        // clang-format off
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3, false, 1, 128>>();  // 10 + 5 + 4 + 10 + 10, 37
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 128>>();  // 1 + 6 + 4 + 4 + 2, 3
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 7 + 4 + 6 + 2 + 4, 26
        Add<C::Type< 64, 256,  32, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 18
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 3, true, 1, 128>>();  // 2
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 5, true, 1, 128>>();  // 1 + 2 + 2 + 2 + 2, 2
        Add<C::Type< 32, 256,  64, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 9
        Add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 22
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 8
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 1 + 13 + 9 + 13 + 7, 7
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 3, true, 1, 128>>();  // 12 + 2 + 6 + 2 + 8, 42
        // clang-format on
    }
}

}  // namespace turbomind::gemm
