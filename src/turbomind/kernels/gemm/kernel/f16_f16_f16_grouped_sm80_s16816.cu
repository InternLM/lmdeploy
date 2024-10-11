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

// template<int N>
// using C_ = Sm80_s16816<Sm80,
//                        Operand_A<half, kColMajor>,     // A
//                        Transform_Default,              // tarnsform A
//                        VoidOperand,                    // U
//                        Operand_B<half, kRowMajor, N>,  // B
//                        Transform_Default,              // transform B
//                        VoidOperand,                    // V
//                        kColMajor,                      // order_C
//                        half,                           // Tc
//                        Striding::kBlocked,
//                        Striding::kIndexed,
//                        Striding::kBlocked,
//                        DynamicScheduler<kColMajor>>;

void Registry::f16_f16_f16_grouped_sm80_s16816()
{

    // Add<C_<16>::Type<256, 128, 32, 4, 2, 1, D, D, 6, true, 1, 1>>();
    // Add<C_<8>::Type<256, 8, 32, 4, 1, 1, D, D, 3, true, 1, 1>>();
    // Add<C_<8>::Type<128, 8, 32, 4, 1, 1, D, D, 5, true, 1, 1>>();
    // Add<C_<8>::Type<128, 8, 64, 4, 1, 1, D, D, 3, true, 1, 1>>();

    using C = Sm80_s16816<Sm80,
                          Operand_A<half, kRowMajor>,          // A
                          Transform_Default,                   // tarnsform A
                          VoidOperand,                         // U
                          Operand_B_Pack<half, kColMajor, 1>,  // B
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

    Add<C::Type<256, 128, 64, 4, 2, 1, D, D, 3, false, 1, 1>>();
    Add<C::Type<128, 256, 64, 2, 4, 1, D, D, 3, false, 1, 1>>(); // 10
    Add<C::Type<128, 256, 32, 2, 4, 1, D, D, 3, false, 1, 1>>();
    Add<C::Type<128, 128, 32, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 6
    Add<C::Type<128, 128, 64, 2, 2, 1, D, D, 3, true, 1, 1>>(); 
    Add<C::Type<128, 128, 32, 2, 2, 1, D, D, 5, true, 1, 1>>(); 
    // Add<C::Type<128, 128, 32, 2, 2, 1, D, D, 4, true, 1, 1>>();
    // Add<C::Type<128, 128, 32, 2, 2, 1, D, D, 5, true, 1, 1>>();
    // Add<C::Type< 96, 128, 32, 2, 2, 1, D, D, 3, false, 1, 1>>();
    Add<C::Type< 96, 64, 64, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 2
    // Add<C::Type< 96, 128, 32, 2, 2, 1, D, S, 4, true, 1, 1>>();
    // Add<C::Type< 64, 256, 32, 1, 4, 1, D, D, 3, true, 1, 1>>();
    // Add<C::Type< 64, 256, 32, 1, 4, 1, D, S, 3, true, 1, 1>>();
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

}  // namespace turbomind::gemm
