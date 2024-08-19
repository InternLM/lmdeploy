// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm90_s16816()
{
    using namespace sm80_s16816;
    using namespace cache_policy;
    //////////////////////////////////////////////////////////////////////////////
    // ! sm_90 + cp.async + evict policy = warp illegal instruction
    //////////////////////////////////////////////////////////////////////////////
    using D = cache_policy::Default;

    using C = Sm80_s16816<Sm90,
                          Operand_A<half, kRowMajor>,          // A
                          Transform_Default,                   // tarnsform A
                          VoidOperand,                         // U
                          Operand_B_Pack<uint4_t, kColMajor>,  // B
                          Transform_HMMA_16816<1, 0>,          // transform B
                          Operand_UV_Pack<uint32_t, true>,     // V
                          kRowMajor,                           // order_C
                          half>;                               // Tc

    // clang-format off
    Add<C::Type<128, 256,  64, 1, 8, 1, D, D, 3, true, 1, 128>>();
    Add<C::Type<128, 256,  32, 1, 8, 1, D, D, 3, true, 1, 128, 128, 128>>();
    Add<C::Type<128, 256,  32, 1, 8, 1, D, D, 4, true, 1, 128, 128, 128>>();
    Add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, 128, 64, 128>>();
    Add<C::Type<128, 128,  32, 1, 4, 1, D, D, 4, true, 1, 128, 64, 128>>();
    Add<C::Type<128, 128,  64, 1, 4, 2, D, D, 3, true, 1, 128, 64, 128>>();

    Add<C::Type<96, 256,  32, 1, 8, 1, D, D, 4, true, 1, 128>>();
    Add<C::Type<96, 256,  32, 1, 8, 1, D, D, 3, true, 1, 128>>();
    Add<C::Type<96, 128,  32, 1, 4, 1, D, D, 4, true, 1, 128>>();
    Add<C::Type<96, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();

    Add<C::Type<64, 256,  32, 1, 4, 1, D, D, 3, true, 1, 128, 64, 128>>();
    Add<C::Type<64, 256,  32, 1, 4, 1, D, D, 4, true, 1, 128, 64, 128>>();
    Add<C::Type<64, 128,  32, 1, 4, 1, D, D, 4, true, 1, 128>>();
    Add<C::Type<64, 128,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
    Add<C::Type<64, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
    Add<C::Type<64,  64,  64, 1, 2, 2, D, D, 6, true, 1, 128>>();

    Add<C::Type<48, 256,  64, 1, 4, 1, D, D, 3, true, 1, 128, 48, 128>>();
    Add<C::Type<48, 128,  64, 1, 4, 1, D, D, 4, true, 1, 128>>();
    Add<C::Type<48, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
    Add<C::Type<48,  64, 128, 1, 2, 2, D, D, 4, true, 1, 128>>();

    Add<C::Type<32, 256,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
    Add<C::Type<32, 128,  64, 1, 4, 1, D, D, 4, true, 1, 128>>();
    Add<C::Type<32, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
    Add<C::Type<32,  64, 128, 1, 2, 2, D, D, 3, true, 1, 128>>();
    Add<C::Type<32,  64, 128, 1, 2, 2, D, D, 4, true, 1, 128>>();

    Add<C::Type<16, 128,  64, 1, 4, 1, D, D, 4, true, 1, 128>>();
    Add<C::Type<16, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
    Add<C::Type<16, 128, 128, 1, 4, 2, D, D, 4, true, 1, 128>>();
    Add<C::Type<16,  64, 128, 1, 2, 2, D, D, 3, true, 1, 128>>();
    Add<C::Type<16,  64, 128, 1, 2, 2, D, D, 4, true, 1, 128>>();
    // clang-format on
}

}  // namespace turbomind::gemm
