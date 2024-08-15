// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm75_s16816()
{
    using namespace sm75_s16816;

    {  // fp x u4
        using C = Sm75_s16816<Operand_A<half, kRowMajor>,
                              Transform_Default,
                              VoidOperand,
                              Operand_B_Pack<uint4_t, kColMajor>,
                              Transform_HMMA_16816<1, 0>,
                              Operand_UV_Pack<uint32_t, true>,
                              kRowMajor,
                              half>;

        using S = cache_policy::Stream;
        using D = cache_policy::Default;

        // clang-format off
        Add<C::Type<128, 256, 32, 1, 8, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128, 32, 1, 4, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 96,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 64, 128, 32, 1, 4, 1, D, D, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 48, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 48,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
