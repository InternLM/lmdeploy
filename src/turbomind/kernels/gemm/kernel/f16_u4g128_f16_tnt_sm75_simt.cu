// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_simt.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm75_simt()
{
    using namespace simt;

    using S = cache_policy::Stream;
    using D = cache_policy::Default;

    {  // quant B
        using Operand_A    = typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand;
        using Operand_B_U4 = typename GetOperand<HMMA_SIMT, OPERAND_B, uint4_t, kRowMajor, true>::Operand;
        using Operand_V    = typename GetOperand<HMMA_SIMT, OPERAND_V, uint32_t, kColMajor, true>::Operand;

        using C = Sm75_Simt<Operand_A,
                            Transform_Default,
                            VoidOperand,
                            Operand_B_U4,
                            Transform_HMMA_SIMT_B,
                            Operand_V,
                            kRowMajor,
                            half>;

        // clang-format off
        Add<C::Type<128, 128, 16, 8, 1, 1, D, D, 2, true, 1, 128>>();
        Add<C::Type< 64, 128, 16, 4, 1, 1, D, D, 2, true, 1, 128>>();
        Add<C::Type< 64, 128, 16, 4, 1, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 128, 32, 4, 1, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 64, 2, 2, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  4, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  1, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
