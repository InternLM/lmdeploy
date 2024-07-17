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
        using Config = Sm75_s16816<Operand_A<half, kRowMajor>,
                                   Transform_Default,
                                   VoidOperand,
                                   Operand_B_Pack<uint4_t, kRowMajor>,
                                   Transform_HMMA_16816<1, 0>,
                                   Operand_UV_Pack<uint32_t, true>,
                                   kRowMajor,
                                   half>;

        using namespace cache_policy;

        Add<Config::Type<128, 128, 64, 1, 4, 2, 2, true, 1, 128>>();
        Add<Config::Type<128, 128, 32, 1, 4, 1, 2, true, 1, 128>>();
        Add<Config::Type<64, 128, 32, 1, 4, 1, 2, true, 1, 128>>();
        Add<Config::Type<32, 128, 32, 1, 4, 1, 2, true, 1, 128>>();
        Add<Config::Type<16, 128, 32, 1, 4, 1, 2, true, 1, 128>>();
        Add<Config::Type<16, 128, 64, 1, 4, 2, 2, true, 1, 128>>();
        Add<Config::Type<16, 128, 128, 1, 4, 2, 2, true, 1, 128>>();
    }
}

}  // namespace turbomind::gemm
