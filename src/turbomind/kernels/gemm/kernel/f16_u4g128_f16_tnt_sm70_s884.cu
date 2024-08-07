// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm70_s884()
{
    using namespace sm70_s884;
    {  // quant B
        using Config = Sm70_s884<typename GetOperand<HMMA_884, OPERAND_A, half, kRowMajor, false>::Operand,
                                 Transform_Default,
                                 VoidOperand,
                                 typename GetOperand<HMMA_884, OPERAND_B, uint4_t, kRowMajor, true>::Operand,
                                 Transform_HMMA_SIMT_B,
                                 typename GetOperand<HMMA_884, OPERAND_V, uint32_t, kColMajor, true>::Operand,
                                 kRowMajor,
                                 half>;

        using namespace cache_policy;

        // Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 2, true, 1, 128, 128, 128>>(); // 23.191
        Add<Config::Type<128, 128, 16, 1, 4, 1, Default, Default, 2, true, 1, 128, 64, 128>>();  // 26.293
        // Add<Config::Type<128, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 27.252
        Add<Config::Type<96, 128, 16, 1, 4, 1, Default, Default, 2, true, 1, 128>>(); // 64.69
        Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Default, 2, true, 1, 128>>(); // 78.862
        // Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 80.243
        Add<Config::Type<64, 256, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>(); // 86.126
        Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 73.578
        Add<Config::Type<32, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 92.494
        Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 146.158
    }
}

}  // namespace turbomind::gemm
