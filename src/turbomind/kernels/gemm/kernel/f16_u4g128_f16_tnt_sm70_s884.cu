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

        // m8n32k8: pack_bv=1, fuse-prefetch=false, (128,48.325), (192,8.44), (256,-10.055), (4096,-19.299)
        // Add<Config::Type<128, 256, 16, 2, 4, 1, Default, Default, 2, true, 1, 128, 128, 128>>();  // 30.823
        // Add<Config::Type<128, 128, 16, 2, 2, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 16.324
        // Add<Config::Type<64, 256, 16, 1, 4, 1, Default, Default, 2, true, 1, 128, 64, 128>>(); 
        // Add<Config::Type<96, 128, 16, 2, 2, 1, Default, Stream, 2, true, 1, 128>>();  // 50.745
        // Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 64.223
        // Add<Config::Type<64, 256, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 75.178
        // Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 77.192
        // Add<Config::Type<32, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128, 32, 128>>();  // 61.234
        // Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 139.073
        // Add<Config::Type<16, 256, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 120.678
        // Add<Config::Type<8, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 184.574
        // Add<Config::Type<8, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 187.233

        // m16n16k8: pack_bv=2, fuse-prefetch=true
        // (8,167), (16,161.104), (32,96.182), (64,91.557), (96,79.896), (128,48.406), (192,9.907), (256,-9.57), (4096,-19.72)
        Add<Config::Type<128, 256, 16, 1, 8, 1, Default, Default, 2, true, 1, 128, 128, 128>>();  // 38.674
        Add<Config::Type<128, 256, 16, 2, 4, 1, Default, Default, 2, true, 1, 128, 128, 128>>();  // 39.991
        Add<Config::Type<128, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 23.998
        Add<Config::Type<96, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>(); // 78.087
        Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Default, 2, true, 1, 128>>(); 
        Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 82.743
        Add<Config::Type<64, 256, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>(); // 82.48
        Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 94.295
        Add<Config::Type<32, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128, 32, 128>>();  // 64.663
        Add<Config::Type<16, 128, 64, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 145.884
        Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>(); 
    }
}

}  // namespace turbomind::gemm
