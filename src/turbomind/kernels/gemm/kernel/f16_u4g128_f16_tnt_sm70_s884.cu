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

        // m8n32k8: pack_bv=1
        // (8,226.234),(16,192.248),(32,120.564),(64,103.483),(96,98.209),(128,54.537),(192,13.739)
        // (256,-6.61),(4096,-16.622),(8192,-16.021)
        Add<Config::Type<128, 256, 16, 2, 4, 1, Default, Default, 2, true, 1, 128, 128, 128>>();  // 50.631
        Add<Config::Type<128, 128, 16, 2, 2, 1, Default, Default, 2, true, 1, 128, 64, 128>>();
        Add<Config::Type<128, 128, 16, 2, 2, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 50.698
        Add<Config::Type<96, 128, 32, 2, 2, 1, Default, Stream, 2, true, 1, 128, 48, 128>>();   // 93.395
        Add<Config::Type<64, 128, 32, 2, 2, 1, Default, Default, 2, true, 1, 128, 32, 128>>();
        Add<Config::Type<64, 128, 32, 2, 2, 1, Default, Stream, 2, true, 1, 128, 32, 128>>();  // 93.482
        Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 32, 128>>();  // 82.113
        Add<Config::Type<64, 256, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 80.686
        Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();           // 92.014
        Add<Config::Type<32, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128, 32, 128>>();  // 110.979
        Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();           // 147.616
        Add<Config::Type<16, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();           // 186.569
        Add<Config::Type<8, 128, 64, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();            // 218.194
        Add<Config::Type<8, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();            // 209.224
        Add<Config::Type<8, 256, 64, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();            // 219.651

        // m16n16k8: pack_bv=2
        // (8,179.471),(16,174.246),(32,114.659),(64,100.813),(96,96.822),(128,53.423),(192,12.433),(256,-7.601),(4096,-17.335)
        // Add<Config::Type<128, 256, 16, 1, 8, 1, Default, Default, 2, true, 1, 128, 128, 128>>(); // 50.934
        // Add<Config::Type<128, 128, 16, 1, 4, 1, Default, Default, 2, true, 1, 128, 64, 128>>();  // 47.874
        // Add<Config::Type<128, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>();  // 47.874
        // Add<Config::Type<96, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>(); // 95.303
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Default, 2, true, 1, 128>>();
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 97.095
        // Add<Config::Type<64, 128, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 86.559
        // Add<Config::Type<64, 256, 16, 1, 4, 1, Default, Stream, 2, true, 1, 128, 64, 128>>(); // 73.869
        // Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 115.205
        // Add<Config::Type<32, 256, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128, 32, 128>>();  // 96.151
        // Add<Config::Type<16, 128, 64, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();  // 175.285
        // Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 2, true, 1, 128>>();
    }
}

}  // namespace turbomind::gemm
