// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::mxfp4_bf16_bf16_nnn_sm90_s16816()
{
    using namespace sm80_s16816;
    using namespace cache_policy;

    using namespace sm80_s16816;
    using namespace cache_policy;
    //////////////////////////////////////////////////////////////////////////////
    // ! sm_90 + cp.async + evict policy = warp illegal instruction
    //////////////////////////////////////////////////////////////////////////////
    using D = cache_policy::Default;

    using C = Sm80_s16816<Sm90,
                          bfloat16_t,
                          Operand_A_Pack<fp4_e2m1_t, kColMajor, 1>,  // A
                          Transform_HMMA_16816<0, 1>,                // tarnsform A
                          Operand_UV_Pack<uint8_t, false>,           // U
                          Operand_B<bfloat16_t, kRowMajor>,          // B
                          Transform_Default,                         // transform B
                          VoidOperand,                               // V
                          kColMajor,                                 // order_C
                          bfloat16_t,                                // Tc
                          Striding::kFlat,
                          Striding::kFlat,
                          Striding::kFlat,
                          GemmScheduler<kColMajor>>;

    Add<C::Type<128, 256, 64, 2, 4, 1, D, D, 3, false, 32, 1, 128, 128>>();
    // Add<C::Type<128, 128, 64, 4, 1, 1, D, D, 3, false, 32, 1>>();
    // Add<C::Type<128, 16, 32, 4, 1, 1, D, D, 3, false, 32, 1>>();
    // Add<C::Type<32, 16, 32, 1, 1, 1, D, D, 3, true, 32, 1>>();
}

}  // namespace turbomind::gemm