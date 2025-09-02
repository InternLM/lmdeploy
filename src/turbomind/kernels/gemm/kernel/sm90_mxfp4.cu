// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;

namespace {

template<class T, int N>
using Config_ = Sm80_s16816<Sm90,
                            T,
                            Operand_A_Pack<fp4_e2m1_t, kColMajor, 1>,  // A
                            Transform_HMMA_16816<0, 1>,                // tarnsform A
                            Operand_UV_Pack<uint8_t, false>,           // U
                            Operand_B<T, kRowMajor, N>,                // B
                            Transform_Default,                         // transform B
                            VoidOperand,                               // V
                            kColMajor,                                 // order_C
                            T,                                         // Tc
                            Striding::kBlocked,
                            Striding::kIndexed,  // indexed input
                            Striding::kBlocked,
                            kRowMajor,
                            1>;

}

void Registry::sm90_mxfp4()
{
    //////////////////////////////////////////////////////////////////////////////
    // ! sm_90 + cp.async + evict policy = warp illegal instruction
    //////////////////////////////////////////////////////////////////////////////
    using D = cache_policy::Default;

    // Not very useful in the context of grouped GEMM
    // Add<C16::Type<256, 128,  32, 8, 1, 1, D, D, 3, false, 32, 1, 128, 128>>();
    // Add<C16::Type<256, 128,  64, 8, 1, 1, D, D, 3, false, 32, 1, 128, 128>>();

    using C16 = Config_<bfloat16_t, 16>;
    Add<C16::Type<128, 128, 32, 4, 1, 1, D, D, 3, false, 32, 1, 128, 128>>();
    Add<C16::Type<128, 128, 32, 4, 1, 1, D, D, 3, false, 32, 1, 128, 64>>();  // For sm_120
    Add<C16::Type<128, 96, 32, 4, 1, 1, D, D, 3, false, 32, 1>>();
    Add<C16::Type<128, 64, 32, 4, 1, 1, D, D, 3, false, 32, 1>>();
    Add<C16::Type<128, 32, 32, 4, 1, 1, D, D, 3, false, 32, 1>>();
    Add<C16::Type<128, 16, 32, 4, 1, 1, D, D, 5, false, 32, 1>>();
    Add<C16::Type<128, 16, 64, 4, 1, 1, D, D, 3, false, 32, 1>>();

    using C8 = Config_<bfloat16_t, 8>;
    Add<C8::Type<128, 8, 32, 4, 1, 1, D, D, 5, false, 32, 1>>();
    Add<C8::Type<128, 8, 64, 4, 1, 1, D, D, 3, false, 32, 1>>();
    Add<C8::Type<64, 8, 64, 4, 1, 1, D, D, 5, false, 32, 1>>();
}

}  // namespace turbomind::gemm
