// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/gemm_universal_sm90_mxfp4_fp8_unfolded.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_64n32_mxfp4_fp8_config.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_mxfp4_fp8.h"
#include "src/turbomind/kernels/gemm/registrar.h"

namespace turbomind::gemm {
namespace {

template<class Tile>
void add(Collector& c)
{
    using Gemm = GemmUniversalSm90MxFp4Fp8Unfolded<kRowMajor, Tile>;
    c.add(std::make_unique<KernelImplSm90MxFp4Fp8<Gemm>>());
}

Registrar reg([](Collector& c, int) {
    add<MxFp4Fp8Tile_64x128>(c);
});

}  // namespace
}  // namespace turbomind::gemm

#endif
