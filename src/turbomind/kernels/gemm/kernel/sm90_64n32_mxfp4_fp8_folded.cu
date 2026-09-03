// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/gemm_universal_sm90_mxfp4_fp8_folded.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_mxfp4_fp8.h"
#include "src/turbomind/kernels/gemm/registrar.h"

namespace turbomind::gemm {
namespace {

template<Order    Raster,
         int      MulticastA,
         int      MulticastB,
         bool     Grouped,
         Striding StridingA,
         int      TileM,
         int      TileN,
         int      Stages,
         class    WGLayout,
         int      MmaN,
         int      ProducerRegsTma,
         int      MathRegsTma,
         int      ProducerRegsIndexed,
         int      MathRegsIndexed,
         int      EpilogueTileM,
         int      EpilogueTileN,
         int      EpilogueStages,
         bool     SupportsFusedSilu>
void add(Collector& c)
{
    c.add(std::make_unique<KernelImplSm90MxFp4Fp8<
        GemmUniversalSm90MxFp4Fp8Folded<Raster,
                                         MulticastA,
                                         MulticastB,
                                         Grouped,
                                         StridingA,
                                         TileM,
                                         TileN,
                                         Stages,
                                         WGLayout,
                                         MmaN,
                                         ProducerRegsTma,
                                         MathRegsTma,
                                         ProducerRegsIndexed,
                                         MathRegsIndexed,
                                         EpilogueTileM,
                                         EpilogueTileN,
                                         EpilogueStages,
                                         SupportsFusedSilu>>>());
}

Registrar reg([](Collector& c, int) {
    add<kRowMajor, 1, 1, false, Striding::kFlat, 8, 128, 4, WG_1x2, 8, 40, 232, 72, 216, 8, 128, 1, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 16, 128, 4, WG_1x2, 16, 40, 232, 72, 216, 16, 128, 1, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 32, 128, 4, WG_1x2, 32, 40, 232, 72, 216, 32, 128, 1, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 64, 128, 4, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 96, 128, 4, WG_1x2, 96, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 128, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 192, 128, 4, WG_1x2, 96, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 256, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 2, 1, false, Striding::kFlat, 256, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 2, false, Striding::kFlat, 256, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 8, 256, 3, WG_1x2, 8, 40, 232, 72, 216, 8, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 16, 256, 3, WG_1x2, 16, 40, 232, 72, 216, 16, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 32, 256, 3, WG_1x2, 32, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 64, 256, 3, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 96, 256, 3, WG_1x2, 96, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, true, Striding::kBlocked, 128, 128, 3, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, true, Striding::kIndexed, 64, 128, 3, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 64, 256, 2, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, true>(c);
    add<kRowMajor, 1, 1, true, Striding::kIndexed, 64, 256, 2, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, true>(c);
});

}  // namespace
}  // namespace turbomind::gemm

#endif
