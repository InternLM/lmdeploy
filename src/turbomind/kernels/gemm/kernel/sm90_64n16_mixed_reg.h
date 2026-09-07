// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/gemm_universal_sm90_mixed.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_mixed.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"

namespace turbomind::gemm::detail {

template<class Format>
struct C {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, class ClusterShape = config::Shape<1, 1>, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0>
    using Type = KernelImplSm90Mixed<GemmUniversalSm90Mixed<Format, Config_, Stages, Raster, Mode, Silu, ClusterShape, MmaN, SeparateMmaAtoms, EpiM, EpiStages>>;
};

}  // namespace turbomind::gemm::detail
