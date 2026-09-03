// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/gemm_universal_sm90_mixed.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_mixed.h"
#include "src/turbomind/kernels/gemm/registrar.h"

namespace turbomind::gemm::detail {

template<class Format,
         Order raster,
         Striding striding,
         class Tile,
         bool silu = false,
         int multicast_a = 1,
         int multicast_b = 1>
void add(Collector& c)
{
    constexpr bool grouped = striding != Striding::kFlat;
    using Gemm = GemmUniversalSm90Mixed<raster, multicast_a, multicast_b, grouped, striding, Tile, silu, Format>;
    c.add(std::make_unique<KernelImplSm90Mixed<Gemm>>());
}

}  // namespace turbomind::gemm::detail
