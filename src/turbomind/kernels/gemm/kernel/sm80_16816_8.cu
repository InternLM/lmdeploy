// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
using namespace config::geometry;

constexpr auto e4m3_packer =
    pack_e4m3<Arch<80>, kColMajor, HMMA_16816 | OPERAND_A | 1, kColMajor, HMMA_16816 | OPERAND_U | 1, kBfloat16>;

const Family e4m3{15, 200, kBfloat16, kBfloat16, 128, 8, 1, 1, true, true, supports_e4m3<kBfloat16, 1>, e4m3_packer};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int   Stages,
                  Order Raster,
                  class PolicyA,
                  class PolicyB,
                  bool SplitK,
                  int  EpiM         = -1,
                  int  EpiN         = -1,
                  bool FusePrefetch = true,
                  int  GroupAxis    = 1,
                  int  OperandN     = 16>
         class K>
void register_kernels(Collector& c)
{
    {
        // add<K<_256x128x32_8x1x1, 3, kColMajor, D, D, true, 128, 128, true, -1>>(c);

        add<K<_256x128x32_8x1x1, 3, kColMajor, D, D, true, 128, 128>>(c);
        add<K<_256x64x32_4x1x1, 3, kColMajor, D, D, true, 128, 64>>(c);
        add<K<_256x32x64_4x1x1, 3, kColMajor, D, D, true>>(c);
        add<K<_128x128x32_4x1x1, 3, kColMajor, D, D, true, 128, 64>>(c);
        add<K<_128x96x32_4x1x1, 3, kColMajor, D, D, true>>(c);
        add<K<_128x64x32_4x1x1, 3, kColMajor, D, D, true>>(c);
        add<K<_128x32x32_4x1x1, 3, kColMajor, S, D, true>>(c);
        add<K<_128x16x64_4x1x1, 3, kColMajor, S, D, true>>(c);
        add<K<_128x16x32_4x1x1, 5, kColMajor, S, D, true>>(c);

        add<K<_256x8x64_4x1x1, 3, kColMajor, S, D, true, -1, -1, true, 1, 8>>(c);
        add<K<_128x8x64_4x1x1, 3, kColMajor, S, D, true, -1, -1, true, 1, 8>>(c);
        add<K<_64x8x128_4x1x1, 3, kColMajor, S, D, true, -1, -1, true, 1, 8>>(c);
    }
}

using E4M3 = Config_E4M3<Sm80, bfloat16_t>;

Registrar reg(e4m3, register_kernels<E4M3::Type>);
}  // namespace

}  // namespace turbomind::gemm
