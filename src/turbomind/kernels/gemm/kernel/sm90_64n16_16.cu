// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/cublas.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_bf16.h"
#include "src/turbomind/kernels/gemm/kernel/floating_point.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_bf16.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

namespace {
using config::Shape;
using namespace config::geometry;

void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.weight.dtype(), kBfloat16);
    Tensor trans{{linear.weight.shape(1), linear.weight.shape(0)}, kBfloat16, kDEVICE};
    invokeTransposeAxis01(static_cast<nv_bfloat16*>(trans.raw_data()),
                          static_cast<nv_bfloat16*>(linear.weight.raw_data()),
                          linear.weight.shape(0),
                          linear.weight.shape(1),
                          1,
                          stream);
    linear.weight        = std::move(trans);
    linear.k_desc        = MatrixLayout{kBfloat16,
                                 kColMajor,
                                 linear.input_dim,
                                 linear.output_dim,
                                 (int)linear.weight.stride(0),
                                 0,
                                 0,
                                 nullptr,
                                 nullptr};
    linear.q_desc        = {};
    linear.weight_format = kBfloat16;
}

const Family bf16{27, 250, kBfloat16, kBfloat16, 1, 1, 1, 1, true, true, supports_fp<kBfloat16>, pack, 64, kBfloat16};

struct C {
    template<class Config_,
             int      Stages,
             Order    Raster,
             Striding Mode,
             bool     Silu         = false,
             class ClusterShape    = Shape<1, 1>,
             int  L2HintW          = 0,
             int  MmaN             = 0,
             bool SeparateMmaAtoms = false,
             int  EpiM             = 0,
             int  EpiStages        = 0>
    using Type = KernelImplSm90Bf16<GemmUniversalSm90_Bf16<Config_,
                                                           Stages,
                                                           Raster,
                                                           Mode,
                                                           Silu,
                                                           ClusterShape,
                                                           L2HintW,
                                                           MmaN,
                                                           SeparateMmaAtoms,
                                                           EpiM,
                                                           EpiStages>>;
};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int      Stages,
                  Order    Raster,
                  Striding Mode,
                  bool     Silu         = false,
                  class ClusterShape    = Shape<1, 1>,
                  int  L2HintW          = 0,
                  int  MmaN             = 0,
                  bool SeparateMmaAtoms = false,
                  int  EpiM             = 0,
                  int  EpiStages        = 0>
         class K>
void register_kernels(Collector& c)
{
    add_cublas(c, Sm90::is_compatible);

    add<K<_8x256x64_1x2<24, 80>, 3, kRowMajor, Striding::kFlat, true>>(c);      // refs: 274
    add<K<_8x256x64_1x1<24, 80>, 3, kRowMajor, Striding::kFlat, true>>(c);      // refs: 1267
    add<K<_16x256x64_2x1<24, 80>, 3, kRowMajor, Striding::kFlat, true>>(c);     // refs: 21
    add<K<_32x256x64_2x1<120, 192>, 3, kRowMajor, Striding::kFlat, true>>(c);   // refs: 235
    add<K<_64x256x64_2x1<120, 192>, 3, kRowMajor, Striding::kFlat, true>>(c);   // refs: 268
    add<K<_96x256x64_2x1<120, 192>, 3, kRowMajor, Striding::kFlat, true>>(c);   // refs: 295
    add<K<_128x256x64_2x1<120, 192>, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 1160
    add<K<_160x256x64_1x2<40, 232>, 3, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_192x256x64_1x2<40, 232>, 3, kRowMajor, Striding::kFlat, true, Shape<1, 1>, 0, 0, false, 192>>(c);
    add<K<_384x128x64_1x2<40, 232>, 3, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 0, 192>>(c);

    add<K<_8x128x64_1x2<24, 80>, 4, kColMajor, Striding::kBlocked>>(c);     // refs: 2637
    add<K<_16x128x64_1x2<24, 80>, 4, kColMajor, Striding::kBlocked>>(c);    // refs: 1419
    add<K<_32x128x64_1x2<24, 80>, 4, kColMajor, Striding::kBlocked>>(c);    // refs: 1375
    add<K<_96x128x64_1x2<120, 192>, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 740
    add<K<_192x256x64_1x2<40, 232>, 3, kColMajor, Striding::kBlocked, false, Shape<1, 1>, 0, 0, false, 192>>(c);

    add<K<_8x128x64_1x2<40, 80>, 4, kColMajor, Striding::kIndexed>>(c);            // refs: 1436
    add<K<_64x128x64_1x2<120, 192>, 4, kColMajor, Striding::kIndexed>>(c);         // refs: 846
    add<K<_96x128x64_1x2<120, 192>, 4, kColMajor, Striding::kIndexed>>(c);         // refs: 183
    add<K<_128x128x64_1x2<120, 192>, 4, kColMajor, Striding::kIndexed>>(c);        // refs: 665
    add<K<_256x128x64_1x2<128, 184>, 4, kColMajor, Striding::kIndexed>>(c);        // refs: 2207
    add<K<_8x256x64_1x2<40, 80>, 3, kColMajor, Striding::kIndexed, true>>(c);      // refs: 399
    add<K<_8x256x64_1x1<40, 80>, 3, kColMajor, Striding::kIndexed, true>>(c);      // refs: 1362
    add<K<_16x256x64_2x1<40, 80>, 3, kColMajor, Striding::kIndexed, true>>(c);     // refs: 1035
    add<K<_32x256x64_2x1<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);   // refs: 495
    add<K<_64x256x64_2x1<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);   // refs: 803
    add<K<_96x256x64_2x1<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);   // refs: 604
    add<K<_128x256x64_2x1<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 2420
    add<K<_160x256x64_1x2<104, 200>, 3, kColMajor, Striding::kIndexed, true>>(c);

    add<K<_256x128x64_1x2<128, 184>, 4, kRowMajor, Striding::kIndexed>>(c);  // refs: 1200

    add<K<_256x128x64_1x2<128, 184>, 4, kColMajor, Striding::kIndexed, false, Shape<1, 1>, 1>>(c);  // refs: 1543
}

Registrar reg(bf16, register_kernels<C::Type>);
}  // namespace

}  // namespace turbomind::gemm
