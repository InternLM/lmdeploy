// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_81616.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 576;

// Non-quant MLA config: CTA_H=16, CTA_S=32, WARP_H=8, WARP_S=16, Stages=2
template<class T>
using Decoding_F =
    AttentionUniversal<arch::Sm80,
                       Mainloop<Sm80_CpAsync<2>, Impl<MMA_81616, T, T, 16, 1, 32, 8, 1, 16, kHeadDim, 2>>,
                       GetBlockIterFactory<T, T, 32, kHeadDim>,
                       DecodingCtaMap>;

// Quant config: CTA_H=8, CTA_S=64, WARP_H=8, WARP_S=16, Stages=5
template<class T, class Tkv>
using Decoding_Q =
    AttentionUniversal<arch::Sm80,
                       Mainloop<Sm80_CpAsync<5>, Impl<MMA_81616, T, Tkv, 8, 1, 64, 8, 1, 16, kHeadDim, 5>>,
                       GetBlockIterFactory<T, Tkv, 64, kHeadDim>,
                       DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<Decoding_F<half>>();
    c.add<Decoding_Q<half, uint8_t>>();
    c.add<Decoding_Q<half, uint4_t>>();

#if ENABLE_BF16
    c.add<Decoding_F<nv_bfloat16>>();
    c.add<Decoding_Q<nv_bfloat16, uint8_t>>();
    c.add<Decoding_Q<nv_bfloat16, uint4_t>>();
#endif
});
}  // namespace

}  // namespace turbomind::attention
