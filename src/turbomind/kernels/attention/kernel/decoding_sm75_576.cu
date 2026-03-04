// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_81616.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 576;

// MLA config for all Tkv: CTA_H=16, CTA_S=16, WARP_H=8, WARP_S=16, Stages=2
template<class T, class Tkv>
using KT = AttentionUniversal<arch::Sm75,
    Mainloop<arch::Sm70, Impl<MMA_81616, T, Tkv, 16, 1, 16, 8, 1, 16, kHeadDim, 2>>,
    GetBlockIterFactory<T, Tkv, 16, kHeadDim>,
    DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, half>>();
    c.add<KT<half, uint8_t>>();
    c.add<KT<half, uint4_t>>();
});
}

}  // namespace turbomind::attention
