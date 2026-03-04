// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_simt.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"

namespace turbomind::attention {

constexpr int kHeadDim = 576;

// CTA_H=2, CTA_S=16, WARP_H=1, WARP_S=8, Stages=2
template<class T, class Tkv>
using KT = AttentionUniversal<arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_SIMT, T, Tkv, 2, 1, 16, 1, 1, 8, kHeadDim, 2>>,
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
