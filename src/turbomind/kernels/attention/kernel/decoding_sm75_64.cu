// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_81616.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"

namespace turbomind::attention {

constexpr int kHeadDim = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;
constexpr int kStages  = 2;

// Qh = (Qh_+7)/8*8: Qh_=1..8 → Qh=8, Qh_=9 → Qh=16
template<class T, class Tkv, int Qh>
using KT = AttentionUniversal<arch::Sm75,
    Mainloop<arch::Sm70, Impl<MMA_81616, T, Tkv, Qh, 1, kCTA_S, Qh, 1, kWARP_S, kHeadDim, kStages>>,
    GetBlockIterFactory<T, Tkv, kCTA_S, kHeadDim>,
    DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, half, 8>>();
    c.add<KT<half, half, 16>>();

    c.add<KT<half, uint8_t, 8>>();
    c.add<KT<half, uint8_t, 16>>();

    c.add<KT<half, uint4_t, 8>>();
    c.add<KT<half, uint4_t, 16>>();
});
}

}  // namespace turbomind::attention
