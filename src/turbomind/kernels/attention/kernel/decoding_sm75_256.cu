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

constexpr int kHeadDim = 256;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;
constexpr int kStages  = 3;

// Qh = (Qh_+7)/8*8: Qh_=1..8 → Qh=8, Qh_=9 → Qh=16
// For 256 head dim, we use Qh=1 and Qh=9 (which maps to 16)
template<class T, class Tkv, int Qh>
using KT =
    AttentionUniversal<arch::Sm75,
                       Mainloop<arch::Sm70, Impl<MMA_81616, T, Tkv, Qh, 1, kCTA_S, Qh, 1, kWARP_S, kHeadDim, kStages>>,
                       GetBlockIterFactory<T, Tkv, kCTA_S, kHeadDim>,
                       DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, half, 1>>();
    c.add<KT<half, half, 16>>();  // Qh=9 maps to 16

    c.add<KT<half, uint8_t, 1>>();
    c.add<KT<half, uint8_t, 16>>();  // Qh=9 maps to 16

    c.add<KT<half, uint4_t, 1>>();
    c.add<KT<half, uint4_t, 16>>();  // Qh=9 maps to 16
});
}

}  // namespace turbomind::attention
