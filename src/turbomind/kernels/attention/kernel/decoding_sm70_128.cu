// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_simt.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 128;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;
constexpr int kStages  = 2;

// kH = Qh%3==0 ? 3 : (Qh%2==0 ? 2 : 1)
// kH=1 covers Qh ∈ {1,5,7}, kH=2 covers {2,4,8}, kH=3 covers {3,6,9}
template<class T, class Tkv, int kH>
using KT =
    AttentionUniversal<arch::Sm70,
                       Mainloop<arch::Sm70, Impl<MMA_SIMT, T, Tkv, kH, 1, kCTA_S, kH, 1, kWARP_S, kHeadDim, kStages>>,
                       GetBlockIterFactory<T, Tkv, kCTA_S, kHeadDim>,
                       DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, half, 1>>();
    c.add<KT<half, half, 2>>();
    c.add<KT<half, half, 3>>();

    c.add<KT<half, uint8_t, 1>>();
    c.add<KT<half, uint8_t, 2>>();
    c.add<KT<half, uint8_t, 3>>();

    c.add<KT<half, uint4_t, 1>>();
    c.add<KT<half, uint4_t, 2>>();
    c.add<KT<half, uint4_t, 3>>();
});
}

}  // namespace turbomind::attention
