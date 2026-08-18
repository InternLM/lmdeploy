// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_simt.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 192;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;
constexpr int kStages  = 3;
constexpr int kQh      = 1;

// HeadDim=192 uses SIMT+kStages for all Tkv (incl. uint8_t), kQh=1 only
template<class T, class Tkv>
using KT = AttentionUniversal<
    arch::Sm80,
    Mainloop<Sm80_CpAsync<kStages>, Impl<MMA_SIMT, T, Tkv, kQh, 1, kCTA_S, kQh, 1, kWARP_S, kHeadDim, kStages>>,
    GetBlockIterFactory<T, Tkv, kCTA_S, kHeadDim>,
    DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, half>>();
    c.add<KT<half, uint8_t>>();

#if ENABLE_BF16
    c.add<KT<nv_bfloat16, nv_bfloat16>>();
    c.add<KT<nv_bfloat16, uint8_t>>();
#endif
});
}  // namespace

}  // namespace turbomind::attention
