// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_81616.h"
#include "src/turbomind/kernels/attention/impl_simt.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/attention/mainloop.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"

namespace turbomind::attention {

constexpr int kHeadDim = 128;
constexpr int kCTA_S   = 64;
constexpr int kWARP_S  = 16;

template<class Mainloop_, class CacheIter>
using KT = AttentionUniversal<arch::Sm80, Mainloop_, CacheIter, DecodingCtaMap>;

// T==Tkv, Qh<=2: SIMT, stages=3
template<class T, int Qh>
using Decoding_SIMT = KT<
    Mainloop<Sm80_CpAsync<3>, Impl<MMA_SIMT, T, T, Qh, 1, kCTA_S, Qh, 1, kWARP_S, kHeadDim, 3>>,
    GetBlockIterFactory<T, T, kCTA_S, kHeadDim>>;

// Qh>2: MMA_81616; Stages=3 for T==Tkv, Stages=5 for quant Tkv
// Qh = (Qh_+7)/8*8: Qh_=3..8→Qh=8, Qh_=9→Qh=16
template<class T, class Tkv, int Qh, int Stages>
using Decoding_MMA = KT<
    Mainloop<Sm80_CpAsync<Stages>, Impl<MMA_81616, T, Tkv, Qh, 1, kCTA_S, Qh, 1, kWARP_S, kHeadDim, Stages>>,
    GetBlockIterFactory<T, Tkv, kCTA_S, kHeadDim>>;

namespace {
Registrar reg([](Collector& c) {
    c.add<Decoding_SIMT<half, 1>>();
    c.add<Decoding_SIMT<half, 2>>();
    c.add<Decoding_MMA<half, half, 8, 3>>();
    c.add<Decoding_MMA<half, half, 16, 3>>();
    c.add<Decoding_MMA<half, uint8_t, 8, 5>>();
    c.add<Decoding_MMA<half, uint8_t, 16, 5>>();
    c.add<Decoding_MMA<half, uint4_t, 8, 5>>();
    c.add<Decoding_MMA<half, uint4_t, 16, 5>>();

#if ENABLE_BF16
    c.add<Decoding_SIMT<nv_bfloat16, 1>>();
    c.add<Decoding_SIMT<nv_bfloat16, 2>>();
    c.add<Decoding_MMA<nv_bfloat16, nv_bfloat16, 8, 3>>();
    c.add<Decoding_MMA<nv_bfloat16, nv_bfloat16, 16, 3>>();
    c.add<Decoding_MMA<nv_bfloat16, uint8_t, 8, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, uint8_t, 16, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, uint4_t, 8, 5>>();
    c.add<Decoding_MMA<nv_bfloat16, uint4_t, 16, 5>>();
#endif
});
}

}  // namespace turbomind::attention
