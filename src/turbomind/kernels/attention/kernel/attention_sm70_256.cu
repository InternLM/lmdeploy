// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_884.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 256;

// Default config: CTA_Q=64 for throughput
constexpr int kCTA_Q  = 64;
constexpr int kCTA_S  = 64;
constexpr int kWARP_Q = 16;
constexpr int kStages = 2;

template<class T>
using KT = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884, T, T, 1, kCTA_Q, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, kCTA_S, kHeadDim>,
    AttentionCtaMap>;

// Higher-occupancy config: CTA_Q=32 halves Q shared memory, allowing more CTAs/SM
// Better for latency-sensitive prefill with HeadDim=256 on V100 (96KB smem)
constexpr int kCTA_Q2  = 32;
constexpr int kWARP_Q2 = 16;

template<class T>
using KT2 = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884, T, T, 1, kCTA_Q2, kCTA_S, 1, kWARP_Q2, kCTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, kCTA_S, kHeadDim>,
    AttentionCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half>>();
    c.add<KT2<half>>();
});
}

}  // namespace turbomind::attention
