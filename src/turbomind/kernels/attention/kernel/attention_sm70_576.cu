// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_884.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

// HeadDim=576 on Sm70: kCTA_S=32, WARP_S=kCTA_S to fit within V100's 96 KB shared memory limit
constexpr int kHeadDim = 576;
constexpr int kCTA_Q   = 64;
constexpr int kCTA_S   = 32;
constexpr int kWARP_Q  = 16;
constexpr int kStages  = 2;

template<class T>
using KT = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884, T, T, 1, kCTA_Q, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, kCTA_S, kHeadDim>,
    AttentionCtaMap>;

namespace {
Registrar reg([](Collector& c) { c.add<KT<half>>(); });
}

}  // namespace turbomind::attention
