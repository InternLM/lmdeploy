// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_1688.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"

namespace turbomind::attention {

constexpr int kHeadDim = 128;
constexpr int kCTA_Q   = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_Q  = 16;
constexpr int kStages  = 2;

template<class T>
using KT = AttentionUniversal<arch::Sm75,
    Mainloop<arch::Sm70, Impl<MMA_1688, T, T, 1, kCTA_Q, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, kCTA_S, kHeadDim>,
    AttentionCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half>>();
});
}

}  // namespace turbomind::attention
