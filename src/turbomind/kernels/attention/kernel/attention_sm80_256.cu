// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_16816.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 256;
constexpr int kCTA_Q   = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_Q  = 16;
constexpr int kStages  = 2;

template<class T>
using KT = AttentionUniversal<
    arch::Sm80,
    Mainloop<Sm80_CpAsync<kStages>, Impl<MMA_16816, T, T, 1, kCTA_Q, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, kCTA_S, kHeadDim>,
    AttentionCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half>>();
#if ENABLE_BF16
    c.add<KT<nv_bfloat16>>();
#endif
});
}  // namespace

}  // namespace turbomind::attention
