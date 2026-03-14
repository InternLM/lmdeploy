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

constexpr int kHeadDim = 256;

// V100 (SM70) has 96 KB shared memory per SM.
// fp16 KV:          CTA_S=64, 2 stages (64KB smem).
// Quantized KV:     CTA_S=64, 3 stages (smaller element size fits 3 stages).

// ── fp16 KV: CTA_S=64, 2 stages ──────────────────────────────
template<class T, int kH>
using KT =
    AttentionUniversal<arch::Sm70,
                       Mainloop<arch::Sm70, Impl<MMA_SIMT, T, T, kH, 1, 64, kH, 1, 16, kHeadDim, 2>>,
                       GetBlockIterFactory<T, T, 64, kHeadDim>,
                       DecodingCtaMap>;

// ── Quantized KV: CTA_S=64, 3 stages ─────────────────────────
template<class T, class Tkv, int kH>
using KT_quant =
    AttentionUniversal<arch::Sm70,
                       Mainloop<arch::Sm70, Impl<MMA_SIMT, T, Tkv, kH, 1, 64, kH, 1, 16, kHeadDim, 3>>,
                       GetBlockIterFactory<T, Tkv, 64, kHeadDim>,
                       DecodingCtaMap>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half, 1>>();
    c.add<KT<half, 2>>();
    c.add<KT<half, 3>>();

    c.add<KT_quant<half, uint8_t, 1>>();
    c.add<KT_quant<half, uint8_t, 2>>();
    c.add<KT_quant<half, uint8_t, 3>>();

    c.add<KT_quant<half, uint4_t, 1>>();
    c.add<KT_quant<half, uint4_t, 2>>();
    c.add<KT_quant<half, uint4_t, 3>>();
});
}

}  // namespace turbomind::attention
