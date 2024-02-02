// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "cta_map.h"
#include "impl_sm80.h"
#include "mainloop_sm80.h"
#include "src/turbomind/kernels/attention/attention_universal.h"

namespace turbomind::attention {

template<class Arch, class T, class Tkv, class BlockSeqLen, int HeadDim>
struct AttentionConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<class T, class Tkv, class BlockSeqLen, int HeadDim>
struct AttentionConfig<arch::Sm80, T, Tkv, BlockSeqLen, HeadDim> {
    static constexpr int CTA_Q  = 64;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 64;
    //
    using Attention = Impl<Sm80_16816, T, Tkv, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, HeadDim, 2>;
    using Mainloop  = Mainloop<Sm80_CpAsync<2>, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, BlockSeqLen, AttentionCtaMap>;
};

}  // namespace turbomind::attention