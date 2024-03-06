// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "cta_map.h"
#include "decoding_simt.h"
#include "decoding_sm80.h"
#include "mainloop_sm70.h"
#include "mainloop_sm80.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/mainloop.h"

namespace turbomind::attention {

template<class Arch, class T, class Tkv, int Qh, int HeadDim, class SFINAE = void>
struct DecodingConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<class Arch, class T, class Tkv, int Qh, int HeadDim>
using Decoding = typename DecodingConfig<Arch, T, Tkv, Qh, HeadDim>::Kernel;

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh, HeadDim, std::enable_if_t<(Qh <= 2)>> {
    using Attention = Impl<Sm70_Simt, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, int, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh, HeadDim, std::enable_if_t<(Qh > 2)>> {
    using Attention = Impl<Sm80_81616, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, int, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, int8_t, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, int8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, int, DecodingCtaMap>;
};

template<class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, Tkv, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, Tkv, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using Mainloop  = Mainloop<arch::Sm70, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, int, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, int8_t, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, int8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using Mainloop  = Mainloop<arch::Sm70, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, int, DecodingCtaMap>;
};

}  // namespace turbomind::attention
