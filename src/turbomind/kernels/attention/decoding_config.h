// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "block_iterator.h"
#include "cta_map.h"
#include "impl_81616.h"
#include "impl_simt.h"
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

//////////////////////////////////////////////////////////////
template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh, HeadDim, std::enable_if_t<!(Qh > 2)>> {
    using Attention = Impl<MMA_SIMT, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh, HeadDim, std::enable_if_t<(Qh > 2)>> {
    using Attention = Impl<MMA_81616, T, T, 8, 1, 64, 8, 1, 16, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint8_t, Qh, HeadDim> {
    using Attention = Impl<MMA_81616, T, uint8_t, 8, 1, 64, 8, 1, 16, HeadDim, 5>;
    using CacheIter = GetBlockIterFactory<T, uint8_t, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint4_t, Qh, HeadDim> {
    using Attention = Impl<MMA_81616, T, uint4_t, 8, 1, 64, 8, 1, 16, HeadDim, 5>;
    using CacheIter = GetBlockIterFactory<T, uint4_t, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap>;
};

//////////////////////////////////////////////////////////////

template<class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm75, T, Tkv, Qh, HeadDim> {
    using Attention = Impl<MMA_81616, T, Tkv, 8, 1, 64, 8, 1, 16, HeadDim, 2>;
    using CacheIter = GetBlockIterFactory<T, Tkv, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

//////////////////////////////////////////////////////////////

template<class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, Tkv, Qh, HeadDim> {
    // Qh >= 4 is not beneficial for sm_70
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);

    using Attention = Impl<MMA_SIMT, T, Tkv, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
    using CacheIter = GetBlockIterFactory<T, Tkv, 64, HeadDim>;
    using Kernel    = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap>;
};

}  // namespace turbomind::attention
