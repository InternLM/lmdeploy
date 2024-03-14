// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "block_iterator.h"
#include "cta_map.h"
#include "decoding_simt.h"
#include "decoding_sm80.h"
#include "mainloop_sm70.h"
#include "mainloop_sm80.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/mainloop.h"

namespace turbomind::attention {

template<class Arch, class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<class T, class Tkv, int CTA_S, int HeadDim>
using GetBlockIterFactory = BlockIteratorFactory<T, Tkv, block::Layout<block::Config<T, Tkv, HeadDim>>, CTA_S>;

template<class Arch, class T, class Tkv, int Qh, int HeadDim>
using Decoding = typename DecodingConfig<Arch, T, Tkv, Qh, HeadDim>::Kernel;

template<class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, Tkv, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, Tkv, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using CacheIter = GetBlockIterFactory<T, Tkv, 64, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, DecodingCtaMap>;
};

template<class T, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, 8, HeadDim> {
    using Attention = Impl<Sm80_81616, T, T, 8, 1, 64, 8, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint8_t, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using CacheIter = GetBlockIterFactory<T, uint8_t, 64, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint4_t, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, uint4_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using CacheIter = GetBlockIterFactory<T, uint4_t, 64, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, DecodingCtaMap>;
};


template<class T, class Tkv, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, Tkv, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, Tkv, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using Mainloop  = Mainloop<arch::Sm70, Attention>;
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, DecodingCtaMap>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, uint8_t, Qh, HeadDim> {
    using Attention = Impl<Sm70_Simt, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using Mainloop  = Mainloop<arch::Sm70, Attention>;
    using CacheIter = GetBlockIterFactory<T, uint8_t, 64, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, DecodingCtaMap>;
};

}  // namespace turbomind::attention
