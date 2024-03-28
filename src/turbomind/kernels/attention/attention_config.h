// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "cta_map.h"
#include "impl_sm70.h"
#include "impl_sm75.h"
#include "impl_sm80.h"
#include "linear_iterator.h"
#include "mainloop_sm70.h"
#include "mainloop_sm80.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/decoding_config.h"
#include "src/turbomind/kernels/attention/impl.h"

namespace turbomind::attention {

enum class CacheType {
    kLinear,
    kBlock,
};

template<class Arch, class T, class Tkv, int Qh, int HeadDim, CacheType cache_type>
struct AttentionConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<CacheType type, class T, class Tkv, int CTA_S, int HeadDim>
using GetCacheIterFactory = std::conditional_t<type == CacheType::kLinear,
                                               LinearIteratorFactory<T, CTA_S, HeadDim>,
                                               GetBlockIterFactory<T, Tkv, CTA_S, HeadDim>>;

template<class T, class Tkv, int Qh, int HeadDim>
struct AttentionConfig<arch::Sm80, T, Tkv, Qh, HeadDim, CacheType::kLinear> {
    static constexpr int CTA_Q  = 64 / Qh;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 64;
    //
    using Attention = Impl<Sm80_16816, T, Tkv, Qh, CTA_Q, CTA_S, Qh, WARP_Q, WARP_S, HeadDim, 2>;
    using Mainloop  = Mainloop<Sm80_CpAsync<2>, Attention>;
    using CacheIter = LinearIteratorFactory<T, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, AttentionCtaMap>;
};

template<class T, class Tkv, int Qh, int HeadDim>
struct AttentionConfig<arch::Sm80, T, Tkv, Qh, HeadDim, CacheType::kBlock> {
    static constexpr int CTA_Q  = 64 / Qh;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 64;
    //
    using Attention = Impl<Sm80_16816, T, Tkv, Qh, CTA_Q, CTA_S, Qh, WARP_Q, WARP_S, HeadDim, 3>;
    using Mainloop  = Mainloop<Sm80_CpAsync<3>, Attention>;
    using CacheIter = LinearIteratorFactory<T, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, AttentionCtaMap>;
};

template<class T, class Tkv, int Qh, int HeadDim, CacheType type>
struct AttentionConfig<arch::Sm75, T, Tkv, Qh, HeadDim, type> {
    static constexpr int CTA_Q  = 64 / Qh;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 64;
    //
    using Attention = Impl<Sm75_1688, T, Tkv, Qh, CTA_Q, CTA_S, Qh, WARP_Q, WARP_S, HeadDim, 2>;
    using Mainloop  = Mainloop<arch::Sm70, Attention>;
    using CacheIter = GetCacheIterFactory<type, T, Tkv, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, AttentionCtaMap>;
};

template<class T, class Tkv, int HeadDim, CacheType type>
struct AttentionConfig<arch::Sm70, T, Tkv, 1, HeadDim, type> {
    static constexpr int CTA_Q  = 64;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 16;
    static constexpr int WARP_S = 64;
    //
    using Attention = Impl<Sm70_884, T, Tkv, 1, CTA_Q, CTA_S, 1, WARP_Q, WARP_S, HeadDim, 2>;
    using Mainloop  = Mainloop<arch::Sm70, Attention>;
    using CacheIter = GetCacheIterFactory<type, T, Tkv, CTA_S, HeadDim>;
    using Kernel    = AttentionUniversal<Mainloop, CacheIter, AttentionCtaMap>;
};

}  // namespace turbomind::attention
