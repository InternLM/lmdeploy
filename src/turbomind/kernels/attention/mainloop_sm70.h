// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "iterator_sm70.h"
#include "mainloop.h"

namespace turbomind::attention {

template<class Impl_>
struct Mainloop<arch::Sm70, Impl_> {

    using Impl = Impl_;

    using T   = typename Impl::T;
    using Tkv = typename Impl::Tkv;

    using ThreadMapKV = typename Impl::ThreadMapKV;

    using GmemIterK_ = Sm70GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutK, 0>;
    using GmemIterV_ = Sm70GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutV, 1>;

    /// TODO: hide this behind a SFINAE gate so that `*KVp` stuff won't be needed for non-quantized impls
    using CombinedIterK =
        CombinedIterator<GmemIterK_, Sm70GmemIterator<T, typename Impl::ThreadMapKVp, typename Impl::SmemLayoutKVp, 2>>;
    using CombinedIterV =
        CombinedIterator<GmemIterV_, Sm70GmemIterator<T, typename Impl::ThreadMapKVp, typename Impl::SmemLayoutKVp, 3>>;

    using GmemIterK = std::conditional_t<std::is_same_v<T, Tkv>, GmemIterK_, CombinedIterK>;
    using GmemIterV = std::conditional_t<std::is_same_v<T, Tkv>, GmemIterV_, CombinedIterV>;

    using FragQ = typename Impl::FragQ;
    using FragS = typename Impl::FragS;
    using FragO = typename Impl::FragO;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int CTA_S = Impl::CTA_S;

    template<class CacheIter, class StoreS>
    __device__ void operator()(FragQ&         frag_Q,
                               CacheIter&     cache_iter,
                               FragO&         frag_O,
                               FragM&         frag_M,
                               FragL&         frag_L,
                               int            offset_Q,
                               int            max_step,
                               int            tile_iter,
                               int            mask_iter,
                               float          qk_scale,
                               SharedStorage& storage,
                               const StoreS&  store_S)
    {
        GmemIterK gmem_K{};
        GmemIterV gmem_V{};

        Impl::SetSmemKV(gmem_K, gmem_V, storage, true);

        typename GmemIterK::Fragment tmp_K;

        typename Impl::StateQK state_QK{storage, frag_Q};
        typename Impl::StatePV state_PV{storage};

        Impl::Sync();

        gmem_K.Load<true>(cache_iter, tmp_K, max_step - tile_iter * CTA_S);
        gmem_K.Save(tmp_K);

        constexpr auto nop = [](int) {};

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            typename GmemIterV::Fragment tmp_V;

            gmem_V.Load<is_residue>(cache_iter, tmp_V, is_residue ? max_step - offset_K : CTA_S);
            cache_iter.Advance();

            FragS frag_S{};

            Impl::Sync();
            state_QK.Load(0, 0);

            Impl::ComputeQK(state_QK, frag_S, 0, nop, [&] {});

            gmem_V.Save(tmp_V);

            if (tile_iter > 0) {
                gmem_K.Load<false>(cache_iter, tmp_K, CTA_S);
            }

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            Impl::ConvertStoP(frag_S, state_PV.frag_P, storage.P);

            Impl::Sync();
            state_PV.Load(0, 0);

            Impl::ComputePV(state_PV, frag_O, 0, nop, [&] {});

            gmem_K.Save(tmp_K);
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{}, std::false_type{});
        }
    }

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K)
    {
        Impl::ForeachS(frag_S, [&](int hi, int qi, int si, int ri, float& score) {
            if (offset_Q + qi < offset_K + si) {
                score -= std::numeric_limits<float>::infinity();
            }
        });
    }
};

}  // namespace turbomind::attention
