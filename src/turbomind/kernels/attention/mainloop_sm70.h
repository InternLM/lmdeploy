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

    using SmemIterQ = typename Impl::SmemIterQ;
    using SmemIterK = typename Impl::SmemIterK;
    using SmemIterP = typename Impl::SmemIterP;
    using SmemIterV = typename Impl::SmemIterV;

    using ThreadMapKV = typename Impl::ThreadMapKV;
    using GmemIterK   = Sm70GmemIterator<T, ThreadMapKV, typename Impl::SmemLayoutK>;
    using GmemIterV   = Sm70GmemIterator<T, ThreadMapKV, typename Impl::SmemLayoutV>;

    using FragQ = typename Impl::FragQ;
    using FragS = typename Impl::FragS;
    using FragO = typename Impl::FragO;
    using FragP = typename Impl::FragP;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int CTA_S = Impl::CTA_S;

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS>
    __device__ void operator()(FragQ&         frag_Q,
                               GmemIterK&     gmem_K,
                               GmemIterV&     gmem_V,
                               BlockIter&     block_iter,
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
        gmem_K.SetSmem(storage.KV);
        gmem_V.SetSmem(storage.KV);

        SmemIterQ smem_Q{storage.Q};
        SmemIterP smem_P{storage.P};
        SmemIterK smem_K{storage.KV};
        SmemIterV smem_V{storage.KV};

        typename GmemIterK::Fragment frag_K;

        block_iter.SetTile(tile_iter);

        gmem_K.Load<true>(block_iter, frag_K, max_step - tile_iter * CTA_S);
        gmem_K.Save(frag_K);

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            Impl::Sync();

            typename GmemIterV::Fragment frag_V;
            gmem_V.Load<is_residue>(block_iter, frag_V, is_residue ? max_step - offset_K : CTA_S);

            block_iter.Advance();

            FragS frag_S{};

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_S, 0);

            gmem_V.Save(frag_V);

            Impl::Sync();

            if (tile_iter > 0) {
                gmem_K.Load<false>(block_iter, frag_K, CTA_S);
            }

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_O, 0);

            gmem_K.Save(frag_K);
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
        Impl::ForeachS(frag_S, [&](int hi, int qi, int si, float& score) {
            if (offset_Q + qi < offset_K + si) {
                score -= std::numeric_limits<float>::infinity();
            }
        });
    }
};

}  // namespace turbomind::attention