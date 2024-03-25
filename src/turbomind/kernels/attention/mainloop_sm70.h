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
    using GmemIterK   = Sm70GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutK, 0>;
    using GmemIterV   = Sm70GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutV, 1>;

    using TransformK = typename Impl::TransformK;
    using TransformV = typename Impl::TransformV;

    using FragQ = typename Impl::FragQ;
    using FragK = typename Impl::FragK;
    using FragV = typename Impl::FragV;
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
                               TransformK&    transform_K,
                               TransformV&    transform_V,
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
        gmem_K.SetSmem(Impl::GetSmemK(storage));
        gmem_V.SetSmem(Impl::GetSmemV(storage));

        SmemIterQ smem_Q{storage.Q};
        SmemIterP smem_P{storage.P};
        SmemIterK smem_K{Impl::GetSmemK(storage)};
        SmemIterV smem_V{Impl::GetSmemV(storage)};

        typename GmemIterK::Fragment tmp_K;

        block_iter.SetTile(tile_iter);

        FragK frag_K;
        FragV frag_V;

        Impl::Sync();

        gmem_K.Load<true>(block_iter, tmp_K, max_step - tile_iter * CTA_S);
        gmem_K.Save(tmp_K);

        constexpr auto nop = [](int) {};

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            typename GmemIterV::Fragment tmp_V;

            gmem_V.Load<is_residue>(block_iter, tmp_V, is_residue ? max_step - offset_K : CTA_S);
            block_iter.Advance();

            FragS frag_S{};

            Impl::Sync();
            smem_K.Load(frag_K[0], 0, 0);

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, 0, nop, [&] {});

            gmem_V.Save(tmp_V);

            if (tile_iter > 0) {
                gmem_K.Load<false>(block_iter, tmp_K, CTA_S);
            }

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            FragP frag_P;
            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::Sync();
            smem_V.Load(frag_V[0], 0, 0);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, transform_V, 0, nop, [&] {});

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
