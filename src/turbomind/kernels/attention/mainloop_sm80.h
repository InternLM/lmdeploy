// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator_sm80.h"
#include "mainloop.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::attention {

template<int Stages>
struct Sm80_CpAsync {};

template<int Stages, class Impl_>
struct Mainloop<Sm80_CpAsync<Stages>, Impl_> {

    using Impl = Impl_;

    using T   = typename Impl::T;
    using Tkv = typename Impl::Tkv;

    using SmemIterQ = typename Impl::SmemIterQ;
    using SmemIterK = typename Impl::SmemIterK;
    using SmemIterP = typename Impl::SmemIterP;
    using SmemIterV = typename Impl::SmemIterV;

    using ThreadMapKV = typename Impl::ThreadMapKV;
    using GmemIterK   = Sm80GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutK, 0>;
    using GmemIterV   = Sm80GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutV, 1>;

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

    static constexpr auto kBatchKV = ThreadMapKV::kIterS / 2;

    static constexpr int kTileSizeKV = CTA_S * Impl::SmemLayoutK::kStride;
    static constexpr int kSmemSizeKV = Stages * kTileSizeKV;

    __device__ int SmemKVStep(int& offset)
    {
        auto ret = offset;
        offset += sizeof(Tkv) * kTileSizeKV;
        if (offset == sizeof(Tkv) * kSmemSizeKV) {
            offset -= sizeof(Tkv) * kSmemSizeKV;
        }
        return ret;
    }

    template<class... Args>
    __device__ void operator()(Args&&... args)
    {
        Run(Sm80_CpAsync<Stages>{}, ((Args&&)args)...);
    }

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS, int Stages_>
    __device__ void Run(Sm80_CpAsync<Stages_>,
                        FragQ&         frag_Q,
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
        gmem_K.SetSmem(storage.KV);
        gmem_V.SetSmem(storage.KV);

        SmemIterQ smem_Q{storage.Q};
        SmemIterP smem_P{storage.P};
        SmemIterK smem_K{storage.KV};
        SmemIterV smem_V{storage.KV};

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem(i * kTileSizeKV);
        }

        int kv_offset_r = 0;
        int kv_offset_w = 0;

        block_iter.SetTile(tile_iter);

        gmem_K.Prefetch<true>(block_iter, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
        __pipeline_commit();

        gmem_V.Prefetch<true>(block_iter, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
        __pipeline_commit();

        block_iter.Advance();

        PRAGMA_UNROLL
        for (int stages = 2; stages < Stages - 1; stages += 2) {
            gmem_K.Prefetch<false>(block_iter, CTA_S, SmemKVStep(kv_offset_w));
            __pipeline_commit();

            gmem_V.Prefetch<false>(block_iter, CTA_S, SmemKVStep(kv_offset_w));
            __pipeline_commit();

            block_iter.Advance();
        }

        FragK frag_K;
        FragV frag_V;

        Wait();

        auto rk = SmemKVStep(kv_offset_r);
        smem_K.Load(frag_K[0], 0, rk);

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            auto rv = SmemKVStep(kv_offset_r);
            auto wk = SmemKVStep(kv_offset_w);

            auto prefetch_K = [&](int k) {
                const int begin = k * kBatchKV;
                if (begin < ThreadMapKV::kIterS) {
                    gmem_K.Prefetch<false>(block_iter, begin, kBatchKV, CTA_S, wk);
                }
                if (begin + kBatchKV == ThreadMapKV::kIterS) {
                    __pipeline_commit();
                }
            };

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, rk, prefetch_K, [&] {
                Wait();
                smem_V.Load(frag_V[0], 0, rv);
            });

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            // store_S(frag_S, offset_K);

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            rk      = SmemKVStep(kv_offset_r);
            auto wv = SmemKVStep(kv_offset_w);

            auto prefetch_V = [&](int k) {
                const int begin = k * kBatchKV;
                if (begin < ThreadMapKV::kIterS) {
                    gmem_V.Prefetch<false>(block_iter, begin, kBatchKV, CTA_S, wv);
                }
                if (begin + kBatchKV == ThreadMapKV::kIterS) {
                    block_iter.Advance();
                    __pipeline_commit();
                }
            };

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, transform_V, rv, prefetch_V, [&] {
                Wait();
                smem_K.Load(frag_K[0], 0, rk);
            });
        };

        // PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{}, std::false_type{});
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS>
    __device__ void Run(Sm80_CpAsync<2>,
                        FragQ&         frag_Q,
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
        gmem_K.SetSmem(storage.KV);
        gmem_V.SetSmem(storage.KV);

        SmemIterK smem_K{storage.KV};
        SmemIterV smem_V{storage.KV};
        SmemIterQ smem_Q{storage.Q};
        SmemIterP smem_P{storage.P};

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem(i * kTileSizeKV);
        }

        block_iter.SetTile(tile_iter);

        gmem_K.Prefetch<true>(block_iter, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();

        FragK frag_K;
        FragV frag_V;

        Wait();
        smem_K.Load(frag_K[0], 0, 0);

        constexpr auto nop = [](int) {};

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            gmem_V.Prefetch<is_residue>(block_iter, max_step - offset_K, sizeof(Tkv) * kTileSizeKV);
            block_iter.Advance();
            __pipeline_commit();

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, 0, nop, [&] {
                Wait();
                smem_V.Load(frag_V[0], 0, sizeof(Tkv) * kTileSizeKV);
            });

            gmem_K.Prefetch<false>(block_iter, CTA_S, 0);
            __pipeline_commit();

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, transform_V, sizeof(Tkv) * kTileSizeKV, nop, [&] {
                Wait();
                smem_K.Load(frag_K[0], 0, 0);
            });
        };

        // PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{}, std::false_type{});
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        Impl::Sync();
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