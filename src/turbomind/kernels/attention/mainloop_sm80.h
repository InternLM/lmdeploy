// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator_sm80.h"
#include "mainloop.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cuda_pipeline_primitives.h>
#include <type_traits>

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

    template<class... Args>
    __device__ void operator()(Args&&... args)
    {
        Run(Sm80_CpAsync<Stages>{}, ((Args&&)args)...);
    }

    template<int Idx, class A, class B>
    __device__ static decltype(auto) Select(A&& a, B&& b)
    {
        if constexpr (Idx) {
            return (B&&)b;
        }
        else {
            return (A&&)a;
        }
    }

    template<int Batch, bool Advnace, class GmemIter, class BlockIter>
    __device__ static void Prefetch(GmemIter gmem_iter, BlockIter& block_iter, int k, int smem_offset)
    {
        const int begin = k * Batch;
        if (begin < ThreadMapKV::kIterS) {
            gmem_iter.Prefetch(std::false_type{}, block_iter, begin, Batch, CTA_S, smem_offset);
        }
        if (begin + Batch == ThreadMapKV::kIterS) {
            if constexpr (Advnace) {
                block_iter.Advance();
            }
            __pipeline_commit();
        }
    }

    // static constexpr int kSmemStepSize = sizeof(T) * Impl::SmemLayoutK::kSize;
    static constexpr int kSmemStepSize = Impl::SmemLayoutK::kSize;
    // static constexpr int kSmemStepSize = 1;

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

        block_iter.SetTile(tile_iter);

        PipeIter<Stages, kSmemStepSize> pipe_iter;

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem((++pipe_iter).w);
        }

        // 0
        gmem_K.Prefetch(std::true_type{}, block_iter, max_step - tile_iter * CTA_S, (++pipe_iter).w);
        __pipeline_commit();

        // 1
        gmem_V.Prefetch(std::true_type{}, block_iter, max_step - tile_iter * CTA_S, (++pipe_iter).w);
        __pipeline_commit();

        block_iter.Advance();

        PRAGMA_UNROLL
        for (int stages = 2; stages < Stages - 2; stages += 2) {
            // 2 + 2X
            gmem_K.Prefetch(std::true_type{}, block_iter, CTA_S, (++pipe_iter).w);
            __pipeline_commit();
            // 3 + 2X
            gmem_V.Prefetch(std::true_type{}, block_iter, CTA_S, (++pipe_iter).w);
            __pipeline_commit();

            block_iter.Advance();
        }

        if constexpr (Stages % 2 == 0) {
            // 2 + 2Y
            gmem_K.Prefetch(std::true_type{}, block_iter, CTA_S, (++pipe_iter).w);
            __pipeline_commit();
        }

        auto& gmem_0 = Select<Stages % 2>(gmem_V, gmem_K);
        auto& gmem_1 = Select<Stages % 2>(gmem_K, gmem_V);

        constexpr auto kBatch0 = Stages % 2 ? Impl::kBatchV : Impl::kBatchK;
        constexpr auto kBatch1 = Stages % 2 ? Impl::kBatchK : Impl::kBatchV;

        FragK frag_K;
        FragV frag_V;

        Wait();
        smem_K.Load(frag_K[0], 0, (++pipe_iter).r);

        auto loop = [&](auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            auto prefetch_0 = [&, pipe_iter](int k) {
                Prefetch<kBatch0, Stages % 2 == 0>(gmem_0, block_iter, k, pipe_iter.w);
            };

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, pipe_iter.r, prefetch_0, [&] {
                Wait();
                smem_V.Load(frag_V[0], 0, (++pipe_iter).r);
            });

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            auto prefetch_1 = [&, pipe_iter](int k) {
                Prefetch<kBatch1, Stages % 2 != 0>(gmem_1, block_iter, k, pipe_iter.w);
            };

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, transform_V, pipe_iter.r, prefetch_1, [&] {
                Wait();
                smem_K.Load(frag_K[0], 0, (++pipe_iter).r);
            });
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{});
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

#if 0
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
            gmem_K.ClearSmem(i * kSmemStepSize);
        }

        block_iter.SetTile(tile_iter);

        gmem_K.Prefetch<true>(block_iter, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();

        FragK frag_K;
        FragV frag_V;

        Wait();
        smem_K.Load(frag_K[0], 0, 0);

        constexpr auto nop = [](int){};

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            gmem_V.Prefetch<is_residue>(block_iter, max_step - offset_K, kSmemStepSize);
            __pipeline_commit();

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, 0, nop, [&] {
                Wait();
                smem_V.Load(frag_V[0], 0, kSmemStepSize);
            });

            block_iter.Advance();
            gmem_K.Prefetch<false>(block_iter, CTA_S, 0);
            __pipeline_commit();

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, transform_V, kSmemStepSize, nop, [&] {
                Wait();
                smem_K.Load(frag_K[0], 0, 0);
            });
        };

        PRAGMA_UNROLL
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

#else
    // Load      : K0,K1 | V0,K2,V1,K3 ...
    // Compute   :    K0 | K1,V0,K2,V1 ...
    // Conclusion:
    // - more reigster consumption (209 -> 250)
    // - more interleaved HMMA and FMA
    // - slight performance gain
    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS>
    __device__ void Run(Sm80_CpAsync<2>,
                        FragQ&         frag_Q,
                        GmemIterK&     gmem_K,
                        GmemIterV&     gmem_V,
                        TransformK&    transform_K,
                        TransformV&    transform_V,
                        BlockIter&     block_iter_,
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

        gmem_K.ClearSmem(0);
        gmem_K.ClearSmem(kSmemStepSize);

        block_iter_.SetTile(tile_iter);

        auto block_iter_K = block_iter_;
        auto block_iter_V = block_iter_;

        gmem_K.Prefetch(std::true_type{}, block_iter_K, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();
        block_iter_K.Advance();

        FragK frag_K;
        FragV frag_V;

        Wait();
        smem_K.Load(frag_K[0], 0, 0);

        FragS frag_S{};
        auto  _ = [&](int k) {
            if (k == 0) {
                gmem_K.Prefetch(std::false_type{}, block_iter_K, CTA_S, kSmemStepSize);
                __pipeline_commit();
            }
        };
        Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, 0, _, [&] {
            Wait();
            smem_K.Load(frag_K[0], 0, kSmemStepSize);
        });
        block_iter_K.Advance();

        auto loop = [&](auto is_residue, auto is_mask, auto is_last) {
            const int offset_K = tile_iter * CTA_S;

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }
            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;
            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            auto prefetch_V = [&](int k) {
                if (k == 0) {
                    gmem_V.Prefetch(is_residue, block_iter_V, max_step - offset_K, 0);
                    __pipeline_commit();
                }
            };
            if constexpr (!is_last) {
                clear(frag_S);
                Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, transform_K, kSmemStepSize, prefetch_V, [&] {
                    Wait();
                    smem_V.Load(frag_V[0], 0, 0);
                });
                block_iter_V.Advance();
            }
            else {
                prefetch_V(0);
                Wait();
                smem_V.Load(frag_V[0], 0, 0);
            }

            auto prefetch_K = [&](int k) {
                if (k == 0) {
                    gmem_K.Prefetch(std::false_type{}, block_iter_K, CTA_S, kSmemStepSize);
                    __pipeline_commit();
                }
            };
            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, transform_V, 0, prefetch_K, [&] {
                Wait();
                smem_K.Load(frag_K[0], 0, kSmemStepSize);
            });
            block_iter_K.Advance();
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{}, std::false_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 1; --tile_iter) {
            loop(std::false_type{}, std::false_type{}, std::false_type{});
        }

        if (tile_iter >= 0) {
            loop(std::false_type{}, std::false_type{}, std::true_type{});
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
#endif

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        Impl::Sync();
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