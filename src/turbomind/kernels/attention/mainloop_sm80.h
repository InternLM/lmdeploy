// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator_sm80.h"
#include "mainloop.h"
#include "src/turbomind/kernels/core/pipe_iter.h"
#include <cuda_pipeline_primitives.h>
#include <type_traits>

namespace turbomind::attention {

template<int Stages>
struct Sm80_CpAsync {
};

template<int Stages, class Impl_>
struct Mainloop<Sm80_CpAsync<Stages>, Impl_> {

    using Impl = Impl_;

    using T   = typename Impl::T;
    using Tkv = typename Impl::Tkv;

    static constexpr std::false_type false_c{};
    static constexpr std::true_type  true_c{};

    static constexpr int CTA_S = Impl::CTA_S;

    using ThreadMapKV = typename Impl::ThreadMapKV;

    using GmemIterK_ = Sm80GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutK, 0>;
    using GmemIterV_ = Sm80GmemIterator<Tkv, ThreadMapKV, typename Impl::SmemLayoutV, 1>;

    /// TODO: hide this behind a SFINAE gate so that `*KVp` stuff won't be needed for non-quantized impls
    using CombinedIterK =
        CombinedIterator<GmemIterK_, Sm80GmemIterator<T, typename Impl::ThreadMapKVp, typename Impl::SmemLayoutKVp, 2>>;
    using CombinedIterV =
        CombinedIterator<GmemIterV_, Sm80GmemIterator<T, typename Impl::ThreadMapKVp, typename Impl::SmemLayoutKVp, 3>>;

    using GmemIterK = std::conditional_t<std::is_same_v<T, Tkv>, GmemIterK_, CombinedIterK>;
    using GmemIterV = std::conditional_t<std::is_same_v<T, Tkv>, GmemIterV_, CombinedIterV>;

    using FragQ = typename Impl::FragQ;
    using FragS = typename Impl::FragS;
    using FragO = typename Impl::FragO;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using SharedStorage = typename Impl::SharedStorage;

    template<class... Args>
    __device__ void operator()(Args&&... args)
    {
        Run(Sm80_CpAsync<Stages>{}, ((Args &&) args)...);
    }

    template<int Idx, class A, class B>
    __device__ static decltype(auto) Select(A&& a, B&& b)
    {
        if constexpr (Idx) {
            return (B &&) b;
        }
        else {
            return (A &&) a;
        }
    }

    template<int Batch, bool Advnace, class GmemIter, class BlockIter>
    __device__ static void Prefetch(GmemIter gmem_iter, BlockIter& block_iter, int k, int pipe_iter)
    {
        const int begin = k * Batch;
        if (begin < ThreadMapKV::kIterS) {
            gmem_iter.Prefetch(false_c, block_iter, begin, Batch, CTA_S, pipe_iter);
        }
        if (begin + Batch == ThreadMapKV::kIterS) {
            if constexpr (Advnace) {
                block_iter.Advance();
            }
            __pipeline_commit();
        }
    }

    template<class CacheIter, class StoreS, int Stages_>
    __device__ void Run(Sm80_CpAsync<Stages_>,
                        FragQ&         frag_Q,
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
        // multi-stage: pipe_iter * size
        //   two-stage: constant offset

        GmemIterK gmem_K{};
        GmemIterV gmem_V{};

        Impl::SetSmemKV(gmem_K, gmem_V, storage, false);

        PipeIter<Stages> pipe_iter;

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem((++pipe_iter).w);
        }

        Impl::Sync();

        // 0
        gmem_K.Prefetch(true_c, cache_iter, max_step - tile_iter * CTA_S, (++pipe_iter).w);
        __pipeline_commit();

        // 1
        gmem_V.Prefetch(true_c, cache_iter, max_step - tile_iter * CTA_S, (++pipe_iter).w);
        __pipeline_commit();

        cache_iter.Advance();

        PRAGMA_UNROLL
        for (int stages = 2; stages < Stages - 2; stages += 2) {
            // 2 + 2X
            gmem_K.Prefetch(false_c, cache_iter, CTA_S, (++pipe_iter).w);
            __pipeline_commit();
            // 3 + 2X
            gmem_V.Prefetch(false_c, cache_iter, CTA_S, (++pipe_iter).w);
            __pipeline_commit();

            cache_iter.Advance();
        }

        if constexpr (Stages % 2 == 0) {
            // 2 + 2Y
            gmem_K.Prefetch(false_c, cache_iter, CTA_S, (++pipe_iter).w);
            __pipeline_commit();
        }

        auto& gmem_0 = Select<Stages % 2>(gmem_V, gmem_K);
        auto& gmem_1 = Select<Stages % 2>(gmem_K, gmem_V);

        constexpr auto kBatch0 = Stages % 2 ? Impl::kBatchV : Impl::kBatchK;
        constexpr auto kBatch1 = Stages % 2 ? Impl::kBatchK : Impl::kBatchV;

        typename Impl::StateQK state_QK{storage, frag_Q};
        typename Impl::StatePV state_PV{storage};

        Wait();
        state_QK.Load(0, (++pipe_iter).r);

        auto loop = [&](auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            auto prefetch_0 = [&, pipe_iter](int k) {
                Prefetch<kBatch0, Stages % 2 == 0>(gmem_0, cache_iter, k, pipe_iter.w);
            };

            Impl::ComputeQK(state_QK, frag_S, pipe_iter.r, prefetch_0, [&] {
                Wait();
                state_PV.Load(0, (++pipe_iter).r);
            });

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            Impl::ConvertStoP(frag_S, state_PV.frag_P, storage);

            auto prefetch_1 = [&, pipe_iter](int k) {
                Prefetch<kBatch1, Stages % 2 != 0>(gmem_1, cache_iter, k, pipe_iter.w);
            };

            Impl::ComputePV(state_PV, frag_O, pipe_iter.r, prefetch_1, [&] {
                Wait();
                state_QK.Load(0, (++pipe_iter).r);
            });
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(true_c);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; --tile_iter) {
            loop(false_c);
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

#if 0
    template<class CacheIter, class StoreS>
    __device__ void Run(Sm80_CpAsync<2>,
                        FragQ&         frag_Q,
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

        Impl::SetSmemKV(gmem_K, gmem_V, storage, false);

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem(i);
        }

        gmem_K.Prefetch(true_c, cache_iter, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();

        typename Impl::StateQK state_QK{storage, frag_Q};
        typename Impl::StatePV state_PV{storage};

        Wait();
        state_QK.Load(0, 0);

        constexpr auto _ = [](int){};

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            auto prefetch_V = [&](int k) {
                if (k == 0) {
                    gmem_V.Prefetch(is_residue, cache_iter, max_step - offset_K, 1);
                    __pipeline_commit();
                }
            };
            prefetch_V(0);

            Impl::ComputeQK(state_QK, frag_S, 0, _, [&] {
                Wait();
                state_PV.Load(0, 1);
            });

            cache_iter.Advance();

            auto prefetch_K = [&](int k) {
                if (k == 0) {
                    gmem_K.Prefetch(false_c, cache_iter, CTA_S, 0);
                    __pipeline_commit();
                }
            };
            prefetch_K(0);

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            Impl::ConvertStoP(frag_S, state_PV.frag_P, storage);

            Impl::ComputePV(state_PV, frag_O, 1, _, [&] {
                Wait();
                state_QK.Load(0, 0);
            });
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(true_c, true_c);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; --tile_iter) {
            loop(false_c, false_c);
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

#elif 1
    // Load      : K0,K1 | V0,K2,V1,K3 ...
    // Compute   :    K0 | K1,V0,K2,V1 ...
    // - more register consumption
    // - more interleaved HMMA and FMA
    // - slight performance gain
    template<class CacheIter, class StoreS>
    __device__ void Run(Sm80_CpAsync<2>,
                        FragQ&         frag_Q,
                        CacheIter&     cache_iter_,
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

        Impl::SetSmemKV(gmem_K, gmem_V, storage, false);

        gmem_K.ClearSmem(0);
        gmem_K.ClearSmem(1);

        auto cache_iter_K = cache_iter_;
        auto cache_iter_V = cache_iter_;

        gmem_K.Prefetch(true_c, cache_iter_K, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();
        cache_iter_K.Advance();

        typename Impl::StateQK state_QK{storage, frag_Q};
        typename Impl::StatePV state_PV{storage};

        Wait();
        state_QK.Load(0, 0);

        FragS frag_S{};
        auto  _ = [&](int k) {
            if (k == 0) {
                gmem_K.Prefetch(false_c, cache_iter_K, CTA_S, 1);
                __pipeline_commit();
            }
        };
        Impl::ComputeQK(state_QK, frag_S, 0, _, [&] {
            Wait();
            state_QK.Load(0, 1);
        });
        cache_iter_K.Advance();

        auto loop = [&](auto is_residue, auto is_mask, auto is_last) {
            const int offset_K = tile_iter * CTA_S;

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }
            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            Impl::ConvertStoP(frag_S, state_PV.frag_P, storage);

            auto prefetch_V = [&](int k) {
                if (k == 0) {
                    gmem_V.Prefetch(is_residue, cache_iter_V, max_step - offset_K, 0);
                    __pipeline_commit();
                }
            };
            if constexpr (!is_last) {
                clear(frag_S);
                Impl::ComputeQK(state_QK, frag_S, 1, prefetch_V, [&] {
                    Wait();
                    state_PV.Load(0, 0);
                });
                cache_iter_V.Advance();
            }
            else {
                prefetch_V(0);
                Wait();
                state_PV.Load(0, 0);
            }

            auto prefetch_K = [&](int k) {
                if (k == 0) {
                    gmem_K.Prefetch(false_c, cache_iter_K, CTA_S, 1);
                    __pipeline_commit();
                }
            };
            Impl::ComputePV(state_PV, frag_O, 0, prefetch_K, [&] {
                Wait();
                state_QK.Load(0, 1);
            });
            cache_iter_K.Advance();
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(true_c, true_c, false_c);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 1; --tile_iter) {
            loop(false_c, false_c, false_c);
        }

        if (tile_iter >= 0) {
            loop(false_c, false_c, true_c);
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
