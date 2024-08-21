// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

template<class Tc>
struct ChannelCombination_v2 {
    const Tc* __restrict__ scale_ptr;
    const Tc* __restrict__ bias_ptr;

    template<class T, int V, int S, int C>
    __device__ void operator()(Array<T, V> (&x)[S][C], int2 cta_cs, int2 thr_cs, int2 delta_cs, int2 end_cs) const
    {
        // T scale[S];

        Array<T, S> scale;
        fill(scale, T(1));

        if (scale_ptr) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                const int ss = thr_cs.y + s * delta_cs.y;
                if (ss < end_cs.y) {
                    scale[s] = static_cast<T>(__ldg(scale_ptr + ss + cta_cs.y));
                }
            }
        }

        T bias[S]{};

        if (bias_ptr) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                const int ss = thr_cs.y + s * delta_cs.y;
                if (ss < end_cs.y) {
                    bias[s] = static_cast<T>(__ldg(bias_ptr + ss + cta_cs.y));
                }
            }
        }

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                using namespace ops;
                x[s][c] = x[s][c] * scale[s] + bias[s];
            }
        }
    }
};

template<class Tc>
struct ChannelCombination_v3 {
    const Tc* __restrict__ scale_bias_ptr;

    template<class T, int V, int S, int C, int delta_c, int delta_s, class Pred>
    __device__ void operator()(Array<T, V> (&x)[S][C], int2 cs0, pair<delta_c, delta_s>, Pred& pred) const
    {
        __align__(16) Array<Tc, 2> scale_bias[S];

        if (scale_bias_ptr) {
            constexpr int ds  = sizeof(Tc) * delta_s;
            auto          ptr = reinterpret_cast<const char*>(scale_bias_ptr + cs0.y);
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                if (pred(s, 0)) {
                    Ldg(scale_bias[s], reinterpret_cast<const Tc*>(ptr));
                }
                ptr += ds;
            }
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                auto tmp = cast<T>(scale_bias[s]);
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    using namespace ops;
                    x[s][c] = x[s][c] * tmp[0] + tmp[1];
                }
            }
        }
    }
};

template<class Tc>
struct MatrixCombination_v2 {
    float alpha;
    float beta;

    const Tc* C_ptr;  // can't `__restrict__` since it may be alias of `D`
    int64_t   ldc;

    template<class T, int N, int S, int C, int delta_c, int delta_s, class Pred>
    __device__ void operator()(Array<T, N> (&x)[S][C], int2 cs0, pair<delta_c, delta_s>, Pred& pred) const
    {
        Array<Tc, N> frag[S][C]{};
        if (beta) {
            constexpr int dc  = sizeof(Tc) * delta_c;
            const int     ds  = sizeof(Tc) * delta_s * ldc;
            auto          ptr = reinterpret_cast<const char*>(C_ptr + cs2idx(cs0, (int64_t)ldc));
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    if (pred(s, c)) {
                        Load(frag[s][c], reinterpret_cast<const Tc*>(ptr));
                    }
                    ptr += dc;
                }
                ptr -= dc * C;
                ptr += ds;
            }
        }

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                using namespace ops;
                x[s][c] = x[s][c] * alpha + cast<T>(frag[s][c]) * beta;
            }
        }
    }
};

template<class Tc>
struct MatrixCombination_v3 {
    float alpha;
    float beta;

    const Tc* C_ptr;  // can't `__restrict__` since it may be alias of `D`
    int64_t   ldc;

    template<class T, int N, int S, int C, int delta_c, int delta_s, class Pred>
    __device__ void operator()(Array<T, N> (&x)[S][C], int2 cs0, pair<delta_c, delta_s>, Pred& pred) const
    {

        if (beta) {
            Array<Tc, N>  frag[S][C];
            constexpr int dc  = sizeof(Tc) * delta_c;
            const int     ds  = sizeof(Tc) * delta_s * ldc;
            auto          ptr = reinterpret_cast<const char*>(C_ptr + cs2idx(cs0, (int64_t)ldc));
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    if (pred(s, c)) {
                        Load(frag[s][c], reinterpret_cast<const Tc*>(ptr));
                        using namespace ops;
                        x[s][c] = x[s][c] * alpha + cast<T>(frag[s][c]) * beta;
                    }
                    ptr += dc;
                }
                ptr -= dc * C;
                ptr += ds;
            }
        }
        else if (alpha != 1.f) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    using namespace ops;
                    x[s][c] = x[s][c] * alpha;
                }
            }
        }
    }
};

template<class Act>
struct GatedActivation {
    template<class T, int N>
    __device__ static void apply(Array<T, N>& x)
    {
        static_assert(N % 2 == 0);
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            x[i / 2] = static_cast<T>(Act::apply(x[i]) * x[i + 1]);
        }
    }
};

struct Silu {
    __device__ static float apply(float x)
    {
        return fdividef(x, 1.f + expf(-x));
    }
};

template<class Tc>
struct EpilogueParam {
    int m;
    int n;
    Tc* C;
    int ldc;

    float* partial_C;
    int    partial_C_ld;

    int* locks;  // (m/cta_m, n/cta_n, k)

    // ChannelCombination_v3<Tc> combine_chn;
    MatrixCombination_v3<Tc> combine_mat;
    bool                     silu_act;
};

template<class Tc_, int M, int N, int TM_, int TN_, int THREADS, class RearrangeC, class OperandC, bool SplitK_>
struct Epilogue_ {

    using Dtype = typename OperandC::Dtype;

    static constexpr auto kOrder = OperandC::kOrder;
    static constexpr auto SplitK = SplitK_;

    using Tc = Tc_;

    static constexpr int TM = TM_;
    static constexpr int TN = TN_;

    using SmemLayout = decltype(OperandC::GetSmemLayout::apply(pair<TM, TN>{}));

    using SmemAccessorV2 = SmemAccessorV2<Dtype, SmemLayout, kOrder>;

    using SharedStorage = Array<Dtype, SmemLayout::kSize>;

    using Map = decltype(OperandC::GetThreadMap::apply(pair<M, N>{}, constant<THREADS>{}));

    static constexpr int S       = Map::kIterS;
    static constexpr int C       = Map::kIterC;
    static constexpr int kAccess = Map::kAccessC;

    template<class T>
    using OutputC = Array<T, kAccess>;

    template<class FragC>
    __device__ void Rearrange(FragC& frag_C, SharedStorage& storage, OutputC<Dtype> (&out)[S][C])
    {
        SmemAccessorV2 smem_C{storage.data()};

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

        constexpr int kPeriodC = ceil_div(SmemLayout::C0, Map::kDeltaC);
        constexpr int kPeriodS = ceil_div(SmemLayout::S0, Map::kDeltaS);

        int phases[kPeriodS][kPeriodC];
        PRAGMA_UNROLL
        for (int s = 0; s < kPeriodS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < kPeriodC; ++c) {
                phases[s][c] = SmemLayout::apply(s * Map::kDeltaS + thr_cs.y, c * Map::kDeltaC + thr_cs.x);
            }
        }

        constexpr bool kRaked = true;

        PRAGMA_UNROLL
        for (int m = 0; m < M; m += TM) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += TN) {
                // Store to shared memory
                RearrangeC::apply(frag_C, smem_C, {m, n}, pair<TM, TN>{});

                // Load from shared memory
                PRAGMA_UNROLL
                for (int s = 0; s < S; ++s) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < C; ++c) {
                        const int cc = c * Map::kDeltaC + thr_cs.x;
                        const int ss = s * Map::kDeltaS + thr_cs.y;

                        const int2 mn =
                            kRaked ? cs2mk<kOrder>(c * Map::kDeltaC, s * Map::kDeltaS) : cs2mk<kOrder>(cc, ss);
                        const int  mm   = mn.x - m;
                        const int  nn   = mn.y - n;
                        const bool mask = (M <= TM || (0 <= mm && mm < TM)) && ((N <= TN) || (0 <= nn && nn < TN));

                        const int2 _cs      = mk2cs<kOrder>(m, n);
                        const int  offset_0 = SmemLayout::apply(  //
                            s / kPeriodS * kPeriodS * Map::kDeltaS - _cs.y,
                            c / kPeriodC * kPeriodC * Map::kDeltaC - _cs.x);
                        const int  offset_p = phases[s % kPeriodS][c % kPeriodC];

                        if (mask) {
                            Load(out[s][c], &storage[offset_0 + offset_p]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    template<class VecC, class T, class Pred>
    __device__ void StoreC(const VecC& vec_C, T* data_C, int ldc, int2 cs0, Pred& pred)
    {
        constexpr int dc  = sizeof(T) * Map::kDeltaC;
        const int     ds  = sizeof(T) * Map::kDeltaS * ldc;
        auto          ptr = reinterpret_cast<char*>(data_C + cs2idx(cs0, (int64_t)ldc));
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const auto tmp = cast<T>(vec_C[s][c]);
                if (pred(s, c)) {
                    Store(reinterpret_cast<T*>(ptr), tmp);
                }
                ptr += dc;
            }
            ptr -= dc * C;
            ptr += ds;
        }
    }

    template<class FragC, class Pred>
    __device__ void Reduce(
        FragC& frag_C, int splits, int64_t split_size, const int2& cta_cs, Pred& pred, const EpilogueParam<Tc>& param)
    {
        using Vec         = OutputC<Dtype>;
        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        for (int k = 0; k < splits; ++k) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int     ss  = thr_cs.y + s * Map::kDeltaS;
                    const int     cc  = thr_cs.x + c * Map::kDeltaC;
                    const int64_t idx = k * split_size + (cta_cs.y + ss) * param.partial_C_ld + (cta_cs.x + cc);
                    if (true) {
                        Vec tmp;
                        Load(tmp, &param.partial_C[idx]);
                        using namespace ops;
                        frag_C[s][c] = frag_C[s][c] + tmp;
                    }
                }
            }
        }
    }

    template<class FragC, class Pred>
    __device__ void
    Reduce_v2(FragC& frag_C, int split_id, bool is_last, int2 cs0, Pred& pred, const EpilogueParam<Tc>& param)
    {
        constexpr int dc = sizeof(Dtype) * Map::kDeltaC;
        const int     ds = sizeof(Dtype) * Map::kDeltaS * param.partial_C_ld;

        const auto ptr0 = reinterpret_cast<char*>(param.partial_C + cs2idx(cs0, (int64_t)param.partial_C_ld));

        Pred ld_mask = split_id == 0 ? Pred{} : pred;
        Pred st_mask = is_last ? Pred{} : pred;

        auto ptr = ptr0;
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                OutputC<Dtype> tmp{};  // ! ZERO-filled
                if (ld_mask(s, c)) {
                    Load(tmp, reinterpret_cast<Dtype*>(ptr));
                }
                if (1) {
                    using namespace ops;
                    frag_C[s][c] = frag_C[s][c] + tmp;
                }
                if (st_mask(s, c)) {
                    Store(reinterpret_cast<Dtype*>(ptr), frag_C[s][c]);
                }
                ptr += dc;
            }
            ptr -= dc * C;
            ptr += ds;
        }
    }

    template<class FragC>
    __device__ void operator()(FragC&                   frag_C,
                               const int3&              tile_offset,
                               const int3&              tiled_shape,
                               int                      end_m,
                               int                      end_n,
                               bool                     is_last_split,
                               const EpilogueParam<Tc>& param,
                               SharedStorage&           storage)
    {
        const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * M, tile_offset.y * N);
        const int2 end_cs = mk2cs<kOrder>(end_m, end_n);

        OutputC<Dtype> tmp_C[S][C];

        Rearrange(frag_C, storage, tmp_C);

        Predicate<S, C, false, false> pred{};  //  1 regs

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        const int2 cs0    = {cta_cs.x + thr_cs.x, cta_cs.y + thr_cs.y};

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const int ss = thr_cs.y + s * Map::kDeltaS;
                const int cc = thr_cs.x + c * Map::kDeltaC;
                if (ss < end_cs.y && cc < end_cs.x) {
                    pred.set(s, c);
                }
            }
        }

        if (SplitK_ && tiled_shape.z > 1) {
            int* barrier = &param.locks[tile_offset.x * tiled_shape.y + tile_offset.y];

            sem_wait(barrier, tile_offset.z, threadIdx.x == 0);

            Reduce_v2(tmp_C, tile_offset.z, is_last_split, cs0, pred, param);

            const int post_id = is_last_split ? 0 : tile_offset.z + 1;
            sem_post(barrier, post_id, threadIdx.x == 0);

            if (!is_last_split) {
                return;
            }
        }

        constexpr pair<Map::kDeltaC, Map::kDeltaS> delta_cs{};

        // param.combine_chn(tmp_C, cs0, delta_cs, pred);

        param.combine_mat(tmp_C, cs0, delta_cs, pred);

        if (param.silu_act) {
            constexpr int dc  = sizeof(Tc) * Map::kDeltaC / 2;
            const int     ds  = sizeof(Tc) * Map::kDeltaS * param.ldc;
            auto          ptr = reinterpret_cast<char*>(param.C + cs2idx({cs0.x / 2, cs0.y}, (int64_t)param.ldc));
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    GatedActivation<Silu>::apply(tmp_C[s][c]);
                    if (pred(s, c)) {
                        const auto tmp = cast<Tc>((Array<Dtype, kAccess / 2>&)tmp_C[s][c]);
                        Store(reinterpret_cast<Tc*>(ptr), tmp);
                    }
                    ptr += dc;
                }
                ptr -= dc * C;
                ptr += ds;
            }
        }
        else {
            StoreC(tmp_C, param.C, param.ldc, cs0, pred);
        }
    }
};

}  // namespace turbomind::gemm
