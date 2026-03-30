// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/predicate.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

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

template<bool     scale_S,
         bool     scale_C,
         Striding mode_S,
         Striding mode_C,
         class T,
         int N,
         int S,
         int C,
         int delta_C,
         int delta_S,
         class Pred>
__device__ void Scale(pair<scale_S, scale_C>,
                      pair<mode_S, mode_C>,
                      pair<delta_C, delta_S>,
                      Array<T, N> (&x)[S][C],
                      const MatrixParam& param_S,
                      const MatrixParam& param_C,
                      int                gemm_id,
                      int2               cs0,
                      Pred&              pred)
{
    if (scale_S && param_S.ptr) {
        const auto mat = resolve<T, mode_S>(param_S, gemm_id);
        const T*   ptr = (const T*)mat.ptr.ptr;
        T          param[S];
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            const int ss  = cs0.y + s * delta_S;
            const int idx = mat.idxs ? __ldg(mat.idxs + ss) : ss;
            if (pred(s, 0)) {
                param[s] = __ldg((const T*)(ptr + idx));
            }
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                using namespace ops;
                x[s][c] = x[s][c] * param[s];
            }
        }
    }

    if (scale_C && param_C.ptr) {
        const T*      ptr = (const T*)resolve<T, mode_C>(param_C, gemm_id).ptr.ptr + cs0.x;
        constexpr int dc  = sizeof(Array<T, N>) * delta_C;
        Array<T, N>   param[C];
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            if (pred(0, c)) {
                Ldg(param[c], (const T*)(ptr + dc * c));
            }
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                using namespace ops;
                x[s][c] = x[s][c] * param[c];
            }
        }
    }
}

struct MatrixCombination_v3 {

    MatrixParam param_c;
    float       alpha;
    float       beta;

    template<class Tc, Striding mode, class T, int N, int S, int C, int delta_c, int delta_s, class Pred>
    __device__ void operator()(Tc*,  //
                               constant<mode>,
                               Array<T, N> (&x)[S][C],
                               int2 cs0,
                               int  gemm_id,
                               pair<delta_c, delta_s>,
                               Pred& pred) const
    {
        if (beta) {
            const auto c = resolve<Tc, mode>(param_c, gemm_id);

            Array<Tc, N>  frag[S][C];
            constexpr int dc  = sizeof(Tc) * delta_c;
            const int     ds  = sizeof(Tc) * delta_s * c.ptr.stride;
            const char*   ptr = (const char*)c.ptr.ptr + sizeof(Tc) * dot(cs0, long2{1, c.ptr.stride});
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

struct EpilogueParam {
    MatrixParam c;
    MatrixParam partials;
    int*        locks;

    // MatrixParam scale_S;
    // MatrixParam scale_C;

    MatrixCombination_v3 combine_mat;

    bool silu_act;
};

template<class Tc_,
         int M,
         int N,
         int TM_,
         int TN_,
         int THREADS,
         class RearrangeC,
         class OperandC,
         Striding mode_C,
         bool     SplitK_>
struct Epilogue_ {

    using Dtype = typename OperandC::Dtype;

    static constexpr auto kOrder = OperandC::kOrder;
    static constexpr auto kMode  = mode_C;
    static constexpr bool SplitK = SplitK_;

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

    template<class T, class VecC, class Pred>
    __device__ void StoreC(const VecC& vec_C, const MatrixData& c, int2 cs0, Pred& pred)
    {
        constexpr int dc  = sizeof(T) * Map::kDeltaC;
        const int     ds  = sizeof(T) * Map::kDeltaS * c.ptr.stride;
        char*         ptr = (char*)c.ptr.ptr + sizeof(T) * dot(cs0, long2{1, c.ptr.stride});
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

#if 0
    template<class FragC, class Pred>
    __device__ void
    Reduce(FragC& frag_C, int splits, int64_t split_size, const int2& cta_cs, Pred& pred, const EpilogueParam& param)
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
#endif

    template<class FragC, class Pred>
    __device__ void Reduce(FragC& frag_C, const MatrixData& p, bool is_first, bool is_last, int2 cs0, Pred& pred)
    {
        constexpr int dc = sizeof(Dtype) * Map::kDeltaC;
        const int     ds = sizeof(Dtype) * Map::kDeltaS * p.ptr.stride;

        char* ptr = (char*)p.ptr.ptr + sizeof(Dtype) * dot(cs0, long2{1, p.ptr.stride});

        Pred ld_mask = is_first ? Pred{} : pred;
        Pred st_mask = is_last ? Pred{} : pred;

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
    __device__ void operator()(FragC&               frag_C,
                               const int4&          tile_offset,
                               const int2&          extents,
                               int                  splits,
                               int                  tile_id,
                               bool                 is_last,
                               const EpilogueParam& param,
                               SharedStorage&       storage)
    {
        const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * M, tile_offset.y * N);
        const int2 end_cs = mk2cs<kOrder>(extents);

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

        if (SplitK_ && splits > 1) {
            int* barrier = &param.locks[tile_id];

            sem_wait(barrier, tile_offset.z, threadIdx.x == 0);

            const MatrixData p = resolve<Dtype, kMode>(param.partials, tile_offset.w);

            Reduce(tmp_C, p, tile_offset.z == 0, is_last, cs0, pred);

            const int post_id = is_last ? 0 : tile_offset.z + 1;
            sem_post(barrier, post_id, threadIdx.x == 0);

            if (!is_last) {
                return;
            }
        }

        constexpr pair<Map::kDeltaC, Map::kDeltaS> delta_cs{};

        // opt-in scaling
        // Scale(scale_SC{}, mode_SC{}, delta_cs, tmp_C, param.scale_S, param.scale_C, tile_offset.w, cs0, pred);

        param.combine_mat((Tc*)0, constant<kMode>{}, tmp_C, cs0, tile_offset.w, delta_cs, pred);

        const MatrixData c = resolve<Tc, kMode>(param.c, tile_offset.w);

        if (param.silu_act) {
            constexpr int dc  = sizeof(Tc) * Map::kDeltaC / 2;
            const int     ds  = sizeof(Tc) * Map::kDeltaS * c.ptr.stride;
            auto          ptr = (char*)c.ptr.ptr + sizeof(Tc) * dot({cs0.x / 2, cs0.y}, long2{1, c.ptr.stride});
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
            StoreC<Tc>(tmp_C, c, cs0, pred);
        }
    }
};

}  // namespace turbomind::gemm
