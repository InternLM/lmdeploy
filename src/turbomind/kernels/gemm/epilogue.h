// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

template<class Tc>
struct ChannelCombination {
    const float* __restrict__ scale_ptr;
    const float* __restrict__ bias_ptr;

    __device__ bool is_valid() const noexcept
    {
        return scale_ptr || bias_ptr;
    }

    template<class T, int N>
    __device__ void operator()(Array<T, N>& x, int m, int n)
    {
        using namespace ops;

        float scale = 1.f;
        float bias  = 0.f;

        if (scale_ptr) {
            scale = __ldg(scale_ptr + n);
        }

        if (bias_ptr) {
            bias = __ldg(bias_ptr + n);
        }

        // FMA
        x = x * static_cast<T>(scale) + static_cast<T>(bias);
    }
};

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

    template<class T, int V, int S, int C>
    __device__ void operator()(Array<T, V> (&x)[S][C], int2 cta_cs, int2 thr_cs, int2 delta_cs, int2 end_cs) const
    {
        __align__(16) Array<Tc, 2> scale_bias[S];

        if (scale_bias_ptr) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                const int ss = thr_cs.y + s * delta_cs.y;
                if (ss < end_cs.y) {
                    Ldg(scale_bias[s], scale_bias_ptr + ss + cta_cs.y);
                }
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

    template<class T, int N, int S, int C>
    __device__ void operator()(Array<T, N> (&x)[S][C], int2 cta_cs, int2 thr_cs, int2 delta_cs, int2 end_cs) const
    {
        Array<Tc, N> frag[S][C]{};
        if (beta) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int     ss    = thr_cs.y + s * delta_cs.y;
                    const int     cc    = thr_cs.x + c * delta_cs.x;
                    const int64_t index = (cta_cs.y + ss) * ldc + (cta_cs.x + cc);
                    const bool    mask  = cc < end_cs.x && ss < end_cs.y;
                    if (mask) {
                        Load(frag[s][c], &C_ptr[index]);
                    }
                }
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
struct MatrixCombination {
    float alpha;
    float beta;

    const Tc* C;  // can't `__restrict__` since it may be alias of `D`
    int64_t   ldc;

    __device__ bool is_valid() const noexcept
    {
        return true;
    }

    template<class T, int N>
    __device__ void operator()(Array<T, N>& x, int m, int n)
    {
        using namespace ops;

        Array<Tc, N> c{};

        if (beta) {
            Load(c, &C[m * ldc + n]);
        }

        // FMA
        x = x * static_cast<T>(alpha) + cast<T>(c);
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

template<class Tc_, int M, int N, int PM, int PN, int THREADS, class RearrangeC, class OperandC, bool SplitK_>
struct Epilogue_ {

    using SmemLayout = decltype(OperandC::GetSmemLayout::apply(pair<M, N>{}));
    using Map        = decltype(OperandC::GetThreadMap::apply(pair<M, N>{}, constant<THREADS>{}));

    using Dtype = typename OperandC::Dtype;

    using Tc = Tc_;

    static constexpr auto kOrder = OperandC::kOrder;
    static constexpr auto SplitK = SplitK_;

    using SmemAccessorV2 = SmemAccessorV2<Dtype, SmemLayout, kOrder>;

    using SharedStorage = Array<Dtype, SmemLayout::kSize>;

    static constexpr int S       = Map::kIterS;
    static constexpr int C       = Map::kIterC;
    static constexpr int kAccess = Map::kAccessC;

    struct Param {
        int m;
        int n;
        Tc* C;
        int ldc;

        float* partial_C;
        int    partial_C_ld;

        int* locks;  // (m/cta_m, n/cta_n, k)

        ChannelCombination_v3<Tc> combine_chn;
        MatrixCombination_v2<Tc>  combine_mat;
        bool                      silu_act;
    };

    template<class T>
    using OutputC = Array<T, kAccess>;

    template<class FragC>
    __device__ void Rearrange(FragC& frag_C, SharedStorage& storage, OutputC<Dtype> (&out)[S][C])
    {
        SmemAccessorV2 smem_C{(float*)__cvta_shared_to_generic(0)};

        // SmemAccessorV2 smem_C{0};

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

        PRAGMA_UNROLL
        for (int m = 0; m < M; m += PM) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += PN) {
                // Store to shared memory
                RearrangeC::apply(frag_C, smem_C, {m, n}, pair<PM, PN>{});

                // Load from shared memory
                PRAGMA_UNROLL
                for (int s = 0; s < S; ++s) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < C; ++c) {
                        const int  cc = c * Map::kDeltaC + thr_cs.x;
                        const int  ss = s * Map::kDeltaS + thr_cs.y;
                        const int2 mn = cs2mk<kOrder>(cc, ss);
                        const int  mm = mn.x - m;
                        const int  nn = mn.y - n;
                        // const int mm = mn.x;
                        // const int nn = mn.y;
                        // printf("%d %d\n", mm, nn);
                        const bool mask = (M <= PM || (0 <= mm && mm < PM)) && ((N <= PN) || (0 <= nn && nn < PN));
                        if (mask) {
                            Load(out[s][c], &smem_C(mm, nn));
                            // for (const auto& x : out[s][c]) {
                            //     if (mm < 16 && nn < 32) {
                            //         printf("%d %f %d %d\n", (int)threadIdx.x, x, mm, nn);
                            //     }
                            // }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    template<class VecC, class T>
    __device__ void
    StoreC(const VecC& vec_C, T* data_C, int64_t ldc, const int2& cta_cs, const int2& end_cs, bool force = false)
    {
        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const int     ss    = thr_cs.y + s * Map::kDeltaS;
                const int     cc    = thr_cs.x + c * Map::kDeltaC;
                const int64_t index = (cta_cs.y + ss) * ldc + (cta_cs.x + cc);
                const bool    mask  = cc < end_cs.x && ss < end_cs.y;
                const auto    tmp   = cast<T>(vec_C[s][c]);
                if (force || mask) {
                    Store(&data_C[index], tmp);
                }
            }
        }
    }

    template<class FragC>
    __device__ void
    Reduce(FragC& frag_C, int splits, int64_t split_size, const int2& cta_cs, const int2& end_cs, const Param& param)
    {
        using Vec         = OutputC<Dtype>;
        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

        for (int k = 0; k < splits; ++k) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                // Vec tmp_C[1][C]{};

                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int     ss   = thr_cs.y + s * Map::kDeltaS;
                    const int     cc   = thr_cs.x + c * Map::kDeltaC;
                    const int64_t idx  = k * split_size + (cta_cs.y + ss) * param.partial_C_ld + (cta_cs.x + cc);
                    const bool    mask = true;  // cc < end_cs.x && ss < end_cs.y;
                    if (mask) {
                        Vec tmp;
                        Load(tmp, &param.partial_C[idx]);
                        using namespace ops;
                        frag_C[s][c] = frag_C[s][c] + tmp;
                    }
                }

                // PRAGMA_UNROLL
                // for (int c = 0; c < C; ++c) {
                //     using namespace ops;
                //     frag_C[s][c] = frag_C[s][c] + tmp_C[0][c];
                // }
            }
        }
    }

    template<class FragC>
    __device__ void operator()(FragC&         frag_C,
                               const int3&    tile_offset,
                               const int3&    tiled_shape,
                               int            end_m,
                               int            end_n,
                               bool           is_primary_cta,
                               const Param&   param,
                               SharedStorage& storage)
    {
        const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * M, tile_offset.y * N);
        const int2 end_cs = mk2cs<kOrder>(end_m, end_n);

        OutputC<Dtype> tmp_C[S][C];

        Rearrange(frag_C, storage, tmp_C);

        if (SplitK_ && tiled_shape.z > 1) {

            // if (!is_primary_cta) {
            //     return;
            // }

            int*       locks      = &param.locks[(tile_offset.x * tiled_shape.y + tile_offset.y) * tiled_shape.z];
            const auto split_size = cs2idx(mk2cs<kOrder>(param.m, param.n), (int64_t)param.partial_C_ld);
            const int  thread_idx = threadIdx.x;

            if (!is_primary_cta) {
                auto partial = param.partial_C + tile_offset.z * split_size;
                StoreC(tmp_C, partial, param.partial_C_ld, cta_cs, end_cs, true);
                sem_post(&locks[tile_offset.z], 1, thread_idx == 0);
                return;
            }

            // Wait for other splits
            sem_wait_many(&locks[thread_idx], tile_offset.z, thread_idx < tile_offset.z);

            Reduce(tmp_C, tile_offset.z, split_size, cta_cs, end_cs, param);

            if (thread_idx <= tile_offset.z) {
                locks[thread_idx] = 0;
            }
        }

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

        constexpr int2 delta_cs{Map::kDeltaC, Map::kDeltaS};

        param.combine_chn(tmp_C, cta_cs, thr_cs, delta_cs, end_cs);
        param.combine_mat(tmp_C, cta_cs, thr_cs, delta_cs, end_cs);

        if (0 && param.silu_act) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    GatedActivation<Silu>::apply(tmp_C[s][c]);
                    const int     ss   = thr_cs.y + s * Map::kDeltaS;
                    const int     cc   = thr_cs.x + c * Map::kDeltaC;
                    const int64_t idx  = (cta_cs.y + ss) * param.ldc + (cta_cs.x + cc) / 2;
                    const bool    mask = cc < end_cs.x && ss < end_cs.y;
                    const auto    tmp  = cast<Tc>((Array<Dtype, kAccess / 2>&)tmp_C[s][c]);
                    if (mask) {
                        Store(&param.C[idx], tmp);
                    }
                }
            }
        }
        else {
            StoreC(tmp_C, param.C, param.ldc, cta_cs, end_cs);
        }
    }
};

}  // namespace turbomind::gemm