// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include <iterator>

namespace turbomind::gemm {

struct Transform_Default {
    template<class T, int Nf, int Mf, int K, int Nd, int Md, class S>
    __device__ static void apply(Array<T, Nf> (&frag)[K][Mf], int k, Array<T, Nd> (&data)[K][Md], S&, int div)
    {
        static_assert(Nf * Mf == Nd * Md);
        static_assert(Nd % Nf == 0 && Mf % Md == 0);
        static_assert(sizeof(frag) == sizeof(data));

        // Alignment must be manually enforced for `reinterpret_cast`
        auto& frag_k = reinterpret_cast<Array<T, Nd>(&)[Md]>(frag[k]);
        auto& data_k = data[k];

        PRAGMA_UNROLL
        for (int i = 0; i < std::size(frag_k); ++i) {
            frag_k[i] = data_k[i];
        }
    }
};

template<int StatStepS, int StatStepC>
struct Transform_HMMA_16816 {
    template<class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S, int Ns, int Ms, int Ks>
    __device__ static void
    apply(Array<F, Nf> (&frag)[K][Mf], int k, Array<D, Nd> (&data)[K][Md], Array<S, Ns> (&stat)[Ks][Ms], int div)
    {
        static_assert(Nf * Mf == Nd * Md);
        static_assert(Nd % Nf == 0 && Mf % Md == 0);
        static_assert(Nf * Mf == Ns * Ms * 4);

        // static_assert(Nf != Nf);

        auto& frag_k = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
        auto& stat_k = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
        auto& data_k = data[k];

        PRAGMA_UNROLL
        for (int m = 0; m < Md; ++m) {
            // if (threadIdx.x == 0) {
            //     printf("m = %d\n", m);
            // }
            auto tmp = ConvertKvCache<D, F>::convert(data_k[m]);
            PRAGMA_UNROLL
            for (int i = 0; i < Nd; i += 8) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < 2; ++c) {
                        const int idx = (m * Nd + i) / 8 * 2 + s * StatStepS + c * StatStepC;
                        // if (threadIdx.x == 0) {
                        //     printf("idx=%d\n", idx);
                        // }
                        dequant((Array<F, 2>&)tmp[i + s * 4 + c * 2], stat_k[idx]);
                    }
                }
            }

            frag_k[m] = tmp;
        }
    }

    template<class F>
    __device__ static void dequant(Array<F, 2>& x, Array<uint32_t, 1> s)
    {
        Array<F, 2>& _s = (Array<F, 2>&)s;
        // printf("tidx=%d %f %f\n", (int)threadIdx.x, (float)_s[0], (float)_s[1]);
        // printf("tidx=%d %f %f\n", (int)threadIdx.x, (float)x[0], (float)x[1]);
        x[0] = __hfma(x[0], _s[0], _s[1]);
        x[1] = __hfma(x[1], _s[0], _s[1]);
    }
};

struct Transform_HMMA_SIMT_B {
    template<class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S, int Ns, int Ms, int Ks>
    __device__ static void
    apply(Array<F, Nf> (&frag)[K][Mf], int k, Array<D, Nd> (&data)[K][Md], Array<S, Ns> (&stat)[Ks][Ms], int div)
    {
        static_assert(Nf * Mf == Nd * Md);
        static_assert(Nd % Nf == 0 && Mf % Md == 0);

        auto& frag_k = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
        auto& stat_k = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
        auto& data_k = data[k];

        // static_assert(Nf != Nf);

        PRAGMA_UNROLL
        for (int m = 0; m < Md; ++m) {
            auto tmp = ConvertKvCache<D, F>::convert(data_k[m]);
            PRAGMA_UNROLL
            for (int i = 0; i < Nd; i += 2) {
                dequant((Array<F, 2>&)tmp[i], stat_k[(m * Nd + i) / Nf]);
            }
            frag_k[m] = tmp;
        }
    }

    template<class F>
    __device__ static void dequant(Array<F, 2>& x, Array<uint32_t, 1> s)
    {
        Array<F, 2>& _s = (Array<F, 2>&)s;

        x[0] = __hfma(x[0], _s[0], _s[1]);
        x[1] = __hfma(x[1], _s[0], _s[1]);
    }
};

}  // namespace turbomind::gemm
