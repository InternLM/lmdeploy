// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"

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

        auto& frag_k = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
        auto& stat_k = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
        auto& data_k = data[k];

        PRAGMA_UNROLL
        for (int m = 0; m < Md; ++m) {
            auto tmp = ConvertKvCache<D, F>::convert(data_k[m]);
            static_assert(Nd % 8 == 0);
            PRAGMA_UNROLL
            for (int i = 0; i < Nd; i += 8) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < 2; ++c) {
                        const int idx = (m * Nd + i) / 8 * 2 + s * StatStepS + c * StatStepC;
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
        x[0]            = __hfma(x[0], _s[0], _s[1]);
        x[1]            = __hfma(x[1], _s[0], _s[1]);
    }

    __device__ static void dequant(Array<bfloat16_t, 2>& x, Array<uint8_t, 1> s)
    {
        bfloat16_t s1 = __ushort_as_bfloat16((uint16_t)s[0] << 7);
        x[0]          = __hmul(x[0], s1);
        x[1]          = __hmul(x[1], s1);
    }

    __device__ static void dequant(Array<half_t, 2>& x, Array<uint8_t, 1> s)
    {
        // half_t s1 = __ushort_as_half(((uint16_t)s[0] + 15 - 127) << 10);
        // Adjusted in `AdjustUe8m0ScaleForHalf`
        half_t s1 = __ushort_as_half((uint16_t)s[0] << 10);
        x[0]      = __hmul(x[0], s1);
        x[1]      = __hmul(x[1], s1);
    }

    __device__ static void dequant(Array<bfloat16_t, 2>& x, Array<uint16_t, 1> s)
    {
        auto s1 = __ushort_as_bfloat16(s[0]);
        x[0]    = __hmul(x[0], s1);
        x[1]    = __hmul(x[1], s1);
    }

    __device__ static void dequant(Array<half, 2>& x, Array<uint16_t, 1> s)
    {
        auto s1 = __ushort_as_half(s[0]);
        x[0]    = __hmul(x[0], s1);
        x[1]    = __hmul(x[1], s1);
    }
};

// Used by SM70 MMA
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

    __device__ static void dequant(Array<half_t, 2>& x, Array<uint8_t, 1> s)
    {
        // half_t s1 = __ushort_as_half(((uint16_t)s[0] + 15 - 127) << 10);
        // Adjusted in `AdjustUe8m0ScaleForHalf`
        half_t s1 = __ushort_as_half((uint16_t)s[0] << 10);
        x[0]      = __hmul(x[0], s1);
        x[1]      = __hmul(x[1], s1);
    }

    __device__ static void dequant(Array<half, 2>& x, Array<uint16_t, 1> s)
    {
        auto s1 = __ushort_as_half(s[0]);
        x[0]    = __hmul(x[0], s1);
        x[1]    = __hmul(x[1], s1);
    }
};

}  // namespace turbomind::gemm
