// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"

namespace turbomind::gemm {

template<class Tin, class Tout>
struct Converter {};

template<class T>
struct Converter<T, T> {
    template<int N>
    __device__ Array<T, N> operator()(Array<T, N> x)
    {
        return x;
    }
};

template<>
struct Converter<uint16_t, uint4_t> {

    static __device__ Array<uint4_t, 8> pack(const Array<uint8_t, 8>& vi)
    {
        Array<uint32_t, 2> ui = (Array<uint32_t, 2>&)vi;

        ui[0] |= (ui[0] >> 12);
        ui[1] |= (ui[1] >> 12);

        //  7 6 5 4 3 2 1 0
        // _7_67564_3_23120
        uint32_t uo = __byte_perm(ui[0], ui[1], 0x5140);

        return (Array<uint4_t, 8>&)uo;
    }

    template<class U, int N>
    __device__ Array<uint4_t, N> operator()(const Array<U, N>& x)
    {
        static_assert(sizeof(U) == 2);
        auto&             vi = (const Array<uint16_t, N>&)x;
        Array<uint8_t, N> tmp;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            tmp[i] = static_cast<uint8_t>(vi[i]);
        }
        Array<uint4_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            (Array<uint4_t, 8>&)vo[i] = pack((Array<uint8_t, 8>&)tmp[i]);
        }
        return vo;
    }
};

template<>
struct Converter<uint16_t, uint8_t> {
    template<class U, int N>
    __device__ Array<uint8_t, N> operator()(const Array<U, N>& x)
    {
        static_assert(sizeof(U) == 2);
        auto&             vi = (const Array<uint16_t, N>&)x;
        Array<uint8_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            vo[i] = static_cast<uint8_t>(vi[i]);
        }
        return vo;
    }
};

}  // namespace turbomind::gemm