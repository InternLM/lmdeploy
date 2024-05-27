// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"

namespace turbomind::gemm {

struct LDSM_x4_N {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        ldsm_x4(*(Array<uint32_t, 4>*)dst, cast_smem_ptr_to_uint(src));
    }
};

struct LDSM_x4_T {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        ldsm_x4_trans(*(Array<uint32_t, 4>*)dst, cast_smem_ptr_to_uint(src));
    }
};

template<int bytes>
struct LDS {
    template<class S, class D>
    __device__ static void copy(S src, D dst)
    {
        if constexpr (bytes == 1) {
            *(char*)dst = *(const char*)src;
        }
        else if constexpr (bytes == 2) {
            *(short*)dst = *(const short*)src;
        }
        else if constexpr (bytes == 4) {
            *(int1*)dst = *(const int1*)src;
        }
        else if constexpr (bytes == 8) {
            *(int2*)dst = *(const int2*)src;
        }
        else if constexpr (bytes == 16) {
            *(int4*)dst = *(const int4*)src;
        }
        else {
            static_assert(sizeof(S) != sizeof(S), "not implemented");
        }
    }
};

template<class T, bool Trans>
struct SmemCopy_MMA_16816_A {
    static constexpr int S = 16;
    static constexpr int C = 16;

    using Copy = std::conditional_t<Trans, LDSM_x4_T, LDSM_x4_N>;
    using Frag = Array<T, 8>;

    __device__ static int2 get_offset(int lane_id)
    {
        return {
            lane_id / 16 * 8,  // c
            lane_id % 16       // s
        };
    }
};

template<class T, bool Trans>
struct SmemCopy_MMA_16816_B {
    static constexpr int S = 16;
    static constexpr int C = 16;

    using Copy = std::conditional_t<Trans, LDSM_x4_T, LDSM_x4_N>;
    using Frag = Array<T, 8>;

    __device__ static int2 get_offset(int lane_id)
    {
        return {
            lane_id / 8 * 8 % 16,           // c
            lane_id % 8 + lane_id / 16 * 8  // s
        };
    }
};

template<class T>
struct LoadFragment_MMA_16816_Q {  // (M, K)
    static constexpr int C = 8;
    static constexpr int S = 1;

    using Copy = LDS<sizeof(T)>;
    using Frag = Array<T, 2>;

    __device__ static int2 get_offset(int lane_id)
    {
        return {lane_id / 4, 0};
    }

    // __device__ static void apply(const T* src, T* dst)
    // {
    //     PRAGMA_UNROLL
    //     for (int i = 0; i < 2; ++i) {
    //         Lds(*(Array<T, 1>*)dst + i, src + i * 8);
    //     }
    // }
};

template<class Atom_, int S_, int C_>
struct SmemCopy_ {

    using Atom = Atom_;

    static constexpr int S = S_;
    static constexpr int C = C_;

    struct Detail {
        static constexpr int ITER_S   = S / Atom::S;
        static constexpr int ITER_C   = C / Atom::C;
        static constexpr int ITER_CNT = ITER_S * ITER_C;
    };

    using Frag = typename Atom::Frag[Detail::ITER_CNT];

    template<class Accessor>
    __device__ static void copy(Accessor src, Frag& dst, int2 offset_cs)
    {
        static constexpr int DELTA_S = Atom::S;
        static constexpr int DELTA_C = Atom::C;
        const int2           thr_cs  = Atom::get_offset(threadIdx.x % WARP_SIZE);
        PRAGMA_UNROLL
        for (int s = 0; s < Detail::ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Detail::ITER_C; ++c) {
                const int ss = offset_cs.y + thr_cs.y + s * DELTA_S;
                const int cc = offset_cs.x + thr_cs.x + c * DELTA_C;
                Atom::Copy::copy(&src(ss, cc), dst[s * Detail::ITER_C + c].data());
            }
        }
    }
};

template<class T, int S_, int C_, int P_S, int P_C>
struct SmemCopy_Packed {
    //          S                C
    // (CTA_M / Pm / 16m, CTA_K * Pm * 16m)
    // (CTA_K      / 16k, CTA_M      * 16k)

    static constexpr int S = S_;
    static constexpr int C = C_;

    struct Detail {
        static constexpr int ITER_S   = S / 16;
        static constexpr int ITER_C   = C / 16;
        static constexpr int ITER_CNT = ITER_S * ITER_C;
    };

    static constexpr int kFragmentSize = 8 * P_S * P_C;

    using Frag = Array<T, kFragmentSize>[Detail::ITER_CNT];

    template<class Accessor>
    __device__ static void copy(Accessor src, Frag& dst, int2 offset_cs)
    {
        const int lane_id = threadIdx.x / WARP_SIZE;
        PRAGMA_UNROLL
        for (int s = 0; s < Detail::ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Detail::ITER_C; ++c) {
                const int ss = (offset_cs.y + s) / P_S / 16;
                const int cc = (offset_cs.x + c) * P_S * 16 + lane_id * kFragmentSize;
                Lds(dst[s * Detail::ITER_C + c], &src(ss, cc));
            }
        }
    }
};

template<int S_, int C_>
struct VoidSmemCopy {
    static constexpr int S = S_;
    static constexpr int C = C_;

    static constexpr int kFragmentSize = 1;

    using Frag = Array<int, kFragmentSize>[1];

    template<class Accessor>
    __device__ static void copy(Accessor, Frag&, int2)
    {
    }
};

}  // namespace turbomind::gemm