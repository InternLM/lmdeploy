// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

struct VoidSmemCopyAtom {

    static constexpr int M = 1;
    static constexpr int K = 1;

    static constexpr int kFragNum = 1;

    using Frag = Array<int, 1>;

    template<class S, class D>
    __device__ static void copy(S, D, bool)
    {
    }

    __device__ static int2 get_offset(int)
    {
        return {};
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        return {};
    }
};

template<class T, class Layout, Order order>
struct SmemAccessorV2 {
};

template<class T, class Layout>
struct SmemAccessorV2<T, Layout, kRowMajor>: SmemAccessor<T, Layout> {
    using SmemAccessor<T, Layout>::SmemAccessor;
};

template<class T, class Layout>
struct SmemAccessorV2<T, Layout, kColMajor> {
    SmemAccessor<T, Layout> base_;

    __device__ SmemAccessorV2(get_pointer_type<T> ptr): base_{ptr} {}
    __device__ T& operator()(int m, int k)
    {
        return base_(k, m);
    }
};

template<class T, Order order, int M_, int K_, int FragSize, int FragNum_, int RepeatC = 1>
struct SmemCopyAtom_Pack_v2 {
    static constexpr int M = M_;
    static constexpr int K = K_;

    static constexpr int kFragNum = FragNum_;

    using Frag = Array<T, FragSize * kFragNum>;

    __device__ static int2 get_offset(int thread_idx)  // -> (m, k)
    {
        const int lane_id = thread_idx % WARP_SIZE;

        const int c = lane_id / RepeatC * Frag::size();

        return order == kRowMajor ? int2{0, c} : int2{c, 0};
    }

    template<class S, class D>
    __device__ static void copy(S src_ptr, D dst_ptr, bool mask)
    {
        auto dst_raw_ptr = (T*)dst_ptr;  // SubBytePtr<T> -> T*
        if (mask) {
            Lds(*(Frag*)dst_raw_ptr, src_ptr);
        }
    }
};

template<class T, class CopyAtom, Order order, int FragNum_>
struct SmemCopyAtom_Pack_v3 {
    static constexpr int M = CopyAtom::M * FragNum_;
    static constexpr int K = CopyAtom::K;

    static constexpr int kFragNum = FragNum_;

    using Frag = Array<T, CopyAtom::Frag::size() * kFragNum>;

    __device__ static int2 get_offset(int thread_idx)  // -> (m, k)
    {
        const int c = CopyAtom::unique(thread_idx, 0).x * Frag::size();

        return order == kRowMajor ? int2{0, c} : int2{c, 0};
    }

    template<class S, class D>
    __device__ static void copy(S src_ptr, D dst_ptr, bool mask)
    {
        if (mask) {
            auto dst_raw_ptr = (T*)dst_ptr;  // SubBytePtr<T> -> T*
            Lds(*(Frag*)dst_raw_ptr, src_ptr);
        }
    }
};

template<class Operand, int iM, int iK, int dM, int dK>
struct SmemCopy {
    using Atom = typename Operand::SmemCopyAtom;

    static constexpr int kFragNum = Atom::kFragNum;

    static constexpr int ITER_M = iM / Atom::kFragNum;

    static_assert(ITER_M > 0);

    using Frag = typename Atom::Frag[ITER_M];

    using Pack = Packing_v2<Operand::kPack, Operand::kOrder>;

    static constexpr int2 delta = Pack::apply(int2{dM * kFragNum, dK});

    using Layout = typename Operand::SmemLayout;

    static constexpr int2 kMK0 = cs2mk<Operand::kOrder>(Layout::C0, Layout::S0);

    static constexpr int kPeriodM = ceil_div(kMK0.x, delta.x);
    static constexpr int kPeriodK = ceil_div(kMK0.y, delta.y);

    const int2 offset_;

    int phases_[kPeriodK][kPeriodM];

    __device__ SmemCopy(int2 offset): offset_{offset}
    {
        const int2 thr = Atom::get_offset(threadIdx.x);
        PRAGMA_UNROLL
        for (int k = 0; k < kPeriodK; ++k) {
            PRAGMA_UNROLL
            for (int m = 0; m < kPeriodM; ++m) {
                const int2 pack = Pack::apply({offset.x + m * dM * kFragNum, offset.y + k * dK});
                const int2 cs   = mk2cs<Operand::kOrder>({pack.x + thr.x, pack.y + thr.y});
                phases_[k][m]   = Layout::apply(cs.y, cs.x);
            }
        }
    }

    template<class Pointer>
    __device__ void operator()(Pointer src_ptr, Frag& dst, int k, bool mask = true)
    {
        using Accessor = typename Operand::SmemAccessor;
        if constexpr (Operand::kGroupSize == 1) {
            PRAGMA_UNROLL
            for (int m = 0; m < ITER_M; ++m) {
                const int  mm = m / kPeriodM * kPeriodM * dM * kFragNum;
                const int  kk = k / kPeriodK * kPeriodK * dK;
                const int2 cs = mk2cs<Operand::kOrder>(Pack::apply(int2{mm, kk}));
                const int  i0 = Layout::apply(cs.y, cs.x);
                const int  i1 = phases_[k % kPeriodK][m % kPeriodM];
                Atom::copy(&src_ptr[i0 + i1], dst[m].data(), mask);
            }
        }
        else {  // generic case
            Accessor   smem{src_ptr};
            const int2 thr = Atom::get_offset(threadIdx.x);
            PRAGMA_UNROLL
            for (int m = 0; m < ITER_M; ++m) {
                const int  mm = offset_.x + m * dM * kFragNum;
                const int  kk = offset_.y + k * dK;  // Note: this forbids sub-tile group sizes
                const int2 mk = Pack::apply(int2{mm, kk / Operand::kGroupSize});
                Atom::copy(&smem(mk.x + thr.x, mk.y + thr.y), dst[m].data(), mask);
            }
        }
        // else if constexpr (Operand::kPack != 0 && Operand::kGroupSize != 1) {  // group size = 1, pack != 0
        //     const int  mask_k = Operand::kGroupSize == 1;
        //     const int2 pack   = Pack::apply(int2{offset_.x, offset_.y});
        //     const int2 thr    = Atom::get_offset(threadIdx.x);
        //     const int2 cs     = mk2cs<Operand::kOrder>({pack.x + thr.x, (pack.y + thr.y) * mask_k});
        //     auto       smem   = src_ptr + Layout::apply(cs.y, cs.x);
        //     PRAGMA_UNROLL
        //     for (int m = 0; m < ITER_M; ++m) {
        //         const int  mm  = m * dM * kFragNum;
        //         const int  kk  = k * dK;
        //         const int2 cs  = mk2cs<Operand::kOrder>(Pack::apply(int2{mm, kk * mask_k}));
        //         const int  idx = Layout::apply(cs.y, cs.x);
        //         Atom::copy(&smem[idx], dst[m].data(), mask);
        //     }
        // }
    }
};

}  // namespace turbomind::gemm
