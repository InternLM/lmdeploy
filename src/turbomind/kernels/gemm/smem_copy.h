// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

struct VoidSmemCopyAtom {

    static constexpr int M = 1;
    static constexpr int K = 1;

    using Frag = Array<int, 1>;

    template<class S, class D>
    __device__ static void copy(S, D, bool)
    {
    }

    __device__ static int2 get_offset(int)
    {
        return {};
    }
};

template<class T, class Layout, Order order>
struct SmemAccessorV2 {};

template<class T, class Layout>
struct SmemAccessorV2<T, Layout, kRowMajor>: SmemAccessor<T, Layout> {
    using SmemAccessor<T, Layout>::SmemAccessor;
};

template<class T, class Layout>
struct SmemAccessorV2<T, Layout, kColMajor> {
    SmemAccessor<T, Layout> base_;

    __device__    SmemAccessorV2(get_pointer_type<T> ptr): base_{ptr} {}
    __device__ T& operator()(int m, int k)
    {
        return base_(k, m);
    }
};

template<class T, Order order, int M_, int K_, int FragSize, int FragNum, int RepeatC = 1>
struct SmemCopyAtom_Pack_v2 {
    static constexpr int M = M_;
    static constexpr int K = K_;

    using Frag = Array<T, FragSize * FragNum>;

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

template<class Operand, int M, int K, int dM, int dK>
struct SmemCopy {
    using Atom = typename Operand::SmemCopyAtom;

    static constexpr int ITER_M = M / Atom::M;

    using Frag = typename Atom::Frag[ITER_M];
    using Pack = Packing_v2<Operand::kPack, Operand::kOrder>;

    template<class Pointer>
    __device__ static void copy(Pointer src_ptr, Frag& dst, int2 offset, bool mask = true)
    {
        typename Operand::SmemAccessor smem{src_ptr};

        const int2 thr = Atom::get_offset(threadIdx.x);

        // Note: this forbids sub-tile group sizes
        const int kk = offset.y / Operand::kGroupSize;

        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            const int  mm = offset.x + m * dM;
            const int2 mk = Pack::apply(int2{mm, kk});
            Atom::copy(&smem(mk.x + thr.x, mk.y + thr.y), dst[m].data(), mask);
        }
    }
};

}  // namespace turbomind::gemm