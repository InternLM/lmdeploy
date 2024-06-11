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

    static constexpr int S = 1;
    static constexpr int C = 1;

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

template<class T, int S_, int C_, int FragSize, int FragNum, int RepeatC = 1>
struct SmemCopyAtom_Pack_v2 {
    static constexpr int S = S_;
    static constexpr int C = C_;

    using Frag = Array<T, FragSize * FragNum>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {lane_id / RepeatC * Frag::size(), 0};
    }

    template<class S, class D>
    __device__ static void copy(S src_ptr, D dst_ptr, bool mask)
    {
        if (mask) {
            Lds(*(Frag*)dst_ptr, src_ptr);
        }
    }
};

template<class Operand, int M, int K, int dM, int dK>
struct SmemCopy {
    using Atom = typename Operand::SmemCopyAtom;

    static constexpr int2 kShape  = mk2cs<Operand::kOrder>(M, K);
    static constexpr int2 kDelta  = mk2cs<Operand::kOrder>(dM, dK);
    static constexpr int2 kRepeat = mk2cs<Operand::kOrder>(1, Operand::kGroupSize);

    static constexpr int ITER_C = kShape.x / Atom::C;
    static constexpr int ITER_S = kShape.y / Atom::S;

    using Frag = typename Atom::Frag[ITER_S * ITER_C];

    using Pack = Packing<Operand::kPack>;

    template<class Pointer>
    __device__ static void copy(Pointer src_ptr, Frag& dst, int2 offset_mk, bool mask = true)
    {
        typename Operand::SmemAccessor smem{src_ptr};

        const int2 thr    = Atom::get_offset(threadIdx.x);
        const int2 offset = mk2cs<Operand::kOrder>(offset_mk.x, offset_mk.y);

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int  cc = (offset.x + c * kDelta.x) / kRepeat.x;
                const int  ss = (offset.y + s * kDelta.y) / kRepeat.y;
                const int2 cs = Pack::apply(int2{cc, ss});
                Atom::copy(&smem(cs.y + thr.y, cs.x + thr.x), dst[s * ITER_C + c].data(), mask);
            }
        }
    }
};

}  // namespace turbomind::gemm