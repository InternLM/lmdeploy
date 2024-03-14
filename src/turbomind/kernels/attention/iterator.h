// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"
#include "smem_layout.h"
#include "src/turbomind/kernels/attention/data_type.h"
#include <type_traits>

namespace turbomind {

template<class T, class Map, class SmemLayout>
struct BaseGmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, Map::kAccessC>;
    using Pointer     = get_pointer_type<T>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);
    static constexpr int kIterCount   = Map::kIterS * Map::kIterC;

    using Fragment = Array<T, Map::kAccessC>[Map::kIterS][Map::kIterC];

    Pointer smem_;

    int src_offset_;
    int offset_c_;
    int offset_s_;

    static constexpr std::integral_constant<bool, Map::kPartialC> partial_c_{};

    std::conditional_t<partial_c_, bool, std::true_type> pred_c_;

    __device__ BaseGmemIterator()
    {
        int  warp_id = threadIdx.x / WARP_SIZE;
        int  lane_id = threadIdx.x % WARP_SIZE;
        int2 offsets = Map::get_offset(warp_id, lane_id);
        src_offset_  = offsets.x + offsets.y * Map::kDimC;
        offset_c_    = offsets.x;
        offset_s_    = offsets.y;
        if constexpr (partial_c_) {
            pred_c_ = offset_c_ < Map::kDimC;
        }
    }

    __device__ void SetSmem(Pointer smem)
    {
        smem_ = smem;
    }

    __device__ void ClearSmem(int pipe_iter = 0)
    {
        SmemAccessor<T, SmemLayout> data{smem_};
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                if (pred_c_) {
                    Store(&data(offset_s_ + s * Map::kDeltaS,
                                offset_c_ + c * Map::kDeltaC,
                                pipe_iter * SmemLayout::kSize),
                          Array<T, Map::kAccessC>{});
                }
            }
        }
    }
};

template<class T, class Layout>
struct BaseSmemIterator {
    static constexpr int kElemSize = sizeof(T);

    using Accessor = SmemAccessor<T, Layout>;
    T* smem_;

    __device__ explicit BaseSmemIterator(T* smem): smem_{smem} {}
};

template<class Iterator0, class Iterator1>
struct CombinedIterator {
    Iterator0 iterator0_;
    Iterator1 iterator1_;

    // NOTE: can't use reference type here, nvcc does not support variadic templates well in device code
    template<typename... Args>
    __device__ void Prefetch(Args... args)
    {
        iterator0_.Prefetch(args...);
        iterator1_.Prefetch(args...);
    }

    // template<class Partial, class TileIter>
    // __device__ void
    // Prefetch(Partial partial, const TileIter& tile_iter, int s_begin, int s_count, int max_s, int pipe_iter)
    // {
    //     iterator0_.Prefetch(partial, tile_iter, s_begin, s_count, max_s, pipe_iter);
    //     iterator1_.Prefetch(partial, tile_iter, s_begin, s_count, max_s, pipe_iter);
    // }

    // template<class Partial, class TileIter>
    // __device__ void Prefetch(Partial partial, const TileIter& tile_iter, int max_s, int pipe_iter)
    // {
    //     iterator0_.Prefetch(partial, tile_iter, max_s, pipe_iter);
    //     iterator1_.Prefetch(partial, tile_iter, max_s, pipe_iter);
    // }

    __device__ void ClearSmem(int pipe_iter = 0)
    {
        iterator0_.ClearSmem(pipe_iter);
        iterator1_.ClearSmem(pipe_iter);
    }

    template<class P0, class P1>
    __device__ void SetSmem(P0 p0, P1 p1)
    {
        iterator0_.SetSmem(p0);
        iterator1_.SetSmem(p1);
    }
};

template<int Stages, int Step = 1>
struct PipeIter {
    static constexpr int kMaxStep = Stages * Step;

    int r = 0;
    int w = kMaxStep - Step;

    __inline__ __device__ PipeIter& operator++()
    {
        w = r;
        r += Step;
        if (r == kMaxStep) {
            r -= kMaxStep;
        }
        return *this;
    }
};

}  // namespace turbomind
