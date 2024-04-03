// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
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

    struct Fragment {
        typename Iterator0::Fragment frag0;
        typename Iterator1::Fragment frag1;
    };

    // NOTE: can't use reference type here, nvcc does not support variadic templates well in device code
    template<typename... Args>
    __device__ void Prefetch(Args... args)
    {
        iterator0_.Prefetch(args...);
        iterator1_.Prefetch(args...);
    }

    /// TODO: Load(bool_constant, CacheIter&) -> Fragment
    template<bool is_residue, class CacheIter>
    __device__ void Load(const CacheIter& cache_iter, Fragment& frag, int max_s)
    {
        iterator0_.Load<is_residue>(cache_iter, frag.frag0, max_s);
        iterator1_.Load<is_residue>(cache_iter, frag.frag1, max_s);
    }

    __device__ void Save(const Fragment& frag)
    {
        iterator0_.Save(frag.frag0);
        iterator1_.Save(frag.frag1);
    }

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

}  // namespace turbomind
