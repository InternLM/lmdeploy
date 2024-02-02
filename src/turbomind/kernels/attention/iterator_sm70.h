// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"

namespace turbomind {

template<class T, class Map, class SmemLayout>
struct Sm70GmemIterator: BaseGmemIterator<T, Map, SmemLayout> {
    using Base = BaseGmemIterator<T, Map, SmemLayout>;

    using typename Base::AccessType;
    using typename Base::Fragment;

    using Base::local_offset_;
    using Base::init_offset_;
    using Base::dst_offset_;
    using Base::smem_;

    using Base::Base;

    template<bool is_residue, class BlockIter>
    __device__ void Load(const BlockIter& block_iter, Fragment& rmem, int max_s)
    {
        auto      src = block_iter.block + block_iter.local_id * Map::kDimS * Map::kDimC + local_offset_ + init_offset_;
        const int offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                if constexpr (!is_residue) {
                    Ldg(rmem[s][c], &src[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC]);
                }
                else if (offset_s + s * Map::kDeltaS < max_s) {
                    Ldg(rmem[s][c], &src[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC]);
                }
                else {
                    clear(rmem[s][c]);
                }
            }
        }
    }

    __device__ void Save(const Fragment& rmem)
    {
        // typename SmemLayout::Swizzle swizzle{};

        // Array<int, Map::kIterC> idxs;
        // PRAGMA_UNROLL
        // for (int c = 0; c < Map::kIterC; ++c) {
        //     const int idx0 = swizzle(dst_offset_ + c * Map::kDeltaC);
        //     idxs[c]        = idx0;
        // }
        // const int offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        // PRAGMA_UNROLL
        // for (int s = 0; s < Map::kIterS; ++s) {
        //     PRAGMA_UNROLL
        //     for (int c = 0; c < Map::kIterC; ++c) {
        //         Store(&smem_[idxs[c]], rmem[s][c]);
        //     }
        //     PRAGMA_UNROLL
        //     for (int c = 0; c < Map::kIterC; ++c) {
        //         const int s0 = offset_s + s * Map::kDeltaS;
        //         const int s1 = s0 + Map::kDeltaS;
        //         idxs[c]      = swizzle.AdvanceS<Map::kDeltaS>(idxs[c], s0, s1) + Map::kDeltaS * SmemLayout::kStride;
        //     }
        // }

        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                Store(&smem_[SmemLayout::swizzle(dst_offset_ + s * Map::kDeltaS * SmemLayout::kStride
                                                 + c * Map::kDeltaC)],
                      rmem[s][c]);
            }
        }
    }
};

}  // namespace turbomind