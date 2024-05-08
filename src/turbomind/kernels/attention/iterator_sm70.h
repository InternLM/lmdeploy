// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"

namespace turbomind {

template<class T, class Map, class SmemLayout, int Idx>
struct Sm70GmemIterator: BaseGmemIterator<T, Map, SmemLayout> {
    using Base = BaseGmemIterator<T, Map, SmemLayout>;

    using typename Base::AccessType;
    using typename Base::Fragment;

    using Base::src_offset_;
    using Base::offset_c_;
    using Base::offset_s_;
    using Base::smem_;

    using Base::partial_c_;
    using Base::pred_c_;

    using Base::Base;

    template<bool is_residue, class TileIter>
    __device__ void Load(const TileIter& tile_iter, Fragment& rmem, int max_s)
    {
        auto src_data = tile_iter.OffsetPtr<Idx>(src_offset_);
        int  offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                copy(Array<T, Map::kAccessC>{}, rmem[s][c]);
                auto src = &src_data[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC];
                if constexpr (partial_c_) {  // Only quant params is partial C
                    if (pred_c_) {
                        Ldg(rmem[s][c], src);
                    }
                }
                else if (!is_residue || offset_s + s * Map::kDeltaS < max_s) {
                    Ldg(rmem[s][c], src);
                }
            }
        }
    }

    __device__ void Save(const Fragment& rmem)
    {
        typename SmemLayout::Swizzle swizzle{};

        SmemAccessor<T, SmemLayout> data{smem_};
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                if (!partial_c_ || pred_c_) {
                    Store(&data(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC), rmem[s][c]);
                }
            }
        }
    }
};

}  // namespace turbomind
