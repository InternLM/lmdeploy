// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"
#include "smem_layout.h"
#include <type_traits>

namespace turbomind {

template<class T, class Map, class SmemLayout>
struct BaseGmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, Map::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);
    static constexpr int kIterCount   = Map::kIterS * Map::kIterC;

    using Fragment = Array<T, Map::kAccessC>[Map::kIterS][Map::kIterC];

    T* smem_;

    int src_offset_;
    int offset_c_;
    int offset_s_;

    __device__ BaseGmemIterator(int warp_id, int lane_id)
    {
        int2 offsets = Map::get_offset(warp_id, lane_id);
        src_offset_  = offsets.x + offsets.y * Map::kDimC;
        offset_c_    = offsets.x;
        offset_s_    = offsets.y;
    }

    __device__ void SetSmem(T* smem)
    {
        smem_ = smem;
    }

    template<class Offset>
    __device__ void ClearSmem(Offset offset)
    {
        SmemAccessor<T, SmemLayout> data{smem_};
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                Store(&data(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC, offset),
                      Array<T, Map::kAccessC>{});
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

template<class T, int CTA_S, int HeadDim, class BlockSeqLen>
struct BlockTileIter {

    const int tiles_per_block_;
    const T** block_ptrs_;

    int block_id_;

    const T* block;
    int      local_id;

    Array<int, 2> kv_offset_;

    int local_id_offset_;

    __device__
    BlockTileIter(const T** block_ptrs, BlockSeqLen block_seqlen, Array<int, 2> kv_offset, int local_id_offset):
        block_ptrs_{block_ptrs},
        tiles_per_block_{block_seqlen / CTA_S},
        kv_offset_{kv_offset},
        local_id_offset_{local_id_offset}
    {
    }

    __device__ void SetTile(int tile_id)
    {
        tile_id += local_id_offset_;
        if constexpr (std::is_integral_v<BlockSeqLen>) {
            block_id_ = tile_id >> (31 - __clz(tiles_per_block_));  // this is somehow faster than `__ffs`
            local_id  = tile_id & (tiles_per_block_ - 1);
        }
        else {
            block_id_ = tile_id / tiles_per_block_;
            local_id  = tile_id % tiles_per_block_;
        }
        block = block_ptrs_[block_id_];
    }

    __device__ void Advance()
    {
        --local_id;
        if (local_id < 0) {
            local_id += tiles_per_block_;
            block_id_ -= 1;
        }
        if (block_id_ >= 0) {
            block = block_ptrs_[block_id_];
        }
    }

    template<int Idx>
    __device__ const T* OffsetData(int offset) const
    {
        return block + local_id * CTA_S * HeadDim + kv_offset_[Idx] + offset;
    }
};

template<class T, int CTA_S, int HeadDim>
struct LinearTileIter {

    const T* key_;
    const T* key_ptr_;

    int tile_id_;
    int offset_to_val_;

    __device__ LinearTileIter(const T* key, int offset_to_val): key_{key}, offset_to_val_{offset_to_val} {}

    __device__ void SetTile(int tile_id)
    {
        key_ptr_ = key_ + tile_id * CTA_S * HeadDim;
        tile_id_ = tile_id;
    }

    __device__ void Advance()
    {
        --tile_id_;
        if (tile_id_ >= 0) {
            key_ptr_ -= CTA_S * HeadDim;
        }
    }

    template<int Idx>
    __device__ const T* OffsetData(int offset) const
    {
        if constexpr (Idx == 0) {
            return key_ptr_ + offset;
        }
        else {
            return key_ptr_ + offset + offset_to_val_;
        }
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
