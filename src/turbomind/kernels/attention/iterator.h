// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"
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
    int dst_offset_;
    int offset_c_;
    int offset_s_;

    __device__ BaseGmemIterator(int warp_id, int lane_id)
    {
        int2 offsets = Map::get_offset(warp_id, lane_id);
        src_offset_  = offsets.x + offsets.y * Map::kDimC;
        dst_offset_  = offsets.x + offsets.y * SmemLayout::kStride;
        offset_c_    = offsets.x;
        offset_s_    = offsets.y;
    }

    __device__ void SetSmem(T* smem)
    {
        smem_ = smem;
    }

    __device__ void ClearSmem(int offset)
    {
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                constexpr Array<T, Map::kAccessC> kZeros{};
                Store(&smem_[dst_offset_ + offset + s * Map::kDeltaS * SmemLayout::kStride + c * Map::kDeltaC], kZeros);
            }
        }
    }
};

template<int Bits, int Base, int Shift>
struct Swizzle {

    using bit_mask = std::integral_constant<int, (1 << Bits) - 1>;
    using yyy_mask = std::integral_constant<int, bit_mask{} << (Base + Shift)>;
    using shift    = std::integral_constant<int, Shift>;

    template<class Offset>
    __host__ __device__ constexpr static auto apply(Offset offset)
    {
        return offset ^ ((offset & yyy_mask{}) >> shift{});
    }

    template<class Offset>
    __host__ __device__ constexpr auto operator()(Offset offset)
    {
        return apply(offset);
    }
};

struct Identity {

    template<class Offset>
    __device__ constexpr static auto apply(Offset offset)
    {
        return offset;
    }

    template<class Offset>
    __device__ Offset operator()(Offset offset)
    {
        return apply(offset);
    }

    template<int D>
    __device__ int AdvanceS(int offset, int s0, int s1)
    {
        return offset;
    }
};

template<int Stride, class Swizzle_>
struct SmemLayout {
    static constexpr int kStride = Stride;

    using Swizzle = Swizzle_;

    __forceinline__ __device__ static int apply(int s, int c, int offset = 0)
    {
        return Swizzle::apply(s * kStride + c + offset);
    }

    __forceinline__ __device__ int operator()(int s, int c, int offset = 0)
    {
        return apply(s, c, offset);
    }
};

template<class T, class Layout>
struct SmemAccessor {
    T*     ptr_;
    Layout layout_;

    __device__ SmemAccessor(T* ptr): ptr_{ptr} {}

    __device__ T& operator()(int s, int c, int offset = 0)
    {
        return ptr_[layout_(s, c, offset)];
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

    __device__ BlockTileIter(const T** block_ptrs, BlockSeqLen block_seqlen, Array<int, 2> kv_offset):
        block_ptrs_{block_ptrs}, tiles_per_block_{block_seqlen / CTA_S}, kv_offset_{kv_offset}
    {
    }

    __device__ void SetTile(int tile_id)
    {
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
struct LinearTileIter2 {

    const T* key_;
    const T* key_ptr_;

    int tile_id_;
    int offset_to_val_;

    __device__ LinearTileIter2(const T* key, int offset_to_val): key_{key}, offset_to_val_{offset_to_val} {}

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

}  // namespace turbomind
