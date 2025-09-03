// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

TM_HOST_DEVICE constexpr int get_log_tile(int size, int tile_size)
{
    if (tile_size >= 32 && size >= 24)
        return 5;
    if (tile_size >= 16 && size >= 12)
        return 4;
    if (tile_size >= 8 && size >= 6)
        return 3;
    if (tile_size >= 4 && size >= 3)
        return 2;
    if (tile_size >= 2 && size >= 2)
        return 1;
    return 0;
}

TM_HOST_DEVICE constexpr int2 get_tiled_shape(int m, int n, int cta_m, int cta_n)
{
    return {ceil_div(m, cta_m), ceil_div(n, cta_n)};
}

struct CtaMap_ {

    TM_HOST_DEVICE static int3 get_tiled_shape(int m, int n, int k, int cta_m, int cta_n, int split_cnt)
    {
        return {(m + cta_m - 1) / cta_m, (n + cta_n - 1) / cta_n, split_cnt};
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int N)
    {
        return gemm::get_log_tile(tiled_mn.y, N);
    }

    TM_HOST_DEVICE static dim3 get_grid_shape(int3 tiled_shape, int log_tile)
    {
        int tile = 1 << log_tile;
        return {static_cast<unsigned>(tiled_shape.x * tile),
                static_cast<unsigned>((tiled_shape.y + tile - 1) / tile),
                static_cast<unsigned>(tiled_shape.z)};
    }

    TM_DEVICE static int3 get_tile_offset(int log_tile)
    {
        int block_idx_x = blockIdx.x;
        int block_idx_y = blockIdx.y;
        int block_idx_z = blockIdx.z;
        return {(block_idx_x >> log_tile),  //
                (block_idx_y << log_tile) + (block_idx_x & ((1 << log_tile) - 1)),
                block_idx_z};
    }
};

template<Order order_>
class GemmScheduler {

    static constexpr auto order = order_;

    int4 gemm_shape_;
    int4 tiled_shape_;
    int  log_tile_;

    int chunk_offset_;
    int chunks_per_split_;
    int iter_k_per_chunk_;

    int4 tile_offset_;
    int2 iter_k_range_;

public:
    TM_HOST_DEVICE
    GemmScheduler(int4 gemm_shape, int2 tiled_mn, int splits, int log_tile, int cta_k, int chunk_size):
        gemm_shape_{gemm_shape}, tiled_shape_{tiled_mn.x, tiled_mn.y, splits}, log_tile_{log_tile}
    {
        const int chunk_cnt = cdiv(gemm_shape_.z, chunk_size);

        iter_k_per_chunk_ = chunk_size / cta_k;
        chunks_per_split_ = chunk_cnt / splits;
        chunk_offset_     = splits - chunk_cnt % splits;
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int tile_size)
    {
        return gemm::get_log_tile(order == kColMajor ? tiled_mn.y : tiled_mn.x, tile_size);
    }

    TM_HOST_DEVICE static dim3 get_grid_shape(int4 tiled_shape, int log_tile)
    {
        const int tile = 1 << log_tile;
        if constexpr (order == kColMajor) {
            return {(unsigned)(tiled_shape.x * tile), (unsigned)(cdiv(tiled_shape.y, tile)), (unsigned)(tiled_shape.z)};
        }
        else {
            return {(unsigned)(tiled_shape.y * tile), (unsigned)(cdiv(tiled_shape.x, tile)), (unsigned)(tiled_shape.z)};
        }
    }

    TM_HOST_DEVICE dim3 get_grid_shape() const
    {
        return get_grid_shape(tiled_shape_, log_tile_);
    }

    TM_HOST_DEVICE std::true_type init(int block_idx_x, int block_idx_y, int block_idx_z)
    {
        if constexpr (order == kColMajor) {
            tile_offset_ = {(block_idx_x >> log_tile_),
                            (block_idx_y << log_tile_) + (block_idx_x & ((1 << log_tile_) - 1)),
                            (block_idx_z)};
        }
        else {
            tile_offset_ = {(block_idx_y << log_tile_) + (block_idx_x & ((1 << log_tile_) - 1)),
                            (block_idx_x >> log_tile_),
                            (block_idx_z)};
        }
        tile_offset_.w       = 0;
        const int chunk_id   = tile_offset_.z * chunks_per_split_ + max(tile_offset_.z - chunk_offset_, 0);
        const int iter_k_beg = chunk_id * iter_k_per_chunk_;
        const int iter_k_cnt = (chunks_per_split_ + int(tile_offset_.z >= chunk_offset_)) * iter_k_per_chunk_;
        iter_k_range_        = {iter_k_beg, iter_k_beg + iter_k_cnt};

        return {};
    }

    TM_DEVICE std::true_type init()
    {
        return init(blockIdx.x, blockIdx.y, blockIdx.z);
    }

    TM_DEVICE int4 gemm_shape() const
    {
        return gemm_shape_;
    }

    TM_DEVICE int4 tiled_shape() const
    {
        return tiled_shape_;
    }

    TM_DEVICE int4 tile_offset() const
    {
        return tile_offset_;
    }

    TM_DEVICE int2 iter_k_range() const
    {
        return iter_k_range_;
    }

    TM_DEVICE int tile_id() const
    {
        return tile_offset_.x * tiled_shape_.y + tile_offset_.y;
    }
};

template<Order order_>
class DynamicScheduler {

    static constexpr auto order = order_;

    int ctas_;

    const int4* __restrict__ gemm_shapes_;    // [group_num]
    const int4* __restrict__ tiled_shapes_;   // [group_num]
    const int2* __restrict__ offsets_mn_;     // [group_num]
    const int4* __restrict__ tile_offsets_;   // [ctas]
    const int2* __restrict__ iter_k_ranges_;  // [ctas]
    const int* __restrict__ tile_ids_;        // [ctas]

    int4 gemm_shape_;
    int4 tiled_shape_;
    int4 tile_offset_;
    int2 iter_k_range_;
    int2 base_mn_;

public:
    DynamicScheduler(const Tape& tape):
        ctas_{tape.ctas},
        gemm_shapes_{tape.gemm_shapes},
        tiled_shapes_{tape.tiled_shapes},
        tile_offsets_{tape.tile_offsets},
        iter_k_ranges_{tape.iter_k_ranges},
        tile_ids_{tape.tile_ids}
    {
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int tile_size)
    {
        return gemm::get_log_tile(order == kColMajor ? tiled_mn.y : tiled_mn.x, tile_size);
    }

    TM_HOST_DEVICE dim3 get_grid_shape()
    {
        return {(unsigned)ctas_, 1, 1};
    }

    TM_DEVICE bool init()
    {
        const int block_idx = blockIdx.x;

        const auto [cta_m_id, cta_n_id, cta_k_id, group_id] = __ldg(tile_offsets_ + block_idx);

        if (group_id < 0) {
            return false;
        }

        gemm_shape_  = __ldg(gemm_shapes_ + group_id);
        tiled_shape_ = __ldg(tiled_shapes_ + group_id);
        base_mn_     = __ldg(offsets_mn_ + group_id);

        tile_offset_ = {cta_m_id, cta_n_id, cta_k_id, group_id};

        iter_k_range_ = __ldg(iter_k_ranges_ + block_idx);

        return true;
    }

    TM_DEVICE int4 gemm_shape() const
    {
        return gemm_shape_;
    }

    TM_DEVICE int4 tiled_shape() const
    {
        return tiled_shape_;
    }

    TM_DEVICE int4 tile_offset() const
    {
        return tile_offset_;
    }

    TM_DEVICE int2 iter_k_range() const
    {
        return iter_k_range_;
    }

    TM_DEVICE int tile_id() const
    {
        return tile_ids_[blockIdx.x];
    }
};

template<class S>
struct is_dynamic_scheduler: std::false_type {
};

template<Order order>
struct is_dynamic_scheduler<DynamicScheduler<order>>: std::true_type {
};

}  // namespace turbomind::gemm
