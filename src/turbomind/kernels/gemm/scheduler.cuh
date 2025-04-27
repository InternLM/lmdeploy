// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "cutlass/cutlass.h"

namespace turbomind::gemm {

template<Order order>
class TileScheduler {
    int4 gemm_shape_;
    int4 tiled_shape_;
    int  log_tile_;

    int chunk_offset_;
    int chunks_per_split_;
    int iter_k_per_chunk_;

    int4 tile_offset_;
    int2 iter_k_range_;

    int cta_idx_;

    dim3 grid_shape_;

public:
    TM_HOST_DEVICE
    TileScheduler(int4 gemm_shape, int2 tiled_mn, int splits, int log_tile, int cta_k, int chunk_size):
        gemm_shape_{gemm_shape}, tiled_shape_{tiled_mn.x, tiled_mn.y, splits}, log_tile_{log_tile}
    {
        const int chunk_cnt = cdiv(gemm_shape_.z, chunk_size);

        iter_k_per_chunk_ = chunk_size / cta_k;
        chunks_per_split_ = chunk_cnt / splits;

        chunk_offset_ = splits - chunk_cnt % splits;

        grid_shape_ = get_grid_shape(tiled_shape_, log_tile_);
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
        return grid_shape_.x * grid_shape_.y * grid_shape_.z;
    }

    TM_DEVICE void grid_init()
    {
        cta_idx_ = (int)blockIdx.x - (int)gridDim.x;
    }

    TM_HOST_DEVICE bool init(int block_idx_x, int block_idx_y, int block_idx_z)
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

        return tile_offset_.x < tiled_shape_.x && tile_offset_.y < tiled_shape_.y && tile_offset_.z < tiled_shape_.z;
    }

    TM_DEVICE bool next()
    {
        cta_idx_ += gridDim.x;
        int cta_idx_x = cta_idx_ % grid_shape_.x;
        int cta_idx_y = cta_idx_ / grid_shape_.x % grid_shape_.y;
        int cta_idx_z = cta_idx_ / grid_shape_.x / grid_shape_.y;
        return init(cta_idx_x, cta_idx_y, cta_idx_z);
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

}  // namespace turbomind::gemm