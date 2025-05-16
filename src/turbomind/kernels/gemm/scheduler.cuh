// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/cutlass.h"

namespace turbomind::gemm {

template<Order order, class Cluster, int striped_m, bool striped_n>
struct TileScheduler {
    int4 gemm_shape_;
    int4 tiled_shape_;
    int  log_tile_;

    int chunk_offset_;
    int chunks_per_split_;
    int iter_k_per_chunk_;

    int4 tile_offset_;
    int2 iter_k_range_;

    int cluster_idx_;

    int4 cluster_tiled_shape_;

    dim3 swizzled_shape_;
    int  clusters_;

    int2 is_valid_;  // {is_valid_cta_tile, is_valid_cluster_tile}

public:
    TM_HOST_DEVICE
    TileScheduler(int4 gemm_shape, int2 tiled_mn, int splits, int log_tile, int cta_k, int chunk_size):
        gemm_shape_{gemm_shape}, tiled_shape_{tiled_mn.x, tiled_mn.y, splits}, log_tile_{log_tile}
    {
        const int chunk_cnt = cdiv(gemm_shape_.z, chunk_size);

        iter_k_per_chunk_ = chunk_size / cta_k;
        chunks_per_split_ = chunk_cnt / splits;

        chunk_offset_ = splits - chunk_cnt % splits;

        cluster_tiled_shape_   = tiled_shape_;
        cluster_tiled_shape_.x = cdiv(tiled_shape_.x, Cluster::M);
        cluster_tiled_shape_.y = cdiv(tiled_shape_.y, Cluster::N);

        swizzled_shape_ = get_swizzled_shape(cluster_tiled_shape_, log_tile_);

        clusters_ = swizzled_shape_.x * swizzled_shape_.y * swizzled_shape_.z;
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int tile_size)
    {
        return gemm::get_log_tile(order == kColMajor ? tiled_mn.y : tiled_mn.x, tile_size);
    }

    TM_HOST_DEVICE static dim3 get_swizzled_shape(int4 tiled_shape, int log_tile)
    {
        const int tile = 1 << log_tile;
        if constexpr (order == kColMajor) {
            return {unsigned(tiled_shape.x * tile), (unsigned)cdiv(tiled_shape.y, tile), unsigned(tiled_shape.z)};
        }
        else {
            return {unsigned(tiled_shape.y * tile), (unsigned)cdiv(tiled_shape.x, tile), unsigned(tiled_shape.z)};
        }
    }

    TM_DEVICE void unswizzle()
    {
        int                  cluster_idx_x = cluster_idx_ % swizzled_shape_.x;
        int                  cluster_idx_y = cluster_idx_ / swizzled_shape_.x % swizzled_shape_.y;
        [[maybe_unused]] int cluster_idx_z = cluster_idx_ / swizzled_shape_.x / swizzled_shape_.y;

        auto [cluster_cta_m, cluster_cta_n] = Cluster::cta_mn(cute::block_id_in_cluster().x);

        const int offset_x = cluster_cta_m * (striped_m ? cluster_tiled_shape_.x : 1);
        const int offset_y = cluster_cta_n * (striped_n ? cluster_tiled_shape_.y : 1);

        int2 cluster_tile_offset;

        if constexpr (order == kColMajor) {
            cluster_tile_offset = {(cluster_idx_x >> log_tile_),
                                   (cluster_idx_y << log_tile_) + (cluster_idx_x & ((1 << log_tile_) - 1))};
        }
        else {
            cluster_tile_offset = {(cluster_idx_y << log_tile_) + (cluster_idx_x & ((1 << log_tile_) - 1)),
                                   (cluster_idx_x >> log_tile_)};
        }

        tile_offset_ = {offset_x + cluster_tile_offset.x * (striped_m ? 1 : Cluster::M),
                        offset_y + cluster_tile_offset.y * (striped_n ? 1 : Cluster::N),
                        cluster_idx_z,
                        0};

        const int chunk_id   = tile_offset_.z * chunks_per_split_ + max(tile_offset_.z - chunk_offset_, 0);
        const int iter_k_beg = chunk_id * iter_k_per_chunk_;
        const int iter_k_cnt = (chunks_per_split_ + int(tile_offset_.z >= chunk_offset_)) * iter_k_per_chunk_;

        iter_k_range_ = {iter_k_beg, iter_k_beg + iter_k_cnt};

        is_valid_.x =
            tile_offset_.x < tiled_shape_.x && tile_offset_.y < tiled_shape_.y && tile_offset_.z < tiled_shape_.z;
        is_valid_.y = cluster_tile_offset.x < cluster_tiled_shape_.x && cluster_tile_offset.y < cluster_tiled_shape_.y;
    }

    TM_DEVICE void grid_init(int n = 1)
    {
        cluster_idx_ = (int)cute::cluster_id_in_grid().x - n * (int)cute::cluster_grid_dims().x;
    }

    TM_DEVICE bool next(int n = 1)
    {
        cluster_idx_ += n * (int)cute::cluster_grid_dims().x;

        if (cluster_idx_ >= clusters_) {
            return false;
        }

        unswizzle();

        return true;
    }

    TM_DEVICE explicit operator bool() const
    {
        return cluster_idx_ < clusters_;
    }

    TM_DEVICE int2 is_valid_tile() const
    {
        return is_valid_;
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