// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "cutlass/fast_math.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/cutlass.h"

namespace turbomind::gemm {

template<Order order,
         class Cluster,
         int  striped_m,
         bool striped_n,
         int  tile_m,
         int  tile_n,
         bool is_grouped_gemm,
         int  batch_dim>
struct TileScheduler {
    int4 gemm_shape_;
    int2 tiled_shape_;
    // int3 tile_shape_;
    int log_tile_;

    int k_iters_;

    int2 tile_offset_;
    int2 iter_k_range_;

    int cluster_idx_;

    static constexpr int2 tile_{tile_m, tile_n};
    static constexpr int2 cluster_tile_{tile_m * Cluster::M, tile_n* Cluster::N};

    int clusters_;

    //////// v2 /////
    int2 cluster_tiles_;
    int2 padded_cluster_tiles_;
    int2 swizzled_cluster_tiles_;
    /////////////

    int2 is_valid_;  // {is_valid_cta_tile, is_valid_cluster_tile}

    const int* offsets_;

    int group_idx_ = -1;

    int group_beg_ = 0;
    int group_end_ = 0;

    bool is_next_ = true;

    // cutlass::FastDivmod swizzled_shape_x;

public:
    TM_HOST_DEVICE void init(int4 gemm_shape, int log_tile, int3 tile_shape)
    {
        gemm_shape_ = gemm_shape;

        log_tile_ = log_tile;
        k_iters_  = cdiv(gemm_shape_.z, tile_shape.z);

        if constexpr (is_grouped_gemm) {

            printf("gemm shape: %d %d %d\n", gemm_shape.x, gemm_shape.y, gemm_shape.z);
            int num = gemm_shape_.w;

            const int2 swizzle_unit = get_swizzled_shape({1, 1}, log_tile);  // {8, 1}

            tiled_shape_.x = cdiv(gemm_shape.x, tile_.x);
            tiled_shape_.y = cdiv(gemm_shape.y, tile_.y);

            cluster_tiles_.x = cdiv(gemm_shape.x, cluster_tile_.x);  // useless
            cluster_tiles_.y = cdiv(gemm_shape.y, cluster_tile_.y);

            printf("swizzle units: %d %d\n", swizzle_unit.y, swizzle_unit.x);

            // num of tiles won't change after swizzle
            padded_cluster_tiles_.x = (cdiv(gemm_shape.x, cluster_tile_.x * swizzle_unit.y) + num) * swizzle_unit.y;
            padded_cluster_tiles_.y = (cdiv(gemm_shape.y, cluster_tile_.y * swizzle_unit.x) + 0x0) * swizzle_unit.x;

            printf("padded   cluster tiles: %d %d\n", padded_cluster_tiles_.x, padded_cluster_tiles_.y);

            swizzled_cluster_tiles_ = get_swizzled_shape(padded_cluster_tiles_, log_tile);

            printf("swizzled cluster tiles: %d %d\n", swizzled_cluster_tiles_.x, swizzled_cluster_tiles_.y);

            clusters_ = padded_cluster_tiles_.x * padded_cluster_tiles_.y;

            printf("clusters = %d\n", clusters_);
            // M is runtime value
        }
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int tile_size)
    {
        return gemm::get_log_tile(order == kColMajor ? tiled_mn.y : tiled_mn.x, tile_size);
    }

    TM_HOST_DEVICE static int2 get_swizzled_shape(int2 tiled_shape, int log_tile)
    {
        const int tile = 1 << log_tile;

        if constexpr (order == kColMajor) {
            return {tiled_shape.x * tile, (tiled_shape.y + tile - 1) >> log_tile};
        }
        else {
            return {tiled_shape.y * tile, (tiled_shape.x + tile - 1) >> log_tile};
        }
    }

    TM_DEVICE void grid_init(int n = 1)
    {
        cluster_idx_ = (int)cute::cluster_id_in_grid().x - n * (int)cute::cluster_grid_dims().x;
    }

    TM_DEVICE void unswizzle(int cluster_idx)
    {
        int cluster_idx_x, cluster_idx_y;

        if constexpr (is_grouped_gemm) {
            cluster_idx_x = cluster_idx % swizzled_cluster_tiles_.x;
            cluster_idx_y = cluster_idx / swizzled_cluster_tiles_.x;
        }
        else {
            // swizzled_shape_x(cluster_idx_y, cluster_idx_x, cluster_idx);
        }

        auto [cluster_cta_m, cluster_cta_n] = Cluster::cta_mn(cute::block_id_in_cluster().x);

        const int offset_x = cluster_cta_m * (striped_m ? cluster_tiles_.x : 1);
        const int offset_y = cluster_cta_n * (striped_n ? cluster_tiles_.y : 1);

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
                        offset_y + cluster_tile_offset.y * (striped_n ? 1 : Cluster::N)};

        // if (threadIdx.x == 0) {
        //     printf("g:%4d, tiled idx:%4d, cluster:%4d%4d, tiled offset:%4d%4d\n",
        //            group_idx_,
        //            cluster_idx,
        //            cluster_idx_x,
        //            cluster_idx_y,
        //            tile_offset_.x,
        //            tile_offset_.y);
        // }

        iter_k_range_ = {0, k_iters_};

        is_valid_.x = tile_offset_.x < tiled_shape_.x && tile_offset_.y < tiled_shape_.y;
        is_valid_.y = cluster_tile_offset.x < cluster_tiles_.x && cluster_tile_offset.y < cluster_tiles_.y;
    }

    TM_DEVICE int get_start_index(int g)
    {
        return (offsets_[g] / cluster_tile_.x + g) * padded_cluster_tiles_.y;
    }

    TM_DEVICE bool update()
    {
        int group = -1;

        PRAGMA_NO_UNROLL
        for (int g = threadIdx.x % WARP_SIZE; g < gemm_shape_.w; g += WARP_SIZE) {
            int beg = get_start_index(g);
            int end = get_start_index(g + 1);
            if (beg <= cluster_idx_ && cluster_idx_ < end) {
                group = g;
            }
        }

        auto mask = __ballot_sync((uint32_t)-1, group >= 0);

        if (!mask) {
            return false;
        }

        group = __shfl_sync((uint32_t)-1, group, __ffs(mask) - 1);

        if (group != group_idx_) {
            group_idx_ = group;

            gemm_shape_.x = offsets_[group_idx_ + 1] - offsets_[group_idx_];

            tiled_shape_.x   = cdiv(gemm_shape_.x, tile_.x);
            cluster_tiles_.x = cdiv(gemm_shape_.x, cluster_tile_.x);

            swizzled_cluster_tiles_ = get_swizzled_shape(cluster_tiles_, log_tile_);

            group_beg_ = get_start_index(group_idx_);
            group_end_ = get_start_index(group_idx_ + 1);
        }

        return true;

        // if (threadIdx.x == 0 && cluster_idx_ - beg == 0) {
        //     printf("g:%4d, tiled shape:%4d%4d\n", group_idx_, tiled_shape_.x, tiled_shape_.y);
        // }
        // if (threadIdx.x == 0) {
        //     printf("g:%4d, beg:%4d\n", group_idx_, beg);
        // }
    }

    TM_DEVICE bool next(int n = 1)
    {
        cluster_idx_ += n * (int)cute::cluster_grid_dims().x;

        if (cluster_idx_ >= clusters_) {
            return false;
        }

        if constexpr (is_grouped_gemm) {
            if (!update()) {
                return false;
            }
            unswizzle(cluster_idx_ - group_beg_);
        }
        else {
            unswizzle(cluster_idx_);
        }

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

    TM_DEVICE int2 tiled_shape() const
    {
        return tiled_shape_;
    }

    TM_DEVICE int2 tile_offset() const
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