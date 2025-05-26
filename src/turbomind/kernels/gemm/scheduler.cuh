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

template<Order order, class Cluster, int striped_m, bool striped_n, bool is_grouped_gemm, int batch_dim>
struct TileScheduler {
    int4 gemm_shape_;
    int2 tiled_shape_;
    int3 tile_shape_;
    int  log_tile_;

    int k_iters_;

    int2 tile_offset_;
    int2 iter_k_range_;

    int cluster_idx_;

    int2 cluster_tiled_shape_;

    int2 swizzled_shape_;
    int  clusters_;

    int2 is_valid_;  // {is_valid_cta_tile, is_valid_cluster_tile}

    const int* offsets_;
    int        group_idx_ = 0;

    cutlass::FastDivmod swizzled_shape_x;

public:
    TM_HOST_DEVICE void init(int4 gemm_shape, int log_tile, int3 tile_shape)
    {
        gemm_shape_ = gemm_shape;
        tile_shape_ = tile_shape;

        log_tile_ = log_tile;
        k_iters_  = cdiv(gemm_shape_.z, tile_shape.z);

        if constexpr (!is_grouped_gemm) {
            tiled_shape_           = get_tiled_shape(gemm_shape.x, gemm_shape.y, tile_shape.x, tile_shape.y);
            cluster_tiled_shape_.x = cdiv(tiled_shape_.x, Cluster::M);
            cluster_tiled_shape_.y = cdiv(tiled_shape_.y, Cluster::N);
            swizzled_shape_        = get_swizzled_shape(cluster_tiled_shape_, log_tile_);
            clusters_              = swizzled_shape_.x * swizzled_shape_.y;
            swizzled_shape_x       = {(int)swizzled_shape_.x};
        }
        else {
            int num      = gemm_shape_.w;
            tiled_shape_ = get_tiled_shape(gemm_shape.x + num * tile_shape.x, gemm_shape.y, tile_shape.x, tile_shape.y);
            cluster_tiled_shape_.x = cdiv(tiled_shape_.x, Cluster::M);
            cluster_tiled_shape_.y = cdiv(tiled_shape_.y, Cluster::N);
            swizzled_shape_        = get_swizzled_shape(cluster_tiled_shape_, log_tile_);
            clusters_              = swizzled_shape_.x * swizzled_shape_.y;
            // M is runtime value
        }
    }

    TM_HOST_DEVICE void reinit() {}

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

        swizzled_shape_x(cluster_idx_y, cluster_idx_x, cluster_idx);

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
                        offset_y + cluster_tile_offset.y * (striped_n ? 1 : Cluster::N)};

        iter_k_range_ = {0, k_iters_};

        is_valid_.x = tile_offset_.x < tiled_shape_.x && tile_offset_.y < tiled_shape_.y;
        is_valid_.y = cluster_tile_offset.x < cluster_tiled_shape_.x && cluster_tile_offset.y < cluster_tiled_shape_.y;
    }

    TM_DEVICE int update()
    {
        int group = -1;
        for (int g = group_idx_ + threadIdx.x % WARP_SIZE; g < gemm_shape_.w; g += WARP_SIZE) {
            int beg = (offsets_[g + 0] / gemm_shape_.x + g + 0) * gemm_shape_.y;
            int end = (offsets_[g + 1] / gemm_shape_.x + g + 1) * gemm_shape_.y;
            if (beg <= cluster_idx_ && cluster_idx_ < end) {
                group = g;
            }
        }
        auto mask  = __ballot_sync((uint32_t)-1, group >= 0);
        group_idx_ = __shfl_sync((uint32_t)-1, group, __ffs(mask) - 1);

        gemm_shape_.x  = offsets_[group_idx_ + 1] - offsets_[group_idx_];
        tiled_shape_.x = cdiv(gemm_shape_.x, tile_shape_.x);

        swizzled_shape_ = get_swizzled_shape(tiled_shape_, log_tile_);

        auto beg = (offsets_[group_idx_] / gemm_shape_.x + group_idx_) * gemm_shape_.y;

        return cluster_idx_ - beg;
    }

    TM_DEVICE bool next(int n = 1)
    {
        cluster_idx_ += n * (int)cute::cluster_grid_dims().x;

        if (cluster_idx_ >= clusters_) {
            return false;
        }

        auto cluster_idx = is_grouped_gemm ? update() : cluster_idx_;

        unswizzle(cluster_idx);

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