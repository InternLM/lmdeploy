// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

template<Order order, int tile_m, int tile_n, int tile_k, int chunk_k, int split_k, int group_axis_>
struct SchedulerSm70 {

    static constexpr int group_axis = group_axis_;

    static constexpr Array<int, 3> tile_shape{tile_m, tile_n, tile_k};

    static_assert(chunk_k % tile_k == 0);
    static constexpr int chunk_iters = chunk_k / tile_k;

    Array<int, 4> gemm_shape_;
    Array<int, 3> tiles_;

    int log_tile_;

    int splits_;
    int split_chunks_;
    int chunk_offset_;

    const int* offsets_;

    struct Tile {
        Array<int, 3> tile_id;
        Array<int, 3> shape;

        int group_id;
        int split_id;

        int k_iters;

        int linear_tile_id;
    };

    struct SharedStorage {
        int group_id;
        int dynamic_dim;
        int base_tile_id;
    };

    __host__ dim3 get_grid_shape()
    {
        auto tiles = tiles_;
        tiles[2]   = splits_;
        auto shape = get_swizzled_shape(tiles, log_tile_);
        return dim3(shape[0], shape[1], shape[2]);
    }

    __host__ SchedulerSm70(Array<int, 4> gemm_shape, int log_tile = 0, int splits = 1):
        gemm_shape_{gemm_shape}, log_tile_{log_tile}, splits_{splits}
    {
        tiles_[0] = cdiv(gemm_shape[0], tile_m);
        tiles_[1] = cdiv(gemm_shape[1], tile_n);
        tiles_[2] = cdiv(gemm_shape[2], tile_k);

        log_tile_ = log_tile;

        Array<int, 2> log_unit{};
        log_unit[1 - (int)order] = log_tile;

        tiles_[0] = round_up(tiles_[0], 1 << log_unit[0]);
        tiles_[1] = round_up(tiles_[1], 1 << log_unit[1]);

        if constexpr (group_axis >= 0) {
            constexpr int i = group_axis;
            // overwrite dynamic axis <- estimated upper bound
            tiles_[i] = ((gemm_shape_[i] / tile_shape[i] >> log_unit[i]) + gemm_shape_[3]) << log_unit[i];
        }

        int chunks    = cdiv(gemm_shape[2], chunk_k);
        split_chunks_ = chunks / splits_;
        chunk_offset_ = splits - chunks % splits;
    }

    __device__ int2 get_group_offset(int g)
    {
        constexpr int i = group_axis;

        Array<int, 2> log_unit{};
        log_unit[1 - (int)order] = log_tile_;

        int offset      = __ldg(offsets_ + g);
        int tile_offset = ((offset / tile_shape[i] >> log_unit[i]) + g) << log_unit[i];

        return {offset, tile_offset};
    }

    __device__ int find_group(Array<int, 3>& tile_id, SharedStorage& storage)
    {
        constexpr int axis = group_axis;

        int success = 0;

        const int block_dim = blockDim.x;

        for (int g = threadIdx.x; g < gemm_shape_[3]; g += block_dim) {
            auto [beg, beg_tile] = get_group_offset(g);
            auto [end, end_tile] = get_group_offset(g + 1);

            if (beg_tile <= tile_id[axis] && tile_id[axis] < end_tile) {
                storage.group_id     = g;
                storage.dynamic_dim  = end - beg;
                storage.base_tile_id = beg_tile;
                success              = 1;
            }

            if (tile_id[axis] < end_tile) {
                break;
            }
        }

        return __syncthreads_or(success);
    }

    __device__ Array<int, 3> unswizzle(Array<int, 3> cta_id)
    {
        int tile_c = cta_id[0] >> log_tile_;
        int tile_s = cta_id[1] << log_tile_ | (cta_id[0] & ((1 << log_tile_) - 1));

        Array<int, 3> tile_id;

        tile_id[(int)order]     = tile_c;
        tile_id[1 - (int)order] = tile_s;

        tile_id[2] = cta_id[2];

        // if (threadIdx.x == 0) {
        //     printf("%d %d -> %d %d\n", cta_id[0], cta_id[1], tile_id[0], tile_id[1]);
        // }

        return tile_id;
    }

    template<class Reinit>
    __device__ int init(Tile& tile, SharedStorage& storage, Reinit)
    {
        Array<int, 3> cta_id{(int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z};
        Array<int, 3> tile_id = unswizzle(cta_id);
        Array<int, 3> shape{gemm_shape_[0], gemm_shape_[1], gemm_shape_[2]};

        tile.group_id = 0;
        tile.split_id = 0;

        constexpr int axis = group_axis;

        if constexpr (axis >= 0) {
            if (offsets_) {
                if constexpr (!Reinit::value) {
                    if (!find_group(tile_id, storage)) {
                        return false;
                    }
                }
                tile_id[axis] -= storage.base_tile_id;
                shape[axis]   = storage.dynamic_dim;
                tile.group_id = storage.group_id;
            }
        }

        if constexpr (split_k) {
            tile.split_id = tile_id[2];
            int chunk_id  = tile.split_id * split_chunks_ + max(tile.split_id - chunk_offset_, 0);
            tile.k_iters  = (split_chunks_ + int(tile.split_id >= chunk_offset_)) * chunk_iters;
            tile_id[2]    = chunk_id * chunk_iters;
        }
        else {
            tile.k_iters = split_chunks_ * chunk_iters;
            tile_id[2]   = 0;
        }

        tile.tile_id = tile_id;
        tile.shape   = shape;

        tile.linear_tile_id = tile_id[1 - (int)order] * tiles_[(int)order] + tile_id[(int)order];

        return true;
    }

    __host__ __device__ static Array<int, 3> get_swizzled_shape(Array<int, 3> tiles, int log_tile)
    {
        constexpr int i = (int)order;  // expansion axis
        return {tiles[i] << log_tile, (tiles[1 - i] + (1 << log_tile) - 1) >> log_tile, 1};
    }

    __host__ int get_max_swizzle()
    {
        constexpr int axis = 1 - (int)order;

        int n = tiles_[axis];

        if (group_axis == axis) {
            n = cdiv(n, gemm_shape_[3]);
        }

        return get_log_tile(n);
    }

    __host__ __device__ static int get_log_tile(int size)
    {
        if (size >= 24)
            return 5;
        if (size >= 12)
            return 4;
        if (size >= 6)
            return 3;
        if (size >= 3)
            return 2;
        if (size >= 2)
            return 1;
        return 0;
    }
};

}  // namespace turbomind::gemm