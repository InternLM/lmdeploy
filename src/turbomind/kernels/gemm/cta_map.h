// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind::gemm {

struct CtaMap {

    TM_HOST_DEVICE static int3 get_tiled_shape(int m, int n, int k, int cta_m, int cta_n, int split_cnt)
    {
        return {(m + cta_m - 1) / cta_m, (n + cta_n - 1) / cta_n, split_cnt};
    }

    TM_HOST_DEVICE static int get_log_tile(int3 tiled_shape, int N)
    {
        auto n = tiled_shape.y;
        if (N >= 32 && n >= 24)
            return 5;
        if (N >= 16 && n >= 12)
            return 4;
        if (N >= 8 && n >= 6)
            return 3;
        if (N >= 4 && n >= 3)
            return 2;
        if (N >= 2 && n >= 2)
            return 1;
        return 0;
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

struct CtaMapN: public CtaMap {
    TM_HOST_DEVICE static dim3 get_grid_shape(int3 tiled_shape, int log_tile)
    {
        int tile = 1 << log_tile;
        return {static_cast<unsigned>(tiled_shape.y * tile),               // n * tile
                static_cast<unsigned>((tiled_shape.x + tile - 1) / tile),  // m / tile
                static_cast<unsigned>(tiled_shape.z)};
    }
    TM_HOST_DEVICE static int get_log_tile(int3 tiled_shape, int M)
    {
        auto m = tiled_shape.x;
        if (M >= 32 && m >= 24)
            return 5;
        if (M >= 16 && m >= 12)
            return 4;
        if (M >= 8 && m >= 6)
            return 3;
        if (M >= 4 && m >= 3)
            return 2;
        if (M >= 2 && m >= 2)
            return 1;
        return 0;
    }
    TM_DEVICE static int3 get_tile_offset(int log_tile)
    {
        int block_idx_x = blockIdx.x;
        int block_idx_y = blockIdx.y;
        int block_idx_z = blockIdx.z;
        return {(block_idx_y << log_tile) + (block_idx_x & ((1 << log_tile) - 1)),  //
                (block_idx_x >> log_tile),
                block_idx_z};
    }
};

}  // namespace turbomind::gemm
