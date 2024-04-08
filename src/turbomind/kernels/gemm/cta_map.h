// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"

namespace turbomind::gemm {

struct CtaMap2 {
    __host__ __device__ static int3 get_tiled_shape(int m, int n, int k, int cta_m, int cta_n, int split_cnt)
    {
        return {(m + cta_m - 1) / cta_m, (m + cta_n - 1) / cta_n, split_cnt};
    }

    __host__ __device__ static dim3 get_grid_shape(int3 tiled_shape)
    {
        return dim3(tiled_shape.x, tiled_shape.y, tiled_shape.z);
    }

    __device__ static int3 get_tile_offset(int log_tile = 0)
    {
        return int3{(int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z};
    }
};

template<int N>
struct CtaSwizzleMap {
    TM_HOST_DEVICE static int3 get_tiled_shape(int m, int n, int k, int cta_m, int cta_n, int split_cnt)
    {
        return {(m + cta_m - 1) / cta_m, (n + cta_n - 1) / cta_n, split_cnt};
    }

    TM_HOST_DEVICE static int get_log_tile(int3 tiled_shape)
    {
        auto n = tiled_shape.y;
        if (N >= 8 && n >= 6) {
            return 3;
        }
        else if (N >= 4 && n >= 3) {
            return 2;
        }
        else if (N >= 2 && n >= 2) {
            return 1;
        }
        else {
            return 0;
        }
    }

    TM_HOST_DEVICE static dim3 get_grid_shape(int3 tiled_shape)
    {
        int tile = 1 << get_log_tile(tiled_shape);
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

}  // namespace turbomind::gemm