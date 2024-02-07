// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::attention {

struct AttentionCtaMap {
    static __host__ dim3 get_grid_shape(int head_num, int batch_size, int max_q_len, int cta_h, int cta_q)
    {
        return dim3((max_q_len + cta_q - 1) / cta_q, batch_size, head_num / cta_h);
    }
    static __device__ int query_idx()
    {
        return blockIdx.x;
    }
    static __device__ int head_idx()
    {
        return blockIdx.z;
    }
    static __device__ int batch_idx()
    {
        return blockIdx.y;
    }
    static __device__ int split_idx()
    {
        return 0;
    }
    static __device__ int split_count()
    {
        return 1;
    }
};

struct DecodingCtaMap {
    static __host__ dim3 get_grid_shape(int head_num, int batch_size, int split_count, int cta_h)
    {
        return dim3(head_num / cta_h, batch_size, split_count);
    }
    static __device__ int query_idx()
    {
        return 0;
    }
    static __device__ int head_idx()
    {
        return blockIdx.x;
    }
    static __device__ int batch_idx()
    {
        return blockIdx.y;
    }
    static __device__ int split_idx()
    {
        return blockIdx.z;
    }
    static __device__ int split_count()
    {
        return gridDim.z;
    }
};

struct ReduceCtaMap {
    static __host__ dim3 get_grid_shape(int query_num, int head_num, int max_split_cnt)
    {
        return dim3(head_num, query_num, (max_split_cnt + 31) / 32);
    }
    static __device__ int query_idx()
    {
        return blockIdx.y;
    }
    static __device__ int head_idx()
    {
        return blockIdx.x;
    }
    static __device__ int split_idx()
    {
        return blockIdx.z;
    }
};

}  // namespace turbomind::attention