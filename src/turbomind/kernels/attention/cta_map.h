// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::attention {

#if 1
struct AttentionCtaMap {

    int q_cta_cnt_;
    int h_cta_cnt_;
    int batch_size_;
    int split_cnt_;

    __host__ __device__
    AttentionCtaMap(int max_q_len, int batch_size, int head_num, int cta_q, int cta_h, int split_cnt):
        q_cta_cnt_((max_q_len + cta_q - 1) / cta_q),
        h_cta_cnt_(head_num / cta_h),
        batch_size_(batch_size),
        split_cnt_(split_cnt)
    {
    }

    __host__ __device__ void set_split_cnt(int value)
    {
        split_cnt_ = value;
    }

    __host__ dim3 get_grid_shape() const
    {
        return dim3(q_cta_cnt_, batch_size_, split_cnt_ * h_cta_cnt_);
    }
    __device__ int query_idx() const
    {
        return blockIdx.x;
    }
    __device__ int head_idx() const
    {
        return blockIdx.z % h_cta_cnt_;
    }
    __device__ int batch_idx() const
    {
        return blockIdx.y;
    }
    __device__ int split_idx() const
    {
        return blockIdx.z / h_cta_cnt_;
    }
    __device__ int split_count() const
    {
        return split_cnt_;
    }
};
#else
struct AttentionCtaMap {

    int q_cta_cnt_;
    int h_cta_cnt_;
    int batch_size_;
    int split_cnt_;

    __host__ __device__
    AttentionCtaMap(int max_q_len, int batch_size, int head_num, int cta_q, int cta_h, int split_cnt):
        q_cta_cnt_((max_q_len + cta_q - 1) / cta_q),
        h_cta_cnt_(head_num / cta_h),
        batch_size_(batch_size),
        split_cnt_(split_cnt)
    {
    }

    __host__ dim3 get_grid_shape() const
    {
        return dim3(q_cta_cnt_, h_cta_cnt_, batch_size_ * split_cnt_);
    }
    __device__ int query_idx() const
    {
        return blockIdx.x;
    }
    __device__ int head_idx() const
    {
        return blockIdx.y;
    }
    __device__ int batch_idx() const
    {
        return blockIdx.z / split_cnt_;
    }
    __device__ int split_idx() const
    {
        return blockIdx.z % split_cnt_;
    }
    __device__ int split_count() const
    {
        return split_cnt_;
    }
};
#endif

struct DecodingCtaMap {
    static __host__ dim3 get_grid_shape(int head_num, int batch_size, int split_count, int cta_h)
    {
        return dim3(head_num / cta_h, batch_size, split_count);
    }
    __device__ int query_idx() const
    {
        return 0;
    }
    __device__ int head_idx() const
    {
        return blockIdx.x;
    }
    __device__ int batch_idx() const
    {
        return blockIdx.y;
    }
    __device__ int split_idx() const
    {
        return blockIdx.z;
    }
    __device__ int split_count() const
    {
        return gridDim.z;
    }
};

struct ReduceCtaMap {
    static __host__ dim3 get_grid_shape(int query_num, int head_num, int max_split_cnt, int cta_k)
    {
        return dim3(head_num, query_num, (max_split_cnt + cta_k - 1) / cta_k);
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