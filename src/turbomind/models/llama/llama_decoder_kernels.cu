// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/windows/marco.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

namespace turbomind {

template<typename T>
struct res_norm_ops_t {};

template<typename T>
struct res_norm_t {
    res_norm_ops_t<T> f;
    __device__ uint4  addvec(const uint4& a, const uint4& b, const uint4& bias, float& accum) const
    {
        uint4 c;
        c.x = f.cast(f.add(f.cast(a.x), f.cast(b.x), f.cast(bias.x), accum));
        c.y = f.cast(f.add(f.cast(a.y), f.cast(b.y), f.cast(bias.y), accum));
        c.z = f.cast(f.add(f.cast(a.z), f.cast(b.z), f.cast(bias.z), accum));
        c.w = f.cast(f.add(f.cast(a.w), f.cast(b.w), f.cast(bias.w), accum));
        return c;
    }
    __device__ uint4 normvec(const uint4& u, const uint4& s, float factor) const
    {
        uint4 v;
        v.x = f.cast(f.norm(f.cast(u.x), f.cast(s.x), factor));
        v.y = f.cast(f.norm(f.cast(u.y), f.cast(s.y), factor));
        v.z = f.cast(f.norm(f.cast(u.z), f.cast(s.z), factor));
        v.w = f.cast(f.norm(f.cast(u.w), f.cast(s.w), factor));
        return v;
    }
};

template<>
struct res_norm_ops_t<half> {
    __device__ float2 cast(const uint& x) const
    {
        return __half22float2(reinterpret_cast<const half2&>(x));
    }
    __device__ uint cast(const float2& x) const
    {
        auto y = __float22half2_rn(x);
        return reinterpret_cast<uint&>(y);
    }
    __device__ float2 add(const float2& a, const float2& b, const float2& bias, float& accum) const
    {
        float2 c{a.x + b.x + bias.x, a.y + b.y + bias.y};
        accum += c.x * c.x + c.y * c.y;
        return c;
    }
    __device__ float2 norm(const float2& a, const float2& s, float factor) const
    {
        return {a.x * s.x * factor, a.y * s.y * factor};
    }
};

template<>
struct res_norm_ops_t<float> {
    __device__ float cast(const uint& x) const
    {
        return reinterpret_cast<const float&>(x);
    }
    __device__ uint cast(const float& x) const
    {
        return reinterpret_cast<const uint&>(x);
    }
    __device__ float add(const float& a, const float& b, const float& bias, float& accum) const
    {
        float c = a + b + bias;
        accum += c * c;
        return c;
    }
    __device__ float norm(const float& a, const float& s, float factor) const
    {
        return a * s * factor;
    }
};

template<typename T>
__device__ T blockReduceSum(const cg::thread_block& block, T value)
{
    __shared__ float partial[32];

    auto tile = cg::tiled_partition<32>(block);
    value     = cg::reduce(tile, value, cg::plus<float>{});

    if (tile.thread_rank() == 0) {
        partial[tile.meta_group_rank()] = value;
    }

    block.sync();

    value = tile.thread_rank() < tile.meta_group_size() ? partial[tile.thread_rank()] : T{};
    return cg::reduce(tile, value, cg::plus<float>{});
}

template<typename T>
__global__ void fusedAddBiasResidualNorm(T* __restrict__ r_data,
                                         T* __restrict__ x_data,
                                         const T* __restrict__ bias,
                                         const T* __restrict__ scale,
                                         float eps,
                                         int   batch_size,
                                         int   n_dims)
{
    auto block = cg::this_thread_block();
    auto grid  = cg::this_grid();

    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);

    const auto batch_idx            = grid.block_rank();
    uint4* __restrict__ r_ptr       = reinterpret_cast<uint4*>(r_data + batch_idx * n_dims);
    uint4* __restrict__ x_ptr       = reinterpret_cast<uint4*>(x_data + batch_idx * n_dims);
    const uint4* __restrict__ b_ptr = reinterpret_cast<const uint4*>(bias);

    res_norm_t<T> ops;

    float thread_sum{};
    for (auto i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.num_threads()) {
        auto  r  = r_ptr[i];
        auto  x  = x_ptr[i];
        uint4 b  = b_ptr ? b_ptr[i] : uint4{};
        r        = ops.addvec(r, x, b, thread_sum);
        r_ptr[i] = r;
    }

    auto total_sum = blockReduceSum(block, thread_sum);

    float s_inv_mean = rsqrt(total_sum / n_dims + eps);

    const uint4* __restrict__ s_ptr = reinterpret_cast<const uint4*>(scale);
    for (uint i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.num_threads()) {
        auto r   = r_ptr[i];
        auto s   = s_ptr[i];
        auto o   = ops.normvec(r, s, s_inv_mean);
        x_ptr[i] = o;
    }
}

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* in_out, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream)
{
    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);
    FT_CHECK(n_dims % PACK_DIM == 0);
    const int n_pack    = n_dims / PACK_DIM;
    const int n_iter    = ((n_pack + 1023) / 1024);        // iterations when block size == 1024
    int       n_threads = (n_pack + n_iter - 1) / n_iter;  // adjust block size to avoid tail effect
    n_threads           = (n_threads + 31) / 32 * 32;      // round up to the nearest multiple of warp size

    fusedAddBiasResidualNorm<<<batch_size, n_threads, 0, stream>>>(
        residual, in_out, bias, scale, eps, batch_size, n_dims);
}

template void
invokeFusedAddBiasResidualRMSNorm(float*, float*, const float*, const float*, float, int, int, cudaStream_t);
template void invokeFusedAddBiasResidualRMSNorm(half*, half*, const half*, const half*, float, int, int, cudaStream_t);

}  // namespace turbomind
