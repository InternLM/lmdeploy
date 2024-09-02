// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

namespace turbomind {

template<int N>
struct QuantTypeConverter {
    using Type = short4;
};

template<>
struct QuantTypeConverter<8> {
    using Type = short4;
};

template<>
struct QuantTypeConverter<4> {
    using Type = short2;
};

template<typename T>
struct res_norm_ops_t {
};

template<typename T, typename T_Q>
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
    __device__ T absmax(const uint4& x) const
    {
        auto v1 = f.max(f.max(f.abs(x.x)), f.max(f.abs(x.y)));
        auto v2 = f.max(f.max(f.abs(x.z)), f.max(f.abs(x.w)));
        auto v  = f.max(v1, v2);
        return v;
    }
    __device__ T_Q quant(const uint4& x, float scale) const
    {
        return f.quant(x, scale);
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
    __device__ half2 abs(const uint& x) const
    {
        uint t = const_cast<uint&>(x);
        return cuda_abs(reinterpret_cast<half2&>(t));
    }
    __device__ half max(const half2& x) const
    {
        return (x.x > x.y) ? x.x : x.y;
    }
    __device__ half max(const half& x, const half& y) const
    {
        return (x > y) ? x : y;
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
    __device__ float2 mul(const uint& x, float scale) const
    {
        float2 res = cast(x);
        res.x *= scale;
        res.y *= scale;
        return res;
    }
    __device__ short4 quant(const uint4& x, float scale) const
    {
        short4 res;
        res.x = cuda_cast<int16_t, float2>(mul(x.x, scale));
        res.y = cuda_cast<int16_t, float2>(mul(x.y, scale));
        res.z = cuda_cast<int16_t, float2>(mul(x.z, scale));
        res.w = cuda_cast<int16_t, float2>(mul(x.w, scale));
        return res;
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
    __device__ float abs(const uint& x) const
    {
        uint t = const_cast<uint&>(x);
        return cuda_abs(reinterpret_cast<float&>(t));
    }
    // for generality
    __device__ float max(const float& x) const
    {
        return x;
    }
    __device__ float max(const float& x, const float& y) const
    {
        return (x > y) ? x : y;
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
    __device__ float2 mul(const uint& x, const uint& y, float scale) const
    {
        float2 res;
        res.x = cast(x) * scale;
        res.y = cast(y) * scale;
        return res;
    }
    __device__ short2 quant(const uint4& x, float scale) const
    {
        short2 res;
        res.x = cuda_cast<int16_t, float2>(mul(x.x, x.y, scale));
        res.y = cuda_cast<int16_t, float2>(mul(x.z, x.w, scale));
        return res;
    }
};

#ifdef ENABLE_BF16
template<>
struct res_norm_ops_t<__nv_bfloat16> {
    __device__ float2 cast(const uint& x) const
    {
        return cuda_cast<float2, __nv_bfloat162>(reinterpret_cast<const __nv_bfloat162&>(x));
    }
    __device__ uint cast(const float2& x) const
    {
        auto y = cuda_cast<__nv_bfloat162, float2>(x);
        return reinterpret_cast<uint&>(y);
    }
    __device__ __nv_bfloat162 abs(const uint& x) const
    {
        uint t = const_cast<uint&>(x);
        return cuda_abs(reinterpret_cast<__nv_bfloat162&>(t));
    }
    __device__ __nv_bfloat16 max(const __nv_bfloat162& x) const
    {
        return (x.x > x.y) ? x.x : x.y;
    }
    __device__ __nv_bfloat16 max(const __nv_bfloat16& x, const __nv_bfloat16& y) const
    {
        return (x > y) ? x : y;
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
    __device__ float2 mul(const uint& x, float scale) const
    {
        float2 res = cast(x);
        res.x *= scale;
        res.y *= scale;
        return res;
    }
    __device__ short4 quant(const uint4& x, float scale) const
    {
        short4 res;
        res.x = cuda_cast<int16_t, float2>(mul(x.x, scale));
        res.y = cuda_cast<int16_t, float2>(mul(x.y, scale));
        res.z = cuda_cast<int16_t, float2>(mul(x.z, scale));
        res.w = cuda_cast<int16_t, float2>(mul(x.w, scale));
        return res;
    }
};

#endif

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
__device__ T blockReduceMax(const cg::thread_block& block, T value)
{
    __shared__ float partial[32];

    auto tile = cg::tiled_partition<32>(block);
    value     = cg::reduce(tile, value, cg::greater<float>{});

    if (tile.thread_rank() == 0) {
        partial[tile.meta_group_rank()] = value;
    }

    block.sync();

    value = tile.thread_rank() < tile.meta_group_size() ? partial[tile.thread_rank()] : T(-1e20f);
    return cg::reduce(tile, value, cg::greater<float>{});
}

// r' = r + x
// x' = norm(r') * scales
template<typename T, bool enable_quant>
__global__ void fusedAddBiasResidualNorm(T* __restrict__ r_data,
                                         T* __restrict__ x_data,
                                         const T* __restrict__ bias,
                                         const T* __restrict__ scale,
                                         int8_t* __restrict__ quant_out,
                                         float* __restrict__ quant_scale,
                                         float eps,
                                         int   batch_size,
                                         int   n_dims)
{
    auto block = cg::this_thread_block();
    auto grid  = cg::this_grid();

    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);

    const auto batch_idx            = block.group_index().x;
    uint4* __restrict__ r_ptr       = reinterpret_cast<uint4*>(r_data + batch_idx * n_dims);
    uint4* __restrict__ x_ptr       = reinterpret_cast<uint4*>(x_data + batch_idx * n_dims);
    const uint4* __restrict__ b_ptr = reinterpret_cast<const uint4*>(bias);
    using T_Q                       = typename QuantTypeConverter<PACK_DIM>::Type;
    T_Q* __restrict__ q_ptr         = reinterpret_cast<T_Q*>(quant_out + batch_idx * n_dims);

    res_norm_t<T, T_Q> ops;

    float thread_sum{};
    for (auto i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
        auto  r  = r_ptr[i];
        auto  x  = x_ptr[i];
        uint4 b  = b_ptr ? b_ptr[i] : uint4{};
        r        = ops.addvec(r, x, b, thread_sum);
        r_ptr[i] = r;
    }

    auto total_sum = blockReduceSum(block, thread_sum);

    float s_inv_mean = rsqrt(total_sum / n_dims + eps);

    const uint4* __restrict__ s_ptr = reinterpret_cast<const uint4*>(scale);
    if constexpr (enable_quant) {
        float thread_max{};
        for (uint i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
            auto r     = r_ptr[i];
            auto s     = s_ptr[i];
            auto o     = ops.normvec(r, s, s_inv_mean);
            x_ptr[i]   = o;
            thread_max = cuda_max(thread_max, cuda_cast<float>(ops.absmax(o)));
        }
        auto total_max = blockReduceMax(block, thread_max);
        if (block.thread_rank() == 0) {
            quant_scale[batch_idx] = total_max / 127.0f;
        }
        const float tmp_scale = 127.0f / total_max;
        for (uint i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
            auto x   = x_ptr[i];
            auto q   = ops.quant(x, tmp_scale);
            q_ptr[i] = q;
        }
    }
    else {
        for (uint i = block.thread_rank(); i < n_dims / PACK_DIM; i += block.size()) {
            auto r   = r_ptr[i];
            auto s   = s_ptr[i];
            auto o   = ops.normvec(r, s, s_inv_mean);
            x_ptr[i] = o;
        }
    }
}

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(T*           residual,
                                       T*           in_out,
                                       const T*     bias,
                                       const T*     scale,
                                       int8_t*      quant_out,
                                       float*       quant_scale,
                                       float        eps,
                                       int          batch_size,
                                       int          n_dims,
                                       cudaStream_t stream)
{
    constexpr int PACK_DIM = sizeof(uint4) / sizeof(T);
    FT_CHECK(n_dims % PACK_DIM == 0);
    const int n_pack    = n_dims / PACK_DIM;
    const int n_iter    = ((n_pack + 1023) / 1024);        // iterations when block size == 1024
    int       n_threads = (n_pack + n_iter - 1) / n_iter;  // adjust block size to avoid tail effect
    n_threads           = (n_threads + 31) / 32 * 32;      // round up to the nearest multiple of warp size

    if (quant_out == nullptr) {
        fusedAddBiasResidualNorm<T, false><<<batch_size, n_threads, 0, stream>>>(
            residual, in_out, bias, scale, quant_out, quant_scale, eps, batch_size, n_dims);
    }
    else {
        fusedAddBiasResidualNorm<T, true><<<batch_size, n_threads, 0, stream>>>(
            residual, in_out, bias, scale, quant_out, quant_scale, eps, batch_size, n_dims);
    }
    sync_check_cuda_error();
}

template<typename T>
__global__ void maskOutput(T* output, const int* mask, int dim)
{
    int batch_idx = blockIdx.x;
    output += dim * batch_idx;
    int masked = mask[batch_idx];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = (masked) ? output[i] : T();
    }
}

template<typename T>
void invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream)
{
    maskOutput<<<batch_size, 1024, 0, stream>>>(output, mask, dim);
}

#ifdef ENABLE_FP32
template void invokeFusedAddBiasResidualRMSNorm(
    float*, float*, const float*, const float*, int8_t*, float*, float, int, int, cudaStream_t);
template void invokeMask(float* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
#endif
template void invokeFusedAddBiasResidualRMSNorm(
    half*, half*, const half*, const half*, int8_t*, float*, float, int, int, cudaStream_t);
template void invokeMask(half* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeFusedAddBiasResidualRMSNorm(__nv_bfloat16*,
                                                __nv_bfloat16*,
                                                const __nv_bfloat16*,
                                                const __nv_bfloat16*,
                                                int8_t*,
                                                float*,
                                                float,
                                                int,
                                                int,
                                                cudaStream_t);
template void invokeMask(__nv_bfloat16* output, const int* mask, int batch_size, int dim, cudaStream_t stream);
#endif
}  // namespace turbomind
