// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"

namespace turbomind {

// fp16, bf16
// n is divided by 2 for this impl
template<typename T>
__global__ void rootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n)
{
    using T2 = typename TypeConverter<T>::Type;
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    T2*       out_ptr   = (T2*)out;
    const T2* input_ptr = (const T2*)input;
    const T2* scale_ptr = (const T2*)scale;

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float2 tmp2 = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
        mean += tmp2.x * tmp2.x;
        mean += tmp2.y * tmp2.y;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(.5f * mean / (float)n + eps);
    }
    __syncthreads();

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float2 tmp2                   = cuda_cast<float2>(input_ptr[blockIdx.x * n + idx]);
        float2 sca2                   = cuda_cast<float2>(scale_ptr[idx]);
        tmp2.x                        = tmp2.x * s_inv_mean * sca2.x;
        tmp2.y                        = tmp2.y * s_inv_mean * sca2.y;
        out_ptr[blockIdx.x * n + idx] = cuda_cast<T2>(tmp2);
    }
}

template<>
__global__ void rootMeanSquareNorm(float* out, const float* input, const float* scale, float eps, int m, int n)
{
    __shared__ float s_inv_mean;
    float            mean = 0.f;

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float tmp = input[blockIdx.x * n + idx];
        mean += tmp * tmp;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_inv_mean = rsqrt(mean / static_cast<float>(n) + eps);
    }
    __syncthreads();

    for (uint idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float tmp                 = input[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] = tmp * s_inv_mean * scale[idx];
    }
}

template<typename T>
void invokeRootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream)
{
    if (sizeof(T) == 2) {
        FT_CHECK(n % 2 == 0);
        n /= 2;
    }
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    rootMeanSquareNorm<<<grid, block, 0, stream>>>(out, input, scale, eps, m, n);
}

template void invokeRootMeanSquareNorm(float*, const float*, const float*, float, int, int, cudaStream_t);
template void invokeRootMeanSquareNorm(half*, const half*, const half*, float, int, int, cudaStream_t);

// #ifdef ENABLE_BF16

// template void invokeRootMeanSquareNorm(__nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);

// #endif

template<typename T, typename T0>
__device__ T saturate_cast(T0 x)
{
    return x;
}

template<>
__device__ half saturate_cast<half, float>(float x)
{
    return (x > 64512.f || x < -64512.f) ? (x > 0.f ? 64512.f : -64512.f) : x;
}

template<typename T>
__global__ void addResidual(T* out, const T* in, size_t n)
{
    auto idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = static_cast<T>(static_cast<float>(out[idx]) + static_cast<float>(in[idx]));
    }
}

template<typename T>
void invokeAddResidual(T* out, const T* in, int m, int n, cudaStream_t stream)
{
    auto total = static_cast<size_t>(m) * n;
    dim3 block(std::min((unsigned long)total, 1024UL));
    dim3 grid((total + block.x - 1) / block.x);

    addResidual<<<grid, block, 0, stream>>>(out, in, total);
}

template void invokeAddResidual(float*, const float*, int, int, cudaStream_t);
template void invokeAddResidual(half*, const half*, int, int, cudaStream_t);

// ids [seq_len, batch_size]
// input_ids [batch_size, max_input_len]
__global__ void
fixInputIds(int* ids, const int* input_ids, const int* input_lengths, int batch_size, int seq_len, int max_input_len)
{
    int seq_id   = threadIdx.x;
    int batch_id = blockIdx.x;
    for (; seq_id < input_lengths[batch_id]; seq_id += blockDim.x) {
        ids[seq_id * batch_size + batch_id] = input_ids[batch_id * max_input_len + seq_id];
    }
}

void invokeFixInputIds(int*         ids,
                       const int*   input_ids,
                       const int*   input_lengths,
                       int          batch_size,
                       int          seq_len,
                       int          max_input_len,
                       cudaStream_t st)
{
    dim3 block(std::min(1024, max_input_len));
    dim3 grid(batch_size);
    fixInputIds<<<grid, block, 0, st>>>(ids, input_ids, input_lengths, batch_size, seq_len, max_input_len);
}

template<typename T>
__global__ void sliceCausalMask(T* mask, int seq_len, int key_len, int step)
{
    mask += (size_t)blockIdx.x * seq_len * key_len;
    for (int i = threadIdx.x; i < seq_len * key_len; i += blockDim.x) {
        int row = i / key_len;
        int col = i % key_len;
        if (col <= row + step) {
            mask[i] = static_cast<T>(1.f);
        }
        else {
            mask[i] = static_cast<T>(0.f);
        }
    }
}

// [step: step+Q, :] of the K*K causal mask
template<typename T>
void invokeSliceCausalMask(T* mask, int seq_len, int key_len, int step, int batch_size, cudaStream_t stream)
{
    FT_CHECK(step == key_len - seq_len);
    sliceCausalMask<<<batch_size, 256, 0, stream>>>(mask, seq_len, key_len, step);
}

template void invokeSliceCausalMask(half*, int, int, int, int, cudaStream_t);
template void invokeSliceCausalMask(float*, int, int, int, int, cudaStream_t);

// mask [bsz, max_q_len, max_k_len]

template<typename T>
__global__ void createCausalMasks(T* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len)
{
    const auto q_len = q_lens[blockIdx.x];
    const auto k_len = k_lens[blockIdx.x];
    mask += blockIdx.x * max_q_len * max_k_len;
    for (int i = threadIdx.x; i < max_q_len * max_k_len; i += blockDim.x) {
        const int q        = i / max_k_len;  // [0, max_q_len)
        const int k        = i % max_k_len;  // [0, max_k_len)
        bool      is_valid = q < q_len && k < k_len && k <= q + (k_len - q_len);
        mask[i]            = static_cast<T>(is_valid);
    }
}

template<typename T>
void invokeCreateCausalMasks(
    T* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len, int batch_size, cudaStream_t stream)
{
    createCausalMasks<<<batch_size, 512, 0, stream>>>(mask, q_lens, k_lens, max_q_len, max_k_len);
}

template void invokeCreateCausalMasks(float* mask, const int*, const int*, int, int, int, cudaStream_t);
template void invokeCreateCausalMasks(half* mask, const int*, const int*, int, int, int, cudaStream_t);

template<typename T>
__global__ void extend_key_cache(T**          k_dst,
                                 const size_t dst_offset,
                                 const T*     k_src,
                                 const int    head_num,
                                 const int    size_per_head,
                                 const int*   query_length,
                                 const int*   history_length,
                                 const int    max_q_len,
                                 const int    max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;

    // x dim is now handled by uint4 type
    const auto key_src = reinterpret_cast<const uint4*>(k_src);
    const auto key_dst = reinterpret_cast<uint4*>(k_dst[batch_id] + dst_offset);

    const auto seq_len  = query_length[batch_id];
    const auto t_offset = history_length[batch_id];

    const int k_head_size_id = idx % size_per_head_div_x;
    const int k_seq_len_id   = idx / size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        // [B, H, s, D/x] -> [H, D/x, S[t:t+s]]

        const int64_t dst_idx = head_id * size_per_head_div_x * max_seq_len +  // H
                                k_head_size_id * max_seq_len +                 // D/x
                                t_offset + k_seq_len_id;                       // s + offset

        const int64_t src_idx = batch_id * head_num * size_per_head_div_x * max_q_len +  // B
                                head_id * size_per_head_div_x * max_q_len +              // H
                                k_seq_len_id * size_per_head_div_x +                     // s
                                k_head_size_id;                                          // D/x

        key_dst[dst_idx] = key_src[src_idx];
    }
}

template<typename T>
__global__ void extend_value_cache(T**          v_dst,
                                   const size_t dst_offset,
                                   const T*     v_src,
                                   const int    head_num,
                                   const int    size_per_head,
                                   const int*   query_length,
                                   const int*   history_length,
                                   const int    max_q_len,
                                   const int    max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;

    // x dim is now handled by uint4 type
    const auto val_src = reinterpret_cast<const uint4*>(v_src);
    const auto val_dst = reinterpret_cast<uint4*>(v_dst[batch_id] + dst_offset);

    const auto seq_len  = query_length[batch_id];
    const auto t_offset = history_length[batch_id];

    const int v_head_size_id = idx % size_per_head_div_x;
    const int v_seq_len_id   = idx / size_per_head_div_x;

    if (v_seq_len_id < seq_len) {
        // [B, H, s, D/x] -> [H, S[t:t+s], D/x]
        const int64_t dst_idx = head_id * size_per_head_div_x * max_seq_len +      // H
                                (v_seq_len_id + t_offset) * size_per_head_div_x +  // s + offset
                                v_head_size_id;                                    // D/x

        const int64_t src_idx = batch_id * head_num * size_per_head_div_x * max_q_len +  // B
                                head_id * size_per_head_div_x * max_q_len +              // H
                                v_seq_len_id * size_per_head_div_x +                     // s
                                v_head_size_id;                                          // D/x

        val_dst[dst_idx] = val_src[src_idx];
    }
}

inline __device__ float2 float2div(float a, float2 b)
{
    float2 c;
    c.x = b.x / a;
    c.y = b.y / a;
    return c;
}

inline __device__ float2 float2sub(float zp, float2 val)
{
    float2 ret;
    ret.x = val.x - zp;
    ret.y = val.y - zp;
    return ret;
}

static inline __device__ half4 char4_scale_to_half4(char4 value, const float scale, const float zp)
{
    half4 dst;
    dst.x = __float2half(value.x * scale + zp);
    dst.y = __float2half(value.y * scale + zp);
    dst.z = __float2half(value.z * scale + zp);
    dst.w = __float2half(value.w * scale + zp);
    return dst;
}

static inline __device__ uint32_t float4_to_char4(float x, float y, float z, float w)
{
    uint32_t dst;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
    uint32_t a;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(a) : "f"(x));
    uint32_t b;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(b) : "f"(y));
    uint32_t c;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(c) : "f"(z));
    uint32_t d;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(d) : "f"(w));

    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2,  0;\n" : "=r"(dst) : "r"(d), "r"(c));
    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %0;\n" : "+r"(dst) : "r"(b), "r"(a));
#else
    char4 tmp;
    tmp.x       = x;
    tmp.y       = y;
    tmp.z       = z;
    tmp.w       = w;
    dst         = reinterpret_cast<const uint32_t&>(tmp);
#endif
    return dst;
}

template<typename T>
__global__ void extend_value_cache_int8(int8_t**     v_dst,
                                        const size_t dst_offset,
                                        const T*     v_src,
                                        const int    head_num,
                                        const int    size_per_head,
                                        const int*   query_length,
                                        const int*   history_length,
                                        const int    max_q_len,
                                        const int    max_seq_len,
                                        const float  v_scale,
                                        const float  v_zp)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;

    // x dim is now handled by uint4 type
    const auto val_src = reinterpret_cast<const uint4*>(v_src);
    const auto val_dst = reinterpret_cast<uint2*>(v_dst[batch_id] + dst_offset);

    const auto seq_len  = query_length[batch_id];
    const auto t_offset = history_length[batch_id];

    const int v_head_size_id = idx % size_per_head_div_x;
    const int v_seq_len_id   = idx / size_per_head_div_x;

    if (v_seq_len_id < seq_len) {
        // [B, H, s, D/x] -> [H, S[t:t+s], D/x]
        const int64_t dst_idx = head_id * size_per_head_div_x * max_seq_len +      // H
                                (v_seq_len_id + t_offset) * size_per_head_div_x +  // s + offset
                                v_head_size_id;                                    // D/x

        const int64_t src_idx = batch_id * head_num * size_per_head_div_x * max_q_len +  // B
                                head_id * size_per_head_div_x * max_q_len +              // H
                                v_seq_len_id * size_per_head_div_x +                     // s
                                v_head_size_id;                                          // D/x

        // scale to int8 and write
        const auto value  = val_src[src_idx];
        auto       to_ptr = reinterpret_cast<uint32_t*>(val_dst + dst_idx);

        float2 float2_0 = float2div(v_scale, float2sub(v_zp, mmha::half2_to_float2(value.x)));
        float2 float2_1 = float2div(v_scale, float2sub(v_zp, mmha::half2_to_float2(value.y)));
        to_ptr[0]       = float4_to_char4(float2_0.x, float2_0.y, float2_1.x, float2_1.y);

        float2_0  = float2div(v_scale, float2sub(v_zp, mmha::half2_to_float2(value.z)));
        float2_1  = float2div(v_scale, float2sub(v_zp, mmha::half2_to_float2(value.w)));
        to_ptr[1] = float4_to_char4(float2_0.x, float2_0.y, float2_1.x, float2_1.y);
    }
}

template<typename T>
void invokeExtendKVCache(T**          k_dst,
                         T**          v_dst,
                         size_t       dst_offset,
                         const T*     k_src,
                         const T*     v_src,
                         int          local_batch_size,
                         const int*   query_length,
                         int          max_q_len,
                         const int*   history_length,
                         int          max_seq_len,
                         int          size_per_head,
                         int          local_head_num,
                         cudaStream_t stream,
                         int          quant,
                         const float* kv_scale)
{
    constexpr int block_sz = 128;
    constexpr int x        = (sizeof(T) == 4) ? 4 : 8;

    dim3 grid((max_q_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    if (quant & QuantPolicy::kCacheKVInt8) {
        extend_value_cache_int8<<<grid, block_sz, 0, stream>>>(reinterpret_cast<int8_t**>(k_dst),
                                                               dst_offset,
                                                               k_src,
                                                               local_head_num,
                                                               size_per_head,
                                                               query_length,
                                                               history_length,
                                                               max_q_len,
                                                               max_seq_len,
                                                               kv_scale[0],
                                                               kv_scale[1]);

        extend_value_cache_int8<<<grid, block_sz, 0, stream>>>(reinterpret_cast<int8_t**>(v_dst),
                                                               dst_offset,
                                                               v_src,
                                                               local_head_num,
                                                               size_per_head,
                                                               query_length,
                                                               history_length,
                                                               max_q_len,
                                                               max_seq_len,
                                                               kv_scale[2],
                                                               kv_scale[3]);
    }
    else {
        extend_value_cache<<<grid, block_sz, 0, stream>>>(k_dst,
                                                          dst_offset,
                                                          k_src,
                                                          local_head_num,
                                                          size_per_head,
                                                          query_length,
                                                          history_length,
                                                          max_q_len,
                                                          max_seq_len);

        extend_value_cache<<<grid, block_sz, 0, stream>>>(v_dst,
                                                          dst_offset,
                                                          v_src,
                                                          local_head_num,
                                                          size_per_head,
                                                          query_length,
                                                          history_length,
                                                          max_q_len,
                                                          max_seq_len);
    }
}

template void invokeExtendKVCache(float**,
                                  float**,
                                  size_t,
                                  const float*,
                                  const float*,
                                  int,
                                  const int*,
                                  int,
                                  const int*,
                                  int,
                                  int,
                                  int,
                                  cudaStream_t stream,
                                  int,
                                  const float*);

template void invokeExtendKVCache(half**,
                                  half**,
                                  size_t,
                                  const half*,
                                  const half*,
                                  int,
                                  const int*,
                                  int,
                                  const int*,
                                  int,
                                  int,
                                  int,
                                  cudaStream_t stream,
                                  int,
                                  const float*);

template<typename T>
__global__ void transpose_value_cache(T*           v_dst,  //
                                      const T**    v_src,
                                      const size_t src_offset,
                                      const int    head_num,
                                      const int    head_n_rep,
                                      const int    size_per_head,
                                      const int*   seq_length,
                                      const int    max_kv_len,
                                      const int    max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;

    // x dim is now handled by uint4 type
    const auto val_src = reinterpret_cast<const uint4*>(v_src[batch_id] + src_offset);
    const auto val_dst = reinterpret_cast<uint4*>(v_dst);

    const auto seq_len = seq_length[batch_id];

    const int v_head_size_id = idx % size_per_head_div_x;
    const int v_seq_len_id   = idx / size_per_head_div_x;

    if (v_seq_len_id < seq_len) {
        // [B, H, s, D/x] <- [B, H, S[:s], D/x]
        const int64_t src_idx = head_id / head_n_rep * size_per_head_div_x * max_seq_len +  // H
                                v_seq_len_id * size_per_head_div_x +                        // s
                                v_head_size_id;                                             // D/x

        const int64_t dst_idx = batch_id * head_num * size_per_head_div_x * max_kv_len +  // B
                                head_id * size_per_head_div_x * max_kv_len +              // H
                                v_seq_len_id * size_per_head_div_x +                      // s
                                v_head_size_id;                                           // D/x

        val_dst[dst_idx] = val_src[src_idx];
    }
}

template<typename T>
__global__ void transpose_value_cache_int8(T*             v_dst,  //
                                           const int8_t** v_src,
                                           const size_t   src_offset,
                                           const int      head_num,
                                           const int      head_n_rep,
                                           const int      size_per_head,
                                           const int*     seq_length,
                                           const int      max_kv_len,
                                           const int      max_seq_len,
                                           const float    v_scale,
                                           const float    v_zp)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;

    // x dim is now handled by uint4 type
    const auto val_src = reinterpret_cast<const uint2*>(v_src[batch_id] + src_offset);
    const auto val_dst = reinterpret_cast<uint4*>(v_dst);

    const auto seq_len = seq_length[batch_id];

    const int v_head_size_id = idx % size_per_head_div_x;
    const int v_seq_len_id   = idx / size_per_head_div_x;

    if (v_seq_len_id < seq_len) {
        // [B, H, s, D/x] <- [B, H, S[:s], D/x]
        const int64_t src_idx = head_id / head_n_rep * size_per_head_div_x * max_seq_len +  // H
                                v_seq_len_id * size_per_head_div_x +                        // s
                                v_head_size_id;                                             // D/x

        const int64_t dst_idx = batch_id * head_num * size_per_head_div_x * max_kv_len +  // B
                                head_id * size_per_head_div_x * max_kv_len +              // H
                                v_seq_len_id * size_per_head_div_x +                      // s
                                v_head_size_id;                                           // D/x

        // int8x8 -> fp16x8
        const auto from_ptr = reinterpret_cast<const char4*>(val_src + src_idx);
        auto       to_ptr   = reinterpret_cast<half4*>(val_dst + dst_idx);

        to_ptr[0] = char4_scale_to_half4(from_ptr[0], v_scale, v_zp);
        to_ptr[1] = char4_scale_to_half4(from_ptr[1], v_scale, v_zp);
    }
}

template<typename T>
void invokeTransposeKVCache(T*           key_cache_trans,
                            T*           val_cache_trans,
                            const T**    key_cache,
                            const T**    val_cache,
                            size_t       src_offset,
                            int          batch_size,
                            const int*   key_length,
                            int          max_kv_len,
                            int          max_seq_len,
                            int          size_per_head,
                            int          head_num,
                            int          head_n_rep,
                            cudaStream_t stream,
                            int          quant,
                            const float* kv_scale)
{
    constexpr int block_sz = 128;
    constexpr int x        = (sizeof(T) == 4) ? 4 : 8;

    dim3 grid((max_kv_len * size_per_head / x + block_sz - 1) / block_sz, batch_size, head_num);

    if (quant & QuantPolicy::kCacheKVInt8) {
        transpose_value_cache_int8<<<grid, block_sz, 0, stream>>>(key_cache_trans,
                                                                  reinterpret_cast<const int8_t**>(key_cache),
                                                                  src_offset,
                                                                  head_num,
                                                                  head_n_rep,
                                                                  size_per_head,
                                                                  key_length,
                                                                  max_kv_len,
                                                                  max_seq_len,
                                                                  kv_scale[0],
                                                                  kv_scale[1]);

        transpose_value_cache_int8<<<grid, block_sz, 0, stream>>>(val_cache_trans,
                                                                  reinterpret_cast<const int8_t**>(val_cache),
                                                                  src_offset,
                                                                  head_num,
                                                                  head_n_rep,
                                                                  size_per_head,
                                                                  key_length,
                                                                  max_kv_len,
                                                                  max_seq_len,
                                                                  kv_scale[2],
                                                                  kv_scale[3]);
    }
    else {
        transpose_value_cache<<<grid, block_sz, 0, stream>>>(key_cache_trans,
                                                             key_cache,
                                                             src_offset,
                                                             head_num,
                                                             head_n_rep,
                                                             size_per_head,
                                                             key_length,
                                                             max_kv_len,
                                                             max_seq_len);

        transpose_value_cache<<<grid, block_sz, 0, stream>>>(val_cache_trans,
                                                             val_cache,
                                                             src_offset,
                                                             head_num,
                                                             head_n_rep,
                                                             size_per_head,
                                                             key_length,
                                                             max_kv_len,
                                                             max_seq_len);
    }
}

template void invokeTransposeKVCache(float*,
                                     float*,
                                     const float**,
                                     const float**,
                                     size_t,
                                     int,
                                     const int*,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     cudaStream_t stream,
                                     int,
                                     const float*);
template void invokeTransposeKVCache(half*,
                                     half*,
                                     const half**,
                                     const half**,
                                     size_t,
                                     int,
                                     const int*,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     cudaStream_t stream,
                                     int,
                                     const float*);

__global__ void gatherOutput(int*       output_ids,
                             const int* ids,
                             const int* context_length,
                             int        max_context_len,
                             int        max_gen_step,
                             int        max_output_len,
                             int        batch_size)
{
    const int batch_id    = blockIdx.x;
    const int context_len = context_length[batch_id];
    output_ids += batch_id * max_output_len;
    for (int src_idx = threadIdx.x; src_idx < max_gen_step; src_idx += blockDim.x) {
        // skip padding for src
        if (context_len <= src_idx && src_idx < max_context_len) {
            continue;
        }
        // skip padding for dst
        const int dst_idx   = src_idx < context_len ? src_idx : src_idx - (max_context_len - context_len);
        output_ids[dst_idx] = ids[src_idx * batch_size + batch_id];
    }
}

void invokeGatherOutput(int*         output_ids,
                        const int*   ids,
                        const int*   context_length,
                        int          max_context_len,
                        int          max_gen_step,
                        int          max_output_len,
                        int          batch_size,
                        cudaStream_t stream)
{
    int block_size = 512;
    int grid_size  = batch_size;
    gatherOutput<<<grid_size, block_size, 0, stream>>>(
        output_ids, ids, context_length, max_context_len, max_gen_step, max_output_len, batch_size);
}

#define VERSION_SWITCH(VERSION, CONST_NAME, ...)                                                                       \
    [&] {                                                                                                              \
        if (VERSION == 2) {                                                                                            \
            constexpr static int CONST_NAME = 2;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else {                                                                                                         \
            constexpr static int CONST_NAME = 1;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

template<typename T>
FlashAttentionOp<T>::FlashAttentionOp(int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
    batch_size_(batch_size), head_num_(head_num), key_len_(key_len), seq_len_(seq_len), size_per_head_(size_per_head)
{
#ifdef _MSC_VER
    op_version_ = 1;
#else
    op_version_ = std::is_same<half, typename std::decay<T>::type>::value ? 2 : 1;
    if (op_version_ == 2 && getSMVersion() < 80) {
        op_version_ = 1;
    }
#endif
}

template<typename T>
int FlashAttentionOp<T>::get_workspace_size() const
{
#ifdef _MSC_VER
    FlashAttentionOpImpl<T, 1> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
    return attention_op.get_workspace_size();
#else
    return VERSION_SWITCH(op_version_, OP_VERSION, [&]() {
        FlashAttentionOpImpl<T, OP_VERSION> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
        return attention_op.get_workspace_size();
    });
#endif
}

template<typename T>
void FlashAttentionOp<T>::operator()(Params& params, cudaStream_t st) const
{
#ifdef _MSC_VER
    FlashAttentionOpImpl<T, 1> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
    return attention_op(params, st);
#else
    return VERSION_SWITCH(op_version_, OP_VERSION, [&]() {
        FlashAttentionOpImpl<T, OP_VERSION> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
        return attention_op(params, st);
    });
#endif
}

template class FlashAttentionOp<float>;
template class FlashAttentionOp<half>;

}  // namespace turbomind
