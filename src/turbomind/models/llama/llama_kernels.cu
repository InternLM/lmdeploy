// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/turbomind/kernels/decoder_multihead_attention/array_ops.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <type_traits>

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
#ifdef ENABLE_BF16
template void invokeRootMeanSquareNorm(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int, int, cudaStream_t);
#endif

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
#ifdef ENABLE_BF16
template void invokeCreateCausalMasks(__nv_bfloat16* mask, const int*, const int*, int, int, int, cudaStream_t);
#endif

template<typename Ti, typename To>
struct ExtendKvCache {

    static constexpr int MaxElemSize = std::max(sizeof(Ti), sizeof(To));
    static constexpr int X_ELEMS     = 16 / MaxElemSize;

    using Vi = Array<Ti, X_ELEMS>;
    using Vo = Array<To, X_ELEMS>;

    using Transform = ConvertKvCache<Ti, To>;

    struct Params {
        To**       k_dst_ptrs;
        To**       v_dst_ptrs;
        const Ti*  k_src;
        const Ti*  v_src;
        const int* cu_block_counts;
        const int* query_length;
        const int* context_length;
        int        block_length;
        size_t     dst_layer_offset;
        int        max_q_len;
        int        head_num;
        int        head_dim;
        Transform  transform_k;
        Transform  transform_v;
    };

    __device__ void operator()(const Params& params) const
    {
        const int batch_id = blockIdx.y;

        const int query_len    = params.query_length[batch_id];
        const int history_len  = params.context_length[batch_id] - query_len;
        const int cu_block_cnt = params.cu_block_counts[batch_id];

        const int head_id = blockIdx.z;

        const int size_per_head_div_x = params.head_dim / X_ELEMS;
        const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
        const int head_size_id        = idx % size_per_head_div_x;
        const int seq_len_id          = idx / size_per_head_div_x;

        const int cache_block_index  = (seq_len_id + history_len) / params.block_length;
        const int cache_block_offset = (seq_len_id + history_len) % params.block_length;

        const auto k_val_src = params.k_src;
        const auto v_val_src = params.v_src;

        const auto k_val_dst = (params.k_dst_ptrs + cu_block_cnt)[cache_block_index] + params.dst_layer_offset;
        const auto v_val_dst = (params.v_dst_ptrs + cu_block_cnt)[cache_block_index] + params.dst_layer_offset;

        if (seq_len_id < query_len) {
            // [B, H, s, D/x] -> [H, S[t:t+s], D/x]
            const int64_t dst_idx = head_id * params.block_length * size_per_head_div_x +  // H
                                    cache_block_offset * size_per_head_div_x +             // s + offset
                                    head_size_id;                                          // D/x

            const int64_t src_idx = batch_id * params.head_num * params.max_q_len * size_per_head_div_x +  // B
                                    head_id * params.max_q_len * size_per_head_div_x +                     // H
                                    seq_len_id * size_per_head_div_x +                                     // s
                                    head_size_id;                                                          // D/x

            Vi k_vi;
            Vi v_vi;

            Ldg(k_vi, k_val_src + src_idx * X_ELEMS);
            Ldg(v_vi, v_val_src + src_idx * X_ELEMS);

            Vo k_vo = params.transform_k(k_vi);
            Vo v_vo = params.transform_v(v_vi);

            Store(k_val_dst + dst_idx * X_ELEMS, k_vo);
            Store(v_val_dst + dst_idx * X_ELEMS, v_vo);
        }
    }
};

namespace {

template<class Kernel, class Params>
__global__ void KernelWrapper(Params params)
{
    Kernel{}(params);
};

}  // namespace

template<typename T>
void invokeExtendKVCache(void**       k_dst_ptrs,
                         void**       v_dst_ptrs,
                         const T*     k_src,
                         const T*     v_src,
                         const int*   cu_block_counts,
                         const int*   query_length,
                         const int*   context_length,
                         int          batch_size,
                         int          block_length,
                         size_t       dst_layer_offset,
                         int          max_q_len,
                         int          head_dim,
                         int          head_num,
                         int          quant,
                         const float* kv_params,
                         cudaStream_t stream)
{
    constexpr int block_sz = 128;

    auto fn = [&](auto value) {
        using Tout   = decltype(value);
        using Kernel = ExtendKvCache<T, Tout>;

        dim3 grid((max_q_len * head_dim / Kernel::X_ELEMS + block_sz - 1) / block_sz, batch_size, head_num);

        typename Kernel::Params params{(Tout**)k_dst_ptrs,
                                       (Tout**)v_dst_ptrs,
                                       k_src,
                                       v_src,
                                       cu_block_counts,
                                       query_length,
                                       context_length,
                                       block_length,
                                       dst_layer_offset,
                                       max_q_len,
                                       head_num,
                                       head_dim,
                                       {kv_params[0], kv_params[1]},
                                       {kv_params[2], kv_params[3]}};

        KernelWrapper<Kernel><<<grid, block_sz, 0, stream>>>(params);
    };

    (quant & QuantPolicy::kCacheKVInt8) ? fn(int8_t{}) : fn(T{});
}

template void invokeExtendKVCache(void**       k_dst_ptrs,
                                  void**       v_dst_ptrs,
                                  const float* k_src,
                                  const float* v_src,
                                  const int*   cu_block_counts,
                                  const int*   query_length,
                                  const int*   history_length,
                                  int          batch_size,
                                  int          block_length,
                                  size_t       dst_layer_offset,
                                  int          max_q_len,
                                  int          head_dim,
                                  int          head_num,
                                  int          quant,
                                  const float* kv_scale,
                                  cudaStream_t stream);

template void invokeExtendKVCache(void**       k_dst_ptrs,
                                  void**       v_dst_ptrs,
                                  const half*  k_src,
                                  const half*  v_src,
                                  const int*   cu_block_counts,
                                  const int*   query_length,
                                  const int*   history_length,
                                  int          batch_size,
                                  int          block_length,
                                  size_t       dst_layer_offset,
                                  int          max_q_len,
                                  int          head_dim,
                                  int          head_num,
                                  int          quant,
                                  const float* kv_scale,
                                  cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeExtendKVCache(void**       k_dst_ptrs,
                                  void**       v_dst_ptrs,
                                  const __nv_bfloat16*  k_src,
                                  const __nv_bfloat16*  v_src,
                                  const int*   cu_block_counts,
                                  const int*   query_length,
                                  const int*   history_length,
                                  int          batch_size,
                                  int          block_length,
                                  size_t       dst_layer_offset,
                                  int          max_q_len,
                                  int          head_dim,
                                  int          head_num,
                                  int          quant,
                                  const float* kv_scale,
                                  cudaStream_t stream);
#endif

template<typename Ti, typename To>
struct TransposeKvCache {
    static constexpr int MaxElemSize = std::max(sizeof(Ti), sizeof(To));
    static constexpr int X_ELEMS     = 16 / MaxElemSize;

    using Vi = Array<Ti, X_ELEMS>;
    using Vo = Array<To, X_ELEMS>;

    using Transform = ConvertKvCache<Ti, To>;

    struct Params {
        To*        k_dst;
        To*        v_dst;
        const Ti** k_src;
        const Ti** v_src;
        size_t     src_offset;
        int        head_num;
        int        head_n_rep;
        int        size_per_head;
        const int* seq_length;
        int        max_kv_len;
        int        max_seq_len;
        Transform  transform_k;
        Transform  transform_v;
        // float      k_scale;
        // float      k_zp;
        // float      v_scale;
        // float      v_zp;
    };

    __device__ void operator()(const Params& params) const
    {
        const int batch_id = blockIdx.y;
        const int head_id  = blockIdx.z;

        const int idx                 = blockIdx.x * blockDim.x + threadIdx.x;
        const int size_per_head_div_x = params.size_per_head / X_ELEMS;

        const auto k_src = params.k_src[batch_id] + params.src_offset;
        const auto v_src = params.v_src[batch_id] + params.src_offset;
        const auto k_dst = params.k_dst;
        const auto v_dst = params.v_dst;

        const auto seq_len = params.seq_length[batch_id];

        const int v_head_size_id = idx % size_per_head_div_x;
        const int v_seq_len_id   = idx / size_per_head_div_x;

        if (v_seq_len_id < seq_len) {
            // [B, H, s, D/x] <- [B, H, S[:s], D/x]
            const int64_t src_idx = head_id / params.head_n_rep * size_per_head_div_x * params.max_seq_len +  // H
                                    v_seq_len_id * size_per_head_div_x +                                      // s
                                    v_head_size_id;                                                           // D/x

            const int64_t dst_idx = batch_id * params.head_num * size_per_head_div_x * params.max_kv_len +  // B
                                    head_id * size_per_head_div_x * params.max_kv_len +                     // H
                                    v_seq_len_id * size_per_head_div_x +                                    // s
                                    v_head_size_id;                                                         // D/x

            Vi k_vi;
            Vi v_vi;

            Ldg(k_vi, k_src + src_idx * X_ELEMS);
            Ldg(v_vi, v_src + src_idx * X_ELEMS);

            Vo k_vo = params.transform_k(k_vi);
            Vo v_vo = params.transform_v(v_vi);

            Store(k_dst + dst_idx * X_ELEMS, k_vo);
            Store(v_dst + dst_idx * X_ELEMS, v_vo);
        }
    }
};

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
                            const float* kv_params)
{
    constexpr int block_sz = 128;

    auto fn = [&](auto value) {
        using Tin    = decltype(value);
        using Kernel = TransposeKvCache<Tin, T>;

        dim3 grid((max_kv_len * size_per_head / Kernel::X_ELEMS + block_sz - 1) / block_sz, batch_size, head_num);

        typename Kernel::Params params{key_cache_trans,
                                       val_cache_trans,
                                       (const Tin**)key_cache,
                                       (const Tin**)val_cache,
                                       src_offset,
                                       head_num,
                                       head_n_rep,
                                       size_per_head,
                                       key_length,
                                       max_kv_len,
                                       max_seq_len,
                                       {kv_params[0], kv_params[1]},
                                       {kv_params[2], kv_params[3]}};

        KernelWrapper<Kernel><<<grid, block_sz, 0, stream>>>(params);
    };

    (quant & QuantPolicy::kCacheKVInt8) ? fn(int8_t{}) : fn(T{});
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
#ifdef ENABLE_BF16
template void invokeTransposeKVCache(__nv_bfloat16*,
                                     __nv_bfloat16*,
                                     const __nv_bfloat16**,
                                     const __nv_bfloat16**,
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
#endif

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
    int block_size = 128;
    int grid_size  = batch_size;
    gatherOutput<<<grid_size, block_size, 0, stream>>>(
        output_ids, ids, context_length, max_context_len, max_gen_step, max_output_len, batch_size);
}

__global__ void updateOutput(int**      request_output_ids_ptrs,
                             int**      request_seqlen_ptrs,
                             const int* output_ids,
                             const int* sequence_lengths,
                             const int* request_output_ids_lens,
                             int        max_session_len,
                             bool       token_generated)
{
    const int batch_id = blockIdx.x;

    auto request_output_ids = request_output_ids_ptrs[batch_id];
    auto request_seqlen     = request_seqlen_ptrs[batch_id];

    output_ids += max_session_len * batch_id;

    const int seqlen     = sequence_lengths[batch_id] + (int)token_generated;
    const int output_len = min(seqlen, request_output_ids_lens[batch_id]);

    for (int i = threadIdx.x; i < output_len; i += blockDim.x) {
        request_output_ids[i] = output_ids[i];
    }

    *request_seqlen = seqlen;
}

void invokeUpdateOutput(int**        request_output_ids_ptrs,
                        int**        request_seqlen_ptrs,
                        const int*   output_ids,
                        const int*   sequence_lengths,
                        const int*   request_output_ids_lens,
                        int          max_session_len,
                        bool         token_generated,
                        int          batch_size,
                        cudaStream_t stream)
{
    constexpr int block_size = 128;
    const int     grid_size  = batch_size;

    updateOutput<<<grid_size, block_size, 0, stream>>>(request_output_ids_ptrs,
                                                       request_seqlen_ptrs,
                                                       output_ids,
                                                       sequence_lengths,
                                                       request_output_ids_lens,
                                                       max_session_len,
                                                       token_generated);
}

template<int BLOCK_DIM>
__global__ void compactOutputIds(
    int* cu_output_ids, const int* output_ids, const int* sequence_lengths, int session_len, bool token_generated)
{
    typedef cub::BlockReduce<int, BLOCK_DIM>     BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int batch_idx = blockIdx.x;

    int end   = (batch_idx + BLOCK_DIM - 1) / BLOCK_DIM * BLOCK_DIM;  // align to BLOCK_DIM boundary
    int count = 0;
    for (int i = threadIdx.x; i < end; i += blockDim.x) {
        int x = threadIdx.x < batch_idx ? sequence_lengths[threadIdx.x] : 0;
        count += BlockReduce(temp_storage).Sum(x);
        // https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html
        __syncthreads();
    }

    __shared__ int offset;

    if (threadIdx.x == 0) {
        offset = count;
    }

    __syncthreads();

    auto dst = cu_output_ids + offset;

    const int seq_len = sequence_lengths[batch_idx];

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        dst[i] = output_ids[batch_idx * session_len + i];
    }
}

void invokeCompactOutputIds(int*         cu_output_ids,
                            const int*   output_ids,
                            const int*   sequence_lengths,
                            int          max_session_len,
                            bool         token_generated,
                            int          batch_size,
                            cudaStream_t stream)
{
    constexpr int BLOCK_DIM = 128;
    compactOutputIds<BLOCK_DIM><<<batch_size, BLOCK_DIM, 0, stream>>>(
        cu_output_ids, output_ids, sequence_lengths, max_session_len, token_generated);
}

template<int N, int C>
struct IndexedCopyParam {
    Array<void*, N> src_ptr;
    Array<void*, N> dst_ptr;
    Array<int, N>   stride;
    Array<int, C>   src_idx;
    Array<int, C>   dst_idx;
    int             max_stride;
};

template<class T, int N, int C>
__global__ void indexedCopy(IndexedCopyParam<N, C> param)
{
    const int bi = blockIdx.x;
    const int si = param.src_idx[bi];
    const int di = param.dst_idx[bi];
    for (int i = threadIdx.x; i < param.max_stride; i += blockDim.x) {
        PRAGMA_UNROLL
        for (int k = 0; k < N; ++k) {
            if (i < param.stride[k]) {
                *((T*)param.dst_ptr[k] + param.stride[k] * di + i) =
                    *((const T*)param.src_ptr[k] + param.stride[k] * si + i);
            }
        }
    }
}

template<class T, int N>
void invokeIndexedCopyImpl(void**       h_src_ptr,
                           void**       h_dst_ptr,
                           const int*   h_elem_sz,
                           const int*   h_src_idx,
                           const int*   h_dst_idx,
                           int          count,
                           cudaStream_t st)
{
    auto invoke = [&](auto max_count) {
        constexpr int C = decltype(max_count)::value;
        // maximum parameter size: sm<70: 4kB, sm>=70: 32kB
        static_assert(sizeof(IndexedCopyParam<N, C>) <= 4096);
        IndexedCopyParam<N, C> param{};
        std::copy_n(h_src_ptr, N, param.src_ptr.data());
        std::copy_n(h_dst_ptr, N, param.dst_ptr.data());
        std::transform(h_elem_sz, h_elem_sz + N, param.stride.data(), [](int size) {
            // Basic alignment check
            FT_CHECK_WITH_INFO(size % sizeof(T) == 0, fmtstr("misalignment: %d %% %d", size, (int)sizeof(T)));
            return size / sizeof(T);
        });
        param.max_stride = *std::max_element(param.stride.begin(), param.stride.end());
        auto copy_idx    = [](const int* src, int offset, int n, auto dst) {
            return src ? (void)std::copy_n(src + offset, n, dst) : std::iota(dst, dst + n, offset);
        };
        for (int c = 0; c < count; c += C) {
            int batch_size = std::min(count - c, C);
            copy_idx(h_src_idx, c, batch_size, param.src_idx.data());
            copy_idx(h_dst_idx, c, batch_size, param.dst_idx.data());
            indexedCopy<T><<<batch_size, 128, 0, st>>>(param);
        }
    };
    if (count <= 4) {
        invoke(std::integral_constant<int, 4>{});
    }
    if (count <= 8) {
        invoke(std::integral_constant<int, 8>{});
    }
    else if (count <= 16) {
        invoke(std::integral_constant<int, 16>{});
    }
    else if (count <= 32) {
        invoke(std::integral_constant<int, 32>{});
    }
    else if (count <= 64) {
        invoke(std::integral_constant<int, 64>{});
    }
    else if (count <= 128) {
        invoke(std::integral_constant<int, 128>{});
    }
    else {
        invoke(std::integral_constant<int, 256>{});
    }
}

void invokeIndexedCopy(void**       h_src_ptr,
                       void**       h_dst_ptr,
                       const int*   h_elem_sz,
                       const int*   h_src_idx,
                       const int*   h_dst_idx,
                       int          count,
                       int          n_copys,
                       cudaStream_t st)
{
    auto args = std::tuple{h_src_ptr, h_dst_ptr, h_elem_sz, h_src_idx, h_dst_idx, count, st};
    switch (n_copys) {
        case 1:
            return std::apply(invokeIndexedCopyImpl<uint32_t, 1>, args);
        case 2:
            return std::apply(invokeIndexedCopyImpl<uint32_t, 2>, args);
        case 3:
            return std::apply(invokeIndexedCopyImpl<uint32_t, 3>, args);
        case 4:
            return std::apply(invokeIndexedCopyImpl<uint32_t, 4>, args);
        default:
            FT_CHECK(0);
    }
}

__global__ void padLastTokenIds(int* token_ids, const int* context_length, int max_context_len, int batch_size)
{
    for (int bi = threadIdx.x; bi < batch_size; bi += blockDim.x) {
        token_ids[(max_context_len - 1) * batch_size + bi] = token_ids[(context_length[bi] - 1) * batch_size + bi];
    }
}

void invokePadLastTokenIds(
    int* token_ids, const int* context_length, int max_context_len, int batch_size, cudaStream_t stream)
{
    padLastTokenIds<<<1, 512, 0, stream>>>(token_ids, context_length, max_context_len, batch_size);
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
    op_version_ = std::is_same<float, typename std::decay<T>::type>::value ? 1 : 2;
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
#ifdef ENABLE_BF16
template class FlashAttentionOp<__nv_bfloat16>;
#endif

}  // namespace turbomind
