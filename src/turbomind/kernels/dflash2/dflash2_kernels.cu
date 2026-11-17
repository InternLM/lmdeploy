// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/dflash2/dflash2_kernels.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include "src/turbomind/core/check.h"

namespace turbomind {

namespace {

template<typename T>
__device__ __forceinline__ float to_f(T x);
template<>
__device__ __forceinline__ float to_f<__half>(__half x)
{
    return __half2float(x);
}
template<>
__device__ __forceinline__ float to_f<float>(float x)
{
    return x;
}
template<>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

template<typename T>
__device__ __forceinline__ void from_f(T& dst, float x);
template<>
__device__ __forceinline__ void from_f<__half>(__half& dst, float x)
{
    dst = __float2half_rn(x);
}
template<>
__device__ __forceinline__ void from_f<float>(float& dst, float x)
{
    dst = x;
}
template<>
__device__ __forceinline__ void from_f<__nv_bfloat16>(__nv_bfloat16& dst, float x)
{
    dst = __float2bfloat16_rn(x);
}

template<typename T>
__global__ void grouped_conv_kernel(T* __restrict__ out,
                                    const T* __restrict__ hidden,
                                    const T* __restrict__ delta,
                                    const T* __restrict__ base,
                                    int rows,
                                    int H,
                                    int G,
                                    int group_size,
                                    int taps,
                                    int block_size)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * H;
    if (idx >= total) {
        return;
    }
    const int t = idx / H;
    const int h = idx - (long long)t * H;
    const int g = h / group_size;
    const int pos = t % block_size;

    float acc = 0.f;
    for (int tap = 0; tap < taps; ++tap) {
        if (t - tap < 0) {
            break;
        }
        if (tap > 0 && pos < tap) {
            continue;
        }
        const float hv    = to_f(hidden[(long long)(t - tap) * H + h]);
        const float coeff = to_f(base[tap * H + h]) + to_f(delta[((long long)t * taps + tap) * G + g]);
        acc += coeff * hv;
    }
    from_f(out[idx], acc);
}

template<typename T, int RANK>
__global__ void selector_scores_kernel(float* __restrict__ out,
                                       const T* __restrict__ hidden,   // [B, L, rank]
                                       const T* __restrict__ unary,    // [B, L, K]
                                       const T* __restrict__ pred_cb,  // [V, rank]
                                       const T* __restrict__ succ_cb,  // [V, rank]
                                       const int* __restrict__ cand_ids,  // [B, L, K]
                                       const int* __restrict__ anchor,    // [B]
                                       int B,
                                       int L,
                                       int K,
                                       int V,
                                       int rank)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)B * L * K * K;
    if (idx >= total) {
        return;
    }
    int c_   = idx % K;
    long long q = idx / K;
    int p_   = q % K;
    q        = q / K;
    int l    = q % L;
    int b    = q / L;

    const int cand = cand_ids[((long long)b * L + l) * K + c_];
    const int pred = (l == 0) ? anchor[b] : cand_ids[((long long)b * L + (l - 1)) * K + p_];

    const T*    hrow = hidden + ((long long)b * L + l) * rank;
    const T*    prow = pred_cb + (long long)pred * rank;
    const T*    srow = succ_cb + (long long)cand * rank;

    float acc = 0.f;
    for (int r = 0; r < rank; ++r) {
        acc += to_f(prow[r]) * to_f(hrow[r]) * to_f(srow[r]);
    }
    out[idx] = to_f(unary[((long long)b * L + l) * K + c_]) + acc;
}

template<typename T>
void launch_selector(float*       out,
                     const void*  hidden,
                     const void*  unary,
                     const void*  pred_cb,
                     const void*  succ_cb,
                     const int*   cand_ids,
                     const int*   anchor,
                     int          B,
                     int          L,
                     int          K,
                     int          V,
                     int          rank,
                     cudaStream_t stream)
{
    const long long total = (long long)B * L * K * K;
    const int       block = 256;
    const long long grid  = (total + block - 1) / block;
    selector_scores_kernel<T, 0><<<grid, block, 0, stream>>>(out,
                                                             static_cast<const T*>(hidden),
                                                             static_cast<const T*>(unary),
                                                             static_cast<const T*>(pred_cb),
                                                             static_cast<const T*>(succ_cb),
                                                             cand_ids,
                                                             anchor,
                                                             B,
                                                             L,
                                                             K,
                                                             V,
                                                             rank);
}

__device__ __forceinline__ float warp_reduce_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

template<int NT>
__device__ __forceinline__ float block_reduce_sum(float v, float* sh)
{
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    v              = warp_reduce_sum(v);
    if (lane == 0) {
        sh[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        v = (lane < NT / 32) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            sh[0] = v;
        }
    }
    __syncthreads();
    return sh[0];
}

/// One block per (query row, query head); thread d owns output channel d.
template<typename T, int HD>
__global__ void dflash2_attn_kernel(T* __restrict__ out,
                                    const T* __restrict__ q,
                                    const T* __restrict__ k_ctx,
                                    const T* __restrict__ v_ctx,
                                    const T* __restrict__ k_blk,
                                    const T* __restrict__ v_blk,
                                    int  tq,
                                    int  tc,
                                    int  nh,
                                    int  nkv,
                                    float scale,
                                    int  window)
{
    const int bh  = blockIdx.x;
    const int i   = bh / nh;
    const int h   = bh - i * nh;
    const int d   = threadIdx.x;
    const int kvh = h * nkv / nh;
    const int qpos = tc + i;
    const int first_ctx = max(0, qpos - window + 1);
    const int nkeys = tc + tq;

    const T* qrow = q + ((long long)i * nh + h) * HD;

    __shared__ float sh[32];

    auto key_dot = [&](int j, float& s_valid) {
        float part = 0.f;
        bool  valid;
        const T* krow;
        if (j < tc) {
            valid = j >= first_ctx;
            krow  = k_ctx + ((long long)j * nkv + kvh) * HD;
        }
        else {
            valid = true;
            krow  = k_blk + ((long long)(j - tc) * nkv + kvh) * HD;
        }
        if (valid) {
            part = to_f(qrow[d]) * to_f(krow[d]);
        }
        float s = block_reduce_sum<HD>(part, sh);
        s_valid = valid ? s * scale : -CUDART_INF_F;
        return s_valid;
    };

    float maxs = -CUDART_INF_F;
    for (int j = 0; j < nkeys; ++j) {
        float s;
        key_dot(j, s);
        maxs = fmaxf(maxs, s);
    }

    float acc = 0.f;
    for (int j = 0; j < nkeys; ++j) {
        float s;
        key_dot(j, s);
        const float e = (s == -CUDART_INF_F) ? 0.f : __expf(s - maxs);
        const T* vrow = (j < tc) ? (v_ctx + ((long long)j * nkv + kvh) * HD)
                                 : (v_blk + ((long long)(j - tc) * nkv + kvh) * HD);
        acc += e * to_f(vrow[d]);
    }
    float sum = 0.f;
    for (int j = 0; j < nkeys; ++j) {
        float s;
        key_dot(j, s);
        const float e = (s == -CUDART_INF_F) ? 0.f : __expf(s - maxs);
        sum += e;
    }
    // Every d computes the same `sum` (block reduction is identical across d).
    T r;
    from_f(r, acc / sum);
    out[((long long)i * nh + h) * HD + d] = r;
}

template<typename T, int HD>
__global__ void dflash2_rope_kernel(T* __restrict__ x,
                                    const int* __restrict__ positions,
                                    int   rows,
                                    int   heads,
                                    float base)
{
    const int        half  = HD / 2;
    const int        idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long long  total = (long long)rows * heads * half;
    if (idx >= total) {
        return;
    }
    const int j  = idx % half;
    const int hh = (idx / half) % heads;
    const int r  = idx / half / heads;
    const int p  = positions ? positions[r] : r;

    T* row = x + ((long long)r * heads + hh) * HD;
    const float theta = __powf(base, -2.0f * (float)j / (float)HD);
    const float ang   = (float)p * theta;
    const float c     = __cosf(ang);
    const float s     = __sinf(ang);
    const float lo    = to_f(row[j]);
    const float hi    = to_f(row[j + half]);
    T     lo_out, hi_out;
    from_f(lo_out, lo * c - hi * s);
    from_f(hi_out, hi * c + lo * s);
    row[j]        = lo_out;
    row[j + half] = hi_out;
}

template<typename T>
void launch_rope(void*        x,
                 const int*   positions,
                 int          rows,
                 int          heads,
                 int          head_dim,
                 float        base,
                 cudaStream_t stream)
{
    if (head_dim != 128) {
        return;
    }
    const int       half  = head_dim / 2;
    const long long total = (long long)rows * heads * half;
    const int       block = 256;
    const long long grid  = (total + block - 1) / block;
    dflash2_rope_kernel<T, 128><<<grid, block, 0, stream>>>(static_cast<T*>(x), positions, rows, heads, base);
}

template<typename T>
void launch_attn(void*        out,                 const void*  q,
                 const void*  k_ctx,
                 const void*  v_ctx,
                 const void*  k_blk,
                 const void*  v_blk,
                 int          tq,
                 int          tc,
                 int          nh,
                 int          nkv,
                 int          hd,
                 float        scale,
                 int          window,
                 cudaStream_t stream)
{
    TM_CHECK_EQ(hd, 128) << "DFlash2 attention expects head_dim 128";
    const int grid  = tq * nh;
    const int block = 128;
    dflash2_attn_kernel<T, 128><<<grid, block, 0, stream>>>(static_cast<T*>(out),
                                                            static_cast<const T*>(q),
                                                            static_cast<const T*>(k_ctx),
                                                            static_cast<const T*>(v_ctx),
                                                            static_cast<const T*>(k_blk),
                                                            static_cast<const T*>(v_blk),
                                                            tq,
                                                            tc,
                                                            nh,
                                                            nkv,
                                                            scale,
                                                            window);
}

template<typename T>
void launch_conv(void*        out,                 const void*  hidden,
                 const void*  delta,
                 const void*  base,
                 int          rows,
                 int          H,
                 int          G,
                 int          group_size,
                 int          taps,
                 int          block_size,
                 cudaStream_t stream)
{
    const long long total = (long long)rows * H;
    const int       block = 256;
    const long long grid  = (total + block - 1) / block;
    grouped_conv_kernel<T><<<grid, block, 0, stream>>>(static_cast<T*>(out),
                                                       static_cast<const T*>(hidden),
                                                       static_cast<const T*>(delta),
                                                       static_cast<const T*>(base),
                                                       rows,
                                                       H,
                                                       G,
                                                       group_size,
                                                       taps,
                                                       block_size);
}

}  // namespace

void invokeGroupedConv(void*        out,
                       const void*  hidden,
                       const void*  delta,
                       const void*  base,
                       int          rows,
                       int          hidden_size,
                       int          num_groups,
                       int          group_size,
                       int          taps,
                       int          block_size,
                       int          input_bits,
                       cudaStream_t stream)
{
    if (input_bits == 16) {
        launch_conv<__half>(out, hidden, delta, base, rows, hidden_size, num_groups, group_size, taps, block_size, stream);
    }
    else if (input_bits == 32) {
        launch_conv<float>(out, hidden, delta, base, rows, hidden_size, num_groups, group_size, taps, block_size, stream);
    }
    else {
        TM_CHECK(false) << "grouped conv: unsupported input bits " << input_bits;
    }
}

void invokeCandidateSelector(float*       out,
                             const void*  hidden,
                             const void*  unary_logits,
                             const void*  predecessor_codebook,
                             const void*  successor_codebook,
                             const int*   candidate_ids,
                             const int*   anchor_token_ids,
                             int          batch,
                             int          len,
                             int          top_k,
                             int          vocab,
                             int          rank,
                             int          input_bits,
                             cudaStream_t stream)
{
    if (input_bits == 16) {
        launch_selector<__half>(out,
                                hidden,
                                unary_logits,
                                predecessor_codebook,
                                successor_codebook,
                                candidate_ids,
                                anchor_token_ids,
                                batch,
                                len,
                                top_k,
                                vocab,
                                rank,
                                stream);
    }
    else if (input_bits == 32) {
        launch_selector<float>(out,
                               hidden,
                               unary_logits,
                               predecessor_codebook,
                               successor_codebook,
                               candidate_ids,
                               anchor_token_ids,
                               batch,
                               len,
                               top_k,
                               vocab,
                               rank,
                               stream);
    }
    else {
        TM_CHECK(false) << "candidate selector: unsupported input bits " << input_bits;
    }
}

void invokeDflash2Attention(void*        out,
                            const void*  q,
                            const void*  k_ctx,
                            const void*  v_ctx,
                            const void*  k_blk,
                            const void*  v_blk,
                            int          tq,
                            int          tc,
                            int          nh,
                            int          nkv,
                            int          hd,
                            float        scale,
                            int          window,
                            int          input_bits,
                            cudaStream_t stream)
{
    if (input_bits == 16) {
        launch_attn<__half>(out, q, k_ctx, v_ctx, k_blk, v_blk, tq, tc, nh, nkv, hd, scale, window, stream);
    }
    else if (input_bits == 32) {
        launch_attn<float>(out, q, k_ctx, v_ctx, k_blk, v_blk, tq, tc, nh, nkv, hd, scale, window, stream);
    }
    else {
        TM_CHECK(false) << "dflash2 attention: unsupported input bits " << input_bits;
    }
}

void invokeDflash2Rope(void*        x,
                       const int*   positions,
                       int          rows,
                       int          heads,
                       int          head_dim,
                       float        base,
                       int          input_bits,
                       cudaStream_t stream)
{
    if (input_bits == 16) {
        launch_rope<__half>(x, positions, rows, heads, head_dim, base, stream);
    }
    else if (input_bits == 32) {
        launch_rope<float>(x, positions, rows, heads, head_dim, base, stream);
    }
    else {
        TM_CHECK(false) << "dflash2 rope: unsupported input bits " << input_bits;
    }
}

}  // namespace turbomind
