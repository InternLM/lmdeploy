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
                                       const float* __restrict__ unary,// [B, L, K] (fp32)
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
    // unary is index-aligned with the successor (column) candidate: the
    // reference does `unary_logits[:, :, None]` on a [B,L,K] tensor, which
    // inserts the broadcast axis *before* K (i.e. [B,L,1,K] -> by column).
    out[idx] = unary[((long long)b * L + l) * K + c_] + acc;
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
                                                             static_cast<const float*>(unary),
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

template<typename T>
__global__ void residual_norm_kernel(T* __restrict__ out, float* __restrict__ residual, const T* __restrict__ x,
                                     const T* __restrict__ w, int rows, int cols, float eps, int has_x,
                                     float x_scale)
{
    const int r     = blockIdx.x;
    float*    res   = residual + (long long)r * cols;
    T*        orow  = out + (long long)r * cols;
    const T*  xrow  = x ? x + (long long)r * cols : nullptr;

    if (has_x && xrow) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            res[c] += to_f(xrow[c]) * x_scale;
        }
    }
    __syncthreads();

    float ss = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = res[c];
        ss += v * v;
    }
    __shared__ float sh[32];
    ss = block_reduce_sum<256>(ss, sh);
    const float inv = rsqrtf(ss / (float)cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v;
        from_f(v, res[c] * inv * to_f(w[c]));
        orow[c] = v;
    }
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

__global__ void commit_len_kernel(int* out, const int* entry, const int* accept, int has_accept)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = entry[0] + (has_accept ? accept[0] : 0);
    }
}

void invokeDflash2CommitLen(int* out, const int* entry, const int* accept, int has_accept, cudaStream_t stream)
{
    commit_len_kernel<<<1, 1, 0, stream>>>(out, entry, accept, has_accept);
}

__global__ void count_up_kernel(int* out, const int* base, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = base[0] + i;
    }
}

void invokeDflash2CountUp(int* out, const int* base, int n, cudaStream_t stream)
{
    const int block = 128;
    const int grid  = (n + block - 1) / block;
    count_up_kernel<<<grid, block, 0, stream>>>(out, base, n);
}

template<typename T>
__global__ void kv_scatter_kernel(T* __restrict__ dst, const T* __restrict__ src, const int* __restrict__ base,
                                  const int* __restrict__ committed, int rows, int per_row)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * per_row;
    if (idx >= total) {
        return;
    }
    const int i = (int)(idx / per_row);
    const int r = (int)(idx - (long long)i * per_row);
    const int b = base[0];
    int       delta = committed[0] - b;
    if (delta < 0) delta = 0;
    if (delta > rows) delta = rows;
    if (b < 0 || i >= delta) {
        return;
    }
    dst[((long long)(b + i)) * per_row + r] = src[idx];
}

void invokeDflash2KvScatter(void* dst, const void* src, const int* base, const int* committed, int rows, int per_row,
                            int input_bits, cudaStream_t stream)
{
    const long long total = (long long)rows * per_row;
    const int       block = 256;
    const long long grid  = (total + block - 1) / block;
    if (input_bits == 16) {
        kv_scatter_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<__half*>(dst), static_cast<const __half*>(src), base, committed, rows, per_row);
    }
    else if (input_bits == 32) {
        kv_scatter_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(dst), static_cast<const float*>(src), base, committed, rows, per_row);
    }
    else {
        TM_CHECK(false) << "dflash2 kv_scatter: unsupported input bits " << input_bits;
    }
}


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

template<typename T>
__global__ void add_kernel(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    T r;
    from_f(r, to_f(a[i]) + to_f(b[i]));
    out[i] = r;
}

void invokeDflash2Add(void*        out,
                      const void*  a,
                      const void*  b,
                      long long    n,
                      int          input_bits,
                      cudaStream_t stream)
{
    const int       block = 256;
    const long long grid  = (n + block - 1) / block;
    if (input_bits == 16) {
        add_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<__half*>(out), static_cast<const __half*>(a), static_cast<const __half*>(b), n);
    }
    else if (input_bits == 32) {
        add_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(out), static_cast<const float*>(a), static_cast<const float*>(b), n);
    }
    else {
        TM_CHECK(false) << "dflash2 add: unsupported input bits " << input_bits;
    }
}

template<typename T, int K>
__global__ void topk_kernel(int* __restrict__ ids, float* __restrict__ vals, const T* __restrict__ logits,
                            int rows, int vocab)
{
    const int r = blockIdx.x;
    if (r >= rows) {
        return;
    }
    const T* row = logits + (long long)r * vocab;
    // insert-sort style top-K maintained in registers
    float  best[K];
    int    bidx[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        best[i] = -CUDART_INF_F;
        bidx[i] = 0;
    }
    for (int v = 0; v < vocab; ++v) {
        const float x = to_f(row[v]);
        if (x > best[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && best[pos - 1] < x) {
                best[pos] = best[pos - 1];
                bidx[pos] = bidx[pos - 1];
                --pos;
            }
            best[pos] = x;
            bidx[pos] = v;
        }
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
        ids[(long long)r * K + i]  = bidx[i];
        vals[(long long)r * K + i] = best[i];
    }
}

void invokeDflash2TopK(int*         candidate_ids,
                       float*       values,
                       const void*  logits,
                       int          rows,
                       int          vocab,
                       int          k,
                       int          input_bits,
                       cudaStream_t stream)
{
    TM_CHECK_EQ(k, 16) << "DFlash2 top-k expects k = 16";
    if (input_bits == 16) {
        topk_kernel<__half, 16><<<rows, 1, 0, stream>>>(
            candidate_ids, values, static_cast<const __half*>(logits), rows, vocab);
    }
    else if (input_bits == 32) {
        topk_kernel<float, 16><<<rows, 1, 0, stream>>>(
            candidate_ids, values, static_cast<const float*>(logits), rows, vocab);
    }
    else {
        TM_CHECK(false) << "dflash2 topk: unsupported input bits " << input_bits;
    }
}

__global__ void build_block_ids_kernel(int* ids, const int* const* token_ids_ptrs, const int* seq_len, int block,
                                       int mask)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    const int  seq = seq_len ? seq_len[0] : 0;
    const int* tok = token_ids_ptrs ? token_ids_ptrs[0] : nullptr;
    int        next = mask;
    if (seq > 0 && tok) {
        next = tok[seq - 1];
    }
    ids[0] = next;
    for (int i = 1; i < block; ++i) {
        ids[i] = mask;
    }
}

void invokeDflash2BuildBlockIds(int* ids, const int* const* token_ids_ptrs, const int* seq_len, int block,
                                int mask_token_id, cudaStream_t stream)
{
    build_block_ids_kernel<<<1, 1, 0, stream>>>(ids, token_ids_ptrs, seq_len, block, mask_token_id);
}

__global__ void greedy_chain_kernel(int* props, const float* edge, const int* cand, int len, int top_k)
{
    // single thread walks the chain
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int prev = 0;
    for (int l = 0; l < len; ++l) {
        const float* erow = edge + ((long long)l * top_k + prev) * top_k;
        int          c    = 0;
        for (int j = 1; j < top_k; ++j) {
            if (erow[j] > erow[c]) {
                c = j;
            }
        }
        props[l] = cand[l * top_k + c];
        prev      = c;
    }
}

void invokeDflash2GreedyChain(int* props, const float* edge, const int* cand, int len, int top_k,
                              cudaStream_t stream)
{
    greedy_chain_kernel<<<1, 1, 0, stream>>>(props, edge, cand, len, top_k);
}

__global__ void store_tokens_kernel(int* const* token_ids_ptrs, const int* props, const int* base, int n)
{
    int* tok = token_ids_ptrs[0];
    const int b = base[0];
    for (int i = 0; i < n; ++i) {
        tok[b + i] = props[i];
    }
}

void invokeDflash2StoreTokens(int* const* token_ids_ptrs, const int* props, const int* base, int n,
                              cudaStream_t stream)
{
    store_tokens_kernel<<<1, 1, 0, stream>>>(token_ids_ptrs, props, base, n);
}

void invokeDflash2ResidualRMSNorm(void*        out,
                                  float*       residual,
                                  const void*  x,
                                  const void*  w,
                                  int          rows,
                                  int          cols,
                                  float        eps,
                                  int          has_x,
                                  float        x_scale,
                                  int          input_bits,
                                  cudaStream_t stream)
{
    if (input_bits == 16) {
        residual_norm_kernel<__half><<<rows, 256, 0, stream>>>(static_cast<__half*>(out),
                                                               residual,
                                                               static_cast<const __half*>(x),
                                                               static_cast<const __half*>(w),
                                                               rows,
                                                               cols,
                                                               eps,
                                                               has_x,
                                                               x_scale);
    }
    else {
        residual_norm_kernel<float><<<rows, 256, 0, stream>>>(static_cast<float*>(out),
                                                              residual,
                                                              static_cast<const float*>(x),
                                                              static_cast<const float*>(w),
                                                              rows,
                                                              cols,
                                                              eps,
                                                              has_x,
                                                              x_scale);
    }
}

template<typename T>
__global__ void scale_kernel(T* __restrict__ t, long long n, float factor)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    T v;
    from_f(v, to_f(t[i]) * factor);
    t[i] = v;
}

void invokeDflash2Scale(void* t, long long n, float factor, int input_bits, cudaStream_t stream)
{
    const int       block = 256;
    const long long grid  = (n + block - 1) / block;
    if (input_bits == 16) {
        scale_kernel<__half><<<grid, block, 0, stream>>>(static_cast<__half*>(t), n, factor);
    }
    else if (input_bits == 32) {
        scale_kernel<float><<<grid, block, 0, stream>>>(static_cast<float*>(t), n, factor);
    }
}


template<typename T>
__global__ void kv_scatter_int4_kernel(unsigned char* __restrict__ dst_data,
                                       T* __restrict__ dst_param,
                                       const T* __restrict__ src,
                                       const int* __restrict__ base,
                                       const int* __restrict__ committed,
                                       int rows,
                                       int nkv,
                                       int hd)
{
    constexpr int NW = 8;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const long long wh = (long long)blockIdx.x * NW + wid;
    const long long total = (long long)rows * nkv;
    if (wh >= total) {
        return;
    }
    const int i = (int)(wh / nkv);
    const int h = (int)(wh - (long long)i * nkv);

    const int b     = base[0];
    int       delta = committed[0] - b;
    if (delta < 0) delta = 0;
    if (delta > rows) delta = rows;
    if (b < 0 || i >= delta) {
        return;
    }

    const T* srow = src + ((long long)i * nkv + h) * hd;
    const int d0 = lane;
    const int d1 = lane + 32;
    const int d2 = lane + 64;
    const int d3 = lane + 96;
    const float x0 = to_f(srow[d0]);
    const float x1 = to_f(srow[d1]);
    const float x2 = to_f(srow[d2]);
    const float x3 = to_f(srow[d3]);

    float mn = fminf(fminf(x0, x1), fminf(x2, x3));
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    for (int o = 16; o > 0; o >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }

    float scale = (mx - mn) / 15.f;
    if (!(scale > 0.f)) scale = 1.f;
    const float zero = mn;

    auto quant4 = [&](float x) {
        int v = __float2int_rn((x - zero) / scale);
        v = max(0, min(15, v));
        return (unsigned)v;
    };

    const unsigned char b0 = (unsigned char)(quant4(x0) | (quant4(x1) << 4));
    const unsigned char b1 = (unsigned char)(quant4(x2) | (quant4(x3) << 4));

    const long long row = (long long)(b + i) * nkv + h;
    unsigned char*  d   = dst_data + row * (hd / 2) + lane * 2;
    d[0] = b0;
    d[1] = b1;
    if (lane == 0) {
        T* param = dst_param + row * 2;
        from_f(param[0], scale);
        from_f(param[1], zero);
    }
}

template<typename T>
__global__ void dflash2_attn_int4_kernel(T* __restrict__ out,
                                         const T* __restrict__ q,
                                         const unsigned char* __restrict__ k_ctx_data,
                                         const T* __restrict__ k_ctx_param,
                                         const unsigned char* __restrict__ v_ctx_data,
                                         const T* __restrict__ v_ctx_param,
                                         const T* __restrict__ k_blk,
                                         const T* __restrict__ v_blk,
                                         int  tq,
                                         int  tc,
                                         int  nh,
                                         int  nkv,
                                         float scale,
                                         int  window)
{
    constexpr int HD = 128;
    constexpr int ND = HD / 32;
    constexpr int NW = 8;

    const int bh  = blockIdx.x;
    const int i   = bh / nh;
    const int h   = bh - i * nh;
    const int kvh = h * nkv / nh;
    const int qpos = tc + i;
    const int first_ctx = max(0, qpos - window + 1);
    const int nkeys = tc + tq;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    const T* qrow = q + ((long long)i * nh + h) * HD;
    float    qd[ND];
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        qd[k] = to_f(qrow[lane + k * 32]);
    }

    float m = -CUDART_INF_F, l = 0.f, acc[ND];
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        acc[k] = 0.f;
    }

    for (int j = first_ctx + wid; j < nkeys; j += NW) {
        float part = 0.f;
        if (j < tc) {
            const long long row = (long long)j * nkv + kvh;
            const unsigned short pk =
                *reinterpret_cast<const unsigned short*>(k_ctx_data + row * (HD / 2) + lane * 2);
            const unsigned short pv =
                *reinterpret_cast<const unsigned short*>(v_ctx_data + row * (HD / 2) + lane * 2);
            const float ks = to_f(k_ctx_param[row * 2 + 0]);
            const float kz = to_f(k_ctx_param[row * 2 + 1]);
            const float vs = to_f(v_ctx_param[row * 2 + 0]);
            const float vz = to_f(v_ctx_param[row * 2 + 1]);
            const float k0 = (float)(pk & 0x0f) * ks + kz;
            const float k1 = (float)((pk >> 4) & 0x0f) * ks + kz;
            const float k2 = (float)((pk >> 8) & 0x0f) * ks + kz;
            const float k3 = (float)((pk >> 12) & 0x0f) * ks + kz;
            part = qd[0] * k0 + qd[1] * k1 + qd[2] * k2 + qd[3] * k3;

            const float s    = warp_reduce_sum(part) * scale;
            const float mn   = fmaxf(m, s);
            const float corr = (m == -CUDART_INF_F) ? 0.f : __expf(m - mn);
            const float e    = __expf(s - mn);
            l                = l * corr + e;
            const float v0 = (float)(pv & 0x0f) * vs + vz;
            const float v1 = (float)((pv >> 4) & 0x0f) * vs + vz;
            const float v2 = (float)((pv >> 8) & 0x0f) * vs + vz;
            const float v3 = (float)((pv >> 12) & 0x0f) * vs + vz;
            acc[0] = acc[0] * corr + e * v0;
            acc[1] = acc[1] * corr + e * v1;
            acc[2] = acc[2] * corr + e * v2;
            acc[3] = acc[3] * corr + e * v3;
            m = mn;
        }
        else {
            const T* krow = k_blk + ((long long)(j - tc) * nkv + kvh) * HD;
            const T* vrow = v_blk + ((long long)(j - tc) * nkv + kvh) * HD;
#pragma unroll
            for (int k = 0; k < ND; ++k) {
                part += qd[k] * to_f(krow[lane + k * 32]);
            }
            const float s    = warp_reduce_sum(part) * scale;
            const float mn   = fmaxf(m, s);
            const float corr = (m == -CUDART_INF_F) ? 0.f : __expf(m - mn);
            const float e    = __expf(s - mn);
            l                = l * corr + e;
#pragma unroll
            for (int k = 0; k < ND; ++k) {
                acc[k] = acc[k] * corr + e * to_f(vrow[lane + k * 32]);
            }
            m = mn;
        }
    }

    __shared__ float sm[NW];
    __shared__ float sl[NW];
    __shared__ float sa[NW][32][ND];
    if (lane == 0) {
        sm[wid] = m;
        sl[wid] = l;
    }
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        sa[wid][lane][k] = acc[k];
    }
    __syncthreads();

    float M = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        M = fmaxf(M, sm[w]);
    }
    float L = 0.f, A[ND];
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        A[k] = 0.f;
    }
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        const float f = __expf(sm[w] - M);
        L += sl[w] * f;
#pragma unroll
        for (int k = 0; k < ND; ++k) {
            A[k] += sa[w][lane][k] * f;
        }
    }
    T* orow = out + ((long long)i * nh + h) * HD;
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        T r;
        from_f(r, A[k] / L);
        orow[lane + k * 32] = r;
    }
}

template<typename T>
void launch_attn_int4(void*        out,
                      const void*  q,
                      const void*  k_ctx_data,
                      const void*  k_ctx_param,
                      const void*  v_ctx_data,
                      const void*  v_ctx_param,
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
    TM_CHECK_EQ(hd, 128) << "DFlash2 int4 attention expects head_dim 128";
    const int grid  = tq * nh;
    const int block = 256;
    dflash2_attn_int4_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<T*>(out),
        static_cast<const T*>(q),
        static_cast<const unsigned char*>(k_ctx_data),
        static_cast<const T*>(k_ctx_param),
        static_cast<const unsigned char*>(v_ctx_data),
        static_cast<const T*>(v_ctx_param),
        static_cast<const T*>(k_blk),
        static_cast<const T*>(v_blk),
        tq,
        tc,
        nh,
        nkv,
        scale,
        window);
}

void invokeDflash2AttentionInt4(void*        out,
                                const void*  q,
                                const void*  k_ctx_data,
                                const void*  k_ctx_param,
                                const void*  v_ctx_data,
                                const void*  v_ctx_param,
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
        launch_attn_int4<__half>(out, q, k_ctx_data, k_ctx_param, v_ctx_data, v_ctx_param, k_blk, v_blk, tq, tc, nh,
                                 nkv, hd, scale, window, stream);
    }
    else {
        TM_CHECK(false) << "DFlash2 int4 attention: unsupported input bits " << input_bits;
    }
}

void invokeDflash2KvScatterInt4(void*        dst_data,
                                void*        dst_param,
                                const void*  src,
                                const int*   base,
                                const int*   committed,
                                int          rows,
                                int          nkv,
                                int          hd,
                                int          input_bits,
                                cudaStream_t stream)
{
    constexpr int NW = 8;
    const long long total = (long long)rows * nkv;
    const int grid = (int)((total + NW - 1) / NW);
    const int block = NW * 32;
    if (input_bits == 16) {
        kv_scatter_int4_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<unsigned char*>(dst_data),
            static_cast<__half*>(dst_param),
            static_cast<const __half*>(src),
            base,
            committed,
            rows,
            nkv,
            hd);
    }
    else {
        TM_CHECK(false) << "DFlash2 kv_scatter int4: unsupported input bits " << input_bits;
    }
}

}  // namespace turbomind
