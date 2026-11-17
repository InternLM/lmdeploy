// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/dflash2/dflash2_kernels.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

template<typename T>
void launch_conv(void*        out,
                 const void*  hidden,
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

}  // namespace turbomind
