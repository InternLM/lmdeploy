// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/models/llama/cp_utils.h"
#include "src/turbomind/models/llama/llama_utils.h"

namespace turbomind {

int next_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

template<int WarpCnt>
__global__ void ReduceK(float2* cp_ML,
                        float*  partial_M,  // q, h, k
                        float*  partial_L,  // q, h, k
                        int*    split_cnt_,
                        int     max_split_k,
                        int     num_tokens,
                        int     num_heads,
                        int     stride_k,
                        int     offset_k,
                        float   exp_scale)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    offset_k *= blockIdx.z;

    const int q         = blockIdx.x * WarpCnt + warp_id;
    const int h         = blockIdx.y;
    const int split_cnt = (q >= num_tokens) ? 0 : split_cnt_[q];
    if (offset_k >= split_cnt) {
        return;
    }

    float frag_M = -std::numeric_limits<float>::infinity();
    float frag_L = 0.0f;

    const int  ki    = lane_id * stride_k + offset_k;
    const bool mask  = ki < split_cnt && h < num_heads;
    const int  index = (q * num_heads + h) * max_split_k + ki;

    if (mask) {
        frag_M = partial_M[index];
        frag_L = partial_L[index];
    }

    float block_M = frag_M;
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
    }

    float expdiff_M = exp2f((frag_M - block_M) * exp_scale);
    float block_L   = expdiff_M * frag_L;

    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
    }

    if (mask) {
        partial_M[index] = block_M;
        partial_L[index] = block_L;

        if (ki == 0 && gridDim.z == 1) {
            cp_ML[q * num_heads + h] = {block_M, block_L};
        }
    }
}

template<typename T>
void invokeReduceK(CpPostContext* ctx, AttentionParams<T>* params, int split_cnt)
{
    constexpr int MaxN = 32;

    int split_k  = split_cnt;
    int stride_k = 1;
    int offset_k = 1;

    auto invoke = [&](auto n) {
        constexpr int WarpCnt = 4;
        const dim3    block(WarpCnt * WARP_SIZE);
        const dim3    grid((params->token_num + WarpCnt - 1) / WarpCnt, params->num_heads, (split_k + n - 1) / n);
        ReduceK<WarpCnt><<<grid, block, 0, params->stream>>>(  //
            (float2*)ctx->cp_ML + params->cp_rank * params->token_num * params->num_heads,
            params->partial_M,
            params->partial_L,
            params->split_cnt,
            params->max_split_k,
            params->token_num,
            params->num_heads,
            stride_k,
            offset_k * n,
            params->inv_sqrt_dh);
        sync_check_cuda_error();

        stride_k *= n;
        offset_k *= n;
        split_k = (split_k + n - 1) / n;
    };

    auto dispatch_n = [&](int n) {
        n = min(next_power_of_two(n), MaxN);
        switch (n) {
            case 2:
                return invoke(std::integral_constant<int, 2>{});
            case 4:
                return invoke(std::integral_constant<int, 4>{});
            case 8:
                return invoke(std::integral_constant<int, 8>{});
            case 16:
                return invoke(std::integral_constant<int, 16>{});
            case 32:
                return invoke(std::integral_constant<int, 32>{});
            default:
                TM_CHECK(0);
        }
    };

    while (split_k > 1) {
        dispatch_n(split_k);
    }
}

template<int WarpCnt>
__global__ void ReduceCP(float2* cp_ML,  // cp, q, h, 2
                         int     cp_size,
                         int     num_heads,
                         int     total,
                         int     stride,
                         int     offset,
                         float   exp_scale)
{
    __shared__ float2 s_ML[WarpCnt][WARP_SIZE + 1];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    offset *= blockIdx.y;
    const int qh_offset = blockIdx.x * WARP_SIZE;
    if (qh_offset >= total || offset >= cp_size) {
        return;
    }

    float2 ml = {-std::numeric_limits<float>::infinity(), 0.f};

    int qh = qh_offset + lane_id;
    int ki = warp_id * stride + offset;
    if (ki < cp_size && qh < total) {
        ml = cp_ML[ki * total + qh];
    }
    s_ML[warp_id][lane_id] = ml;

    __syncthreads();

    // Reduce
    const int qh_i = lane_id / (WarpCnt * 2) * (WarpCnt * 2) + lane_id % (WarpCnt * 2) / WarpCnt + warp_id * 2;
    const int wi   = lane_id % WarpCnt;

    ml           = s_ML[wi][qh_i];
    float frag_M = ml.x;
    float frag_L = ml.y;

    float block_M = frag_M;
    PRAGMA_UNROLL
    for (int mask = WarpCnt / 2; mask >= 1; mask /= 2) {
        block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
    }

    float expdiff_M = exp2f((frag_M - block_M) * exp_scale);

    float block_L = frag_L * expdiff_M;
    PRAGMA_UNROLL
    for (int mask = WarpCnt / 2; mask >= 1; mask /= 2) {
        block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
    }

    if (wi == 0 && (qh_offset + qh_i < total)) {
        cp_ML[qh_offset + qh_i] = {block_M, block_L};
    }
}

template<typename T>
void invokeReduceCP(CpPostContext* ctx, AttentionParams<T>* params)
{
    constexpr int MaxN  = 8;
    const int     total = params->token_num * params->num_heads;

    int split_k  = params->cp_size;
    int stride_k = 1;
    int offset_k = 1;

    auto invoke = [&](auto n) {
        const dim3 block(n * WARP_SIZE);
        const dim3 grid((total + WARP_SIZE - 1) / WARP_SIZE, (split_k + n - 1) / n);
        const int  shm_size = sizeof(float2) * n * (WARP_SIZE + 1);
        ReduceCP<n><<<grid, block, shm_size, params->stream>>>(  //
            (float2*)ctx->cp_ML,
            params->cp_size,
            params->num_heads,
            total,
            stride_k,
            offset_k * n,
            params->inv_sqrt_dh);
        sync_check_cuda_error();

        stride_k *= n;
        offset_k *= n;
        split_k = (split_k + n - 1) / n;
    };

    auto dispatch_n = [&](int n) {
        n = min(next_power_of_two(n), MaxN);
        switch (n) {
            case 2:
                return invoke(std::integral_constant<int, 2>{});
            case 4:
                return invoke(std::integral_constant<int, 4>{});
            case 8:
                return invoke(std::integral_constant<int, 8>{});
            default:
                TM_CHECK(0);
        }
    };

    while (split_k > 1) {
        dispatch_n(split_k);
    }
}

template<typename T, int WarpCnt, int N, int M, int HeadDim>
__global__ void ReduceOutput(T*                  out,  //
                             float*              partial_O,
                             float*              cp_k_ML,  // q, h, k, 2
                             float2*             cp_ML,    // q, h, 2
                             cutlass::FastDivmod h_divmod,
                             int*                split_cnt_,
                             int                 max_split_cnt,
                             int                 total,
                             int                 num_heads,
                             int                 stride_k,
                             int                 offset_k,
                             float               exp_scale)
{
    __shared__ float s_out[WarpCnt][HeadDim];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // warp_id, q, h
    const int qh = blockIdx.x * M + warp_id % M;
    int       q, h;
    h_divmod(q, h, qh);

    if (q * num_heads + h >= total) {
        return;
    }

    offset_k *= blockIdx.y;
    const int split_cnt = (split_cnt_ != nullptr) ? split_cnt_[q] : 1;
    if (offset_k >= split_cnt) {
        return;
    }

    float  scale = 1.0f;
    float2 global_ML;

    auto get_scale = [&](float2 ml, int ki) {
        int index = (q * num_heads + h) * max_split_cnt * 2 + ki * 2;
        return exp2f((cp_k_ML[index] - ml.x) * exp_scale) / ml.y;
    };

    if (stride_k == 1) {
        global_ML = cp_ML[q * num_heads + h];
    }

    // HeadDim / WARP_SIZE
    // 128     -> 4
    // 64, 192 -> 2
    constexpr int kVecSize = HeadDim % 128 == 0 ? 4 : 2;
    constexpr int iterC    = HeadDim / (WARP_SIZE * kVecSize);

    using namespace ops;
    using VecF = Array<float, kVecSize>;
    using VecT = Array<T, kVecSize>;

    // in most cases,no split_k
    if constexpr (N == 1) {
        VecT frag_O;
        scale = get_scale(global_ML, 0);

        PRAGMA_UNROLL
        for (int c = 0; c < iterC; ++c) {
            Load(frag_O, &out[(q * num_heads + h) * HeadDim + lane_id * kVecSize + c * WARP_SIZE * kVecSize]);
            frag_O = cast<T>(cast<float>(frag_O) * scale);
            Store(&out[(q * num_heads + h) * HeadDim + lane_id * kVecSize + c * WARP_SIZE * kVecSize], frag_O);
        }

        return;
    }

    VecF accu_O[iterC]{};
    VecF frag_O[iterC];

    PRAGMA_UNROLL
    for (int k = 0; k < N; k += WarpCnt / M) {
        const int ki   = (warp_id / M + k) * stride_k + offset_k;
        const int base = (((q * num_heads + h) * max_split_cnt + ki) * HeadDim);  // q, h, k, d

        if (ki < split_cnt) {
            if (stride_k == 1) {
                scale = get_scale(global_ML, ki);
            }

            PRAGMA_UNROLL
            for (int c = 0; c < iterC; ++c) {
                const int index = base + lane_id * kVecSize + c * WARP_SIZE * kVecSize;
                Load(frag_O[c], &partial_O[index]);
                accu_O[c] = accu_O[c] + frag_O[c] * scale;
            }
        }
    }

    PRAGMA_UNROLL
    for (int c = 0; c < iterC; ++c) {
        Store(&s_out[warp_id][c * WARP_SIZE * kVecSize + lane_id * kVecSize], accu_O[c]);
    }

    // PRAGMA_UNROLL
    // for (int w = WarpCnt / 2 / M; w > 0; w /= 2) {
    //     const int ki = warp_id / M;
    //     __syncthreads();
    //     if (ki < w) {
    //         PRAGMA_UNROLL
    //         for (int c = 0; c < iterC; ++c) {
    //             const int index              = c * WARP_SIZE * kVecSize + lane_id * kVecSize;
    //             (VecF&)s_out[warp_id][index] = (VecF&)s_out[warp_id][index] + (VecF&)s_out[warp_id + w * M][index];
    //         }
    //     }
    // }

    __syncthreads();
    if (warp_id / M == 0) {
        PRAGMA_UNROLL
        for (int k = 1; k < WarpCnt / M; ++k) {
            for (int c = 0; c < iterC; ++c) {
                const int index              = c * WARP_SIZE * kVecSize + lane_id * kVecSize;
                (VecF&)s_out[warp_id][index] = (VecF&)s_out[warp_id][index] + (VecF&)s_out[warp_id + k * M][index];
            }
        }
    }

    if (warp_id / M == 0) {
        const int base = gridDim.y == 1 ? (q * num_heads + h) * HeadDim :
                                          (((q * num_heads + h) * max_split_cnt + offset_k) * HeadDim);
        PRAGMA_UNROLL
        for (int c = 0; c < iterC; ++c) {
            const int off = c * WARP_SIZE * kVecSize + lane_id * kVecSize;
            if (gridDim.y == 1) {
                Store(&out[base + off], cast<T>((VecF&)s_out[warp_id][off]));
            }
            else {
                Store(&partial_O[base + off], (VecF&)s_out[warp_id][off]);
            }
        }
    }
}

template<typename T>
void invokeReduceOutput(CpPostContext* ctx, AttentionParams<T>* params, int split_cnt)
{
    constexpr int MaxN = 32;

    int split_k  = split_cnt;
    int stride_k = 1;
    int offset_k = 1;

    cutlass::FastDivmod h_divmod = cutlass::FastDivmod(params->num_heads);

    auto invoke = [&](auto n, auto head_dim) {
        constexpr int WarpCnt = 4;
        constexpr int M       = (WarpCnt + n - 1) / n;  // item per block, 1, 2, 4
        const int     total   = params->token_num * params->num_heads;

        const dim3 block(WarpCnt * WARP_SIZE);
        const dim3 grid((total + M - 1) / M, (split_k + n - 1) / n);
        const int  shm_size = WarpCnt * sizeof(float) * head_dim;
        ReduceOutput<T, WarpCnt, n, M, head_dim><<<grid, block, shm_size, params->stream>>>(  //
            params->out + params->cp_q_offset * params->num_heads * params->size_per_head,
            params->partial_O,
            params->cp_k_ML,
            (float2*)ctx->cp_ML,
            h_divmod,
            split_cnt > 1 ? params->split_cnt : nullptr,
            params->max_split_k,
            total,
            params->num_heads,
            stride_k,
            offset_k * n,
            params->inv_sqrt_dh);

        sync_check_cuda_error();

        stride_k *= n;
        offset_k *= n;
        split_k = (split_k + n - 1) / n;
    };

    auto dispatch_n = [&](int split_k, auto head_dim) {
        int n = min(next_power_of_two(split_k), MaxN);

        switch (n) {
            case 1:
                return invoke(std::integral_constant<int, 1>{}, head_dim);
            case 2:
                return invoke(std::integral_constant<int, 2>{}, head_dim);
            case 4:
                return invoke(std::integral_constant<int, 4>{}, head_dim);
            case 8:
                return invoke(std::integral_constant<int, 8>{}, head_dim);
            case 16:
                return invoke(std::integral_constant<int, 16>{}, head_dim);
            case 32:
                return invoke(std::integral_constant<int, 32>{}, head_dim);
            default:
                TM_CHECK(0);
        }
    };

    auto dispatch_head_dim = [&](int split_k) {
        switch (params->size_per_head) {
            case 64:
                return dispatch_n(split_k, std::integral_constant<int, 64>{});
            case 128:
                return dispatch_n(split_k, std::integral_constant<int, 128>{});
            case 192:
                return dispatch_n(split_k, std::integral_constant<int, 192>{});
            default:
                TM_CHECK(0);
        }
    };

    dispatch_head_dim(split_k);
    while (split_k > 1) {
        dispatch_head_dim(split_k);
    }
}

template<typename T>
void CpReduce(CpPostContext* ctx, AttentionParams<T>* params, int split_cnt)
{
    NvtxScope scope("CpReduce");

    if (split_cnt > 1) {
        invokeReduceK(ctx, params, split_cnt);
    }

    const int count = params->token_num * params->num_heads * 2;
    ctx->d_comm->AllGather(ctx->cp_ML + params->cp_rank * count,  //
                           ctx->cp_ML,
                           count,
                           DataType::kFloat,
                           ctx->attn_cp_group,
                           params->stream);
    sync_check_cuda_error();

    invokeReduceCP(ctx, params);
    invokeReduceOutput(ctx, params, split_cnt);
}

void CpPost(void* context, int split_cnt)
{
    auto ctx = reinterpret_cast<CpPostContext*>(context);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        CpReduce<T>(ctx, static_cast<AttentionParams<T>*>(ctx->attn_param), split_cnt);
    };

    TM_DISPATCH_PRIMARY_DTYPES(ctx->attn_type, invoke);
}

}  // namespace turbomind
