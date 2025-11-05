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

template<typename T, int WarpCnt, int N, int M, int HeadDim>
__global__ void ReduceOutput(T*                  out,  //
                             float*              partial_O,
                             float*              cp_ML,  // q, h, k, 2
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
    const int split_cnt = (split_cnt_ != nullptr) ? max(split_cnt_[q], 1) : 1;
    if (offset_k >= split_cnt) {
        return;
    }

    auto get_scale = [&](int q, int h, int ki) {  // q, h, k, 2
        int index = ((q * num_heads + h) * max_split_cnt + ki) * 2;
        return cp_ML[index];
    };

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
        VecT  frag_O;
        float scale = get_scale(q, h, 0);

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
            float scale = (stride_k == 1) ? get_scale(q, h, ki) : 1.0f;

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
            ctx->cp_ML + params->cp_rank * params->token_num * params->num_heads * params->max_split_k * 2,
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

template<int WarpCnt>
__global__ void ReduceScale(float*              cp_ML,  // cp, q, h, k, 2
                            int                 num_tokens,
                            cutlass::FastDivmod num_heads,
                            int*                split_cnt_,
                            int                 max_split_cnt,
                            int                 cp_size,
                            int                 cp_rank,
                            float               exp_scale)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    int qh = blockIdx.x * WarpCnt + warp_id;
    int q  = num_heads.div(qh);

    if (q >= num_tokens) {
        return;
    }

    float frag_M0 = -std::numeric_limits<float>::infinity();
    float frag_L0 = 0.0f;

    const int split_per_rank = (split_cnt_ == nullptr) ? 1 : max(split_cnt_[q], 1);
    const int split_all_rank = split_per_rank * cp_size;

    int split_i, split_k;
    for (int i = lane_id; i < split_all_rank; i += WARP_SIZE) {
        split_i   = i / split_per_rank;
        split_k   = i % split_per_rank;
        int index = (split_i * num_tokens * num_heads + qh) * max_split_cnt + split_k;

        float frag_M1 = cp_ML[index * 2];
        float frag_L1 = cp_ML[index * 2 + 1];
        float frag_M  = fmaxf(frag_M0, frag_M1);

        frag_L1 = (frag_M1 == -std::numeric_limits<float>::infinity()) ?
                      0.0f :
                      exp2f((frag_M1 - frag_M) * exp_scale) * frag_L1;
        frag_L0 = (frag_M0 == -std::numeric_limits<float>::infinity()) ?
                      0.0f :
                      exp2f((frag_M0 - frag_M) * exp_scale) * frag_L0;

        frag_L0 = frag_L1 + frag_L0;
        frag_M0 = frag_M;
    }

    float block_M = frag_M0;
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
    }

    float block_L =
        (frag_M0 == -std::numeric_limits<float>::infinity()) ? 0.0f : exp2f((frag_M0 - block_M) * exp_scale) * frag_L0;

    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
    }

    for (int i = lane_id; i < split_per_rank; i += WARP_SIZE) {
        split_k   = i % split_per_rank;
        int index = (cp_rank * num_tokens * num_heads + qh) * max_split_cnt + split_k;

        float frag_M1    = cp_ML[index * 2];
        float scale      = (frag_M1 == -std::numeric_limits<float>::infinity()) ?
                               0.0f :
                               exp2f((frag_M1 - block_M) * exp_scale) / block_L;
        cp_ML[index * 2] = scale;  // save to M
    }
}

template<typename T>
void invokeReduceScale(CpPostContext* ctx, AttentionParams<T>* params, int split_cnt)
{
    constexpr int WarpCnt = 4;  // each warp process one token
    const dim3    block(WarpCnt * WARP_SIZE);
    const dim3    grid((params->token_num * params->num_heads + WarpCnt - 1) / WarpCnt);

    ReduceScale<WarpCnt><<<grid, block, 0, params->stream>>>(  //
        ctx->cp_ML,
        params->token_num,
        cutlass::FastDivmod(params->num_heads),
        split_cnt > 1 ? params->split_cnt : nullptr,
        params->max_split_k,
        params->cp_size,
        params->cp_rank,
        params->inv_sqrt_dh);

    sync_check_cuda_error();
}

template<typename T>
void CpReduce(CpPostContext* ctx, AttentionParams<T>* params, int split_cnt)
{
    NvtxScope scope("CpReduce");

    const int count = params->token_num * params->num_heads * params->max_split_k * 2;
    ctx->d_comm->AllGather(ctx->cp_ML + params->cp_rank * count,  //
                           ctx->cp_ML,
                           count,
                           DataType::kFloat,
                           ctx->attn_cp_group,
                           params->stream);
    sync_check_cuda_error();

    invokeReduceScale(ctx, params, split_cnt);

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
