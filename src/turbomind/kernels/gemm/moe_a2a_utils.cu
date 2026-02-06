// Copyright (c) OpenMMLab. All rights reserved.

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/moe_a2a_utils.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<int max_expert_num, int max_top_k, int items_per_thread, int block_dim, int access_size>
__global__ void MoeGateKernel(float*       topk_scales,        // [n, ep, topk]
                              int*         topk_experts,       // [n, ep, topk]
                              int*         token_idx_in_rank,  // [ep, n + 2]
                              const float* logits,             // [n,E]
                              int          token_num,
                              int          expert_num,
                              int          local_expert_num,
                              int          ep_size,
                              int          top_k,
                              bool         softmax,
                              bool         norm_topk,
                              float        routed_scale)
{
    constexpr int threads_per_token = max_expert_num / items_per_thread;  // 8

    // We use bits in a uint32_t to represent selected experts
    static_assert(items_per_thread <= 32);
    // We use warp-level primitives for reduction
    static_assert(threads_per_token <= 32);

    static_assert((threads_per_token & (threads_per_token - 1)) == 0);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int ti = thread_idx / threads_per_token;
    const int ei = thread_idx % threads_per_token;

    const int warp_ti = threadIdx.x % WARP_SIZE / threads_per_token;

    float data[items_per_thread];
    int   idxs[items_per_thread];

    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = -std::numeric_limits<float>::infinity();
        idxs[i] = ei * items_per_thread + i;
    }
    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
            }
        }
    }

    unsigned mask = (unsigned)-1;
    float    max_logit;

    const int warp_ti_offset = warp_ti * threads_per_token;

    auto run = [&](int k) {
        unsigned bit     = 1;
        unsigned max_bit = 0;
        float    max_val = -std::numeric_limits<float>::infinity();
        // local maximum
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if ((mask & bit) && data[i] > max_val) {
                max_bit = bit;
                max_val = data[i];
            }
            // weird thing that nvcc tends to use funnel shift for `bit <<= 1`
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }

        int   g_max_ei  = ei;
        float g_max_val = max_val;
        if constexpr (threads_per_token > 1) {
            // global maximum
            PRAGMA_UNROLL
            for (int m = threads_per_token / 2; m >= 1; m /= 2) {
                g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
            }
            // tie breaking
            const auto active = __ballot_sync((uint32_t)-1, max_val == g_max_val);
            g_max_ei          = __ffs(active >> (unsigned)warp_ti_offset) - 1;
        }
        if (k == 0) {
            max_logit = g_max_val;
        }
        if (ei == g_max_ei) {
            mask -= max_bit;
        }
    };

    run(0);

    for (int k = 1; k < top_k; ++k) {
        run(k);
    }

    mask = ~mask;

    int used[items_per_thread];
    {
        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            used[i] = (mask & bit) > 0;
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }
    }

    float sum_prob{};

    if (softmax) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (!norm_topk || used[i]) {
                data[i] = expf(data[i] - max_logit);
                sum_prob += data[i];
            }
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
        }
        sum_prob = fdividef(1.f, sum_prob);
    }
    else {
        sum_prob = 1.f;
    }

    for (int i = 0; i < items_per_thread; ++i) {
        if (used[i] && ti < token_num) {
            const int   expert_id = idxs[i];
            const float scale     = data[i] * sum_prob;
            const int   token     = ti;
            const int   ep_rank   = expert_id / local_expert_num;
            const int   idx       = atomicAdd(&token_idx_in_rank[ep_rank * (token_num + 2) + token], 1);

            const int index     = token * ep_size * top_k + ep_rank * top_k + idx;
            topk_experts[index] = expert_id - ep_rank * local_expert_num;
            topk_scales[index]  = scale * routed_scale;
        }
    }
}

struct BlockPrefixCallbackOp {
    int            running_total;
    __device__     BlockPrefixCallbackOp(int running_total): running_total(running_total) {}
    __device__ int operator()(int block_aggregate)
    {
        int old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template<int block_dim>
__global__ void MoeRankScanKernel(int* token_idx_in_rank, int tokens)
{
    const int ep_rank = blockIdx.x;
    token_idx_in_rank += ep_rank * (tokens + 2);

    const int token_padded = round_up(tokens, block_dim);

    using BlockScan = cub::BlockScan<int, block_dim>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockPrefixCallbackOp prefix_experts(0);
    BlockPrefixCallbackOp prefix_tokens(0);

    for (int i = threadIdx.x; i < token_padded; i += block_dim) {
        int experts;
        int token_selected_experts = (i < tokens) ? token_idx_in_rank[i] : 0;
        int token_idx              = (token_selected_experts > 0) ? 1 : 0;
        BlockScan(temp_storage).ExclusiveSum(token_idx, token_idx, prefix_tokens);
        BlockScan(temp_storage).ExclusiveSum(token_selected_experts, experts, prefix_experts);
        __syncthreads();
        if (i < tokens) {
            token_idx_in_rank[i] = (token_selected_experts > 0) ? token_idx : -1;
        }
    }

    if (threadIdx.x == 0) {
        token_idx_in_rank[tokens]     = prefix_tokens.running_total;
        token_idx_in_rank[tokens + 1] = prefix_experts.running_total;
    }
}

template<int N>
inline constexpr std::integral_constant<int, N> _Int{};

void invokeMoeGate_a2a(float*       topk_scales,
                       int*         topk_experts,
                       int*         token_idx_in_rank,
                       const float* logits,
                       int          tokens,
                       int          experts,
                       int          ep_size,
                       int          experts_per_token,
                       bool         softmax,
                       bool         norm_topk,
                       float        routed_scale,
                       cudaStream_t stream)
{

    check_cuda_error(cudaMemsetAsync(token_idx_in_rank, 0, sizeof(int) * ep_size * (tokens + 2), stream));
    if (tokens == 0) {
        return;
    }

    auto invoke = [&](auto max_expert_num, auto top_k, auto items_per_thread, auto vec_size) {
        constexpr int thrs_per_tok = max_expert_num.value / items_per_thread.value;
        constexpr int threads      = 256;
        const int     blocks       = ceil_div(tokens, threads / thrs_per_tok);

        check_cuda_error(cudaMemsetAsync(topk_scales, 0, sizeof(float) * tokens * ep_size * experts_per_token, stream));
        check_cuda_error(cudaMemsetAsync(topk_experts, -1, sizeof(int) * tokens * ep_size * experts_per_token, stream));

        MoeGateKernel<max_expert_num.value, top_k.value, items_per_thread.value, threads, vec_size.value>
            <<<blocks, threads, 0, stream>>>(  //
                topk_scales,
                topk_experts,
                token_idx_in_rank,
                logits,
                tokens,
                experts,
                experts / ep_size,
                ep_size,
                experts_per_token,
                softmax,
                norm_topk,
                routed_scale);
        sync_check_cuda_error();

        return true;
    };

    if (!softmax && norm_topk) {
        // norm top-k is part of softmax impl
        TM_CHECK(0) << softmax << " " << norm_topk;
    }

    auto dispatch = [&] {
        if (experts <= 8) {
            if (experts_per_token <= 2) {
                return invoke(_Int<8>, _Int<2>, _Int<8>, _Int<4>);
            }
            else {
                return invoke(_Int<8>, _Int<8>, _Int<8>, _Int<4>);
            }
        }
        else if (experts <= 64) {
            if (experts_per_token <= 4) {
                return invoke(_Int<64>, _Int<4>, _Int<16>, _Int<4>);
            }
            else if (experts_per_token <= 8) {
                return invoke(_Int<64>, _Int<8>, _Int<16>, _Int<4>);
            }
        }
        else if (experts <= 128) {
            if (experts_per_token <= 8) {
                return invoke(_Int<128>, _Int<8>, _Int<16>, _Int<4>);
            }
        }
        else if (experts <= 160) {
            if (experts_per_token <= 8) {
                return invoke(_Int<160>, _Int<8>, _Int<10>, _Int<2>);
            }
        }
        else if (experts <= 512) {
            if (experts_per_token <= 8) {
                return invoke(_Int<512>, _Int<8>, _Int<16>, _Int<4>);
            }
        }
        return false;
    };

    auto success = dispatch();

    TM_CHECK(success) << "unsupported moe config: expert_num=" << experts << ", top_k=" << experts_per_token
                      << ", softmax=" << softmax << ", norm_topk=" << norm_topk;

    {
        constexpr int threads = 256;
        MoeRankScanKernel<threads><<<ep_size, threads, 0, stream>>>(  //
            token_idx_in_rank,
            tokens);
    }
}

template<int block_dim>
__global__ void MoeComputeAccumKernel(int*          accum,  //
                                      const int8_t* masks,
                                      int           log_tile,
                                      int           tiles,
                                      int           token_num,
                                      int           token_num_padded)
{
    using BlockReduce = cub::BlockReduce<int, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tile_id   = blockIdx.x;
    const int expert_id = blockIdx.y;

    const int tile_size  = 1 << log_tile;
    const int tile_start = tile_id * tile_size;
    const int tile_end   = min(tile_start + tile_size, token_num);

    int count = 0;
    for (int token_idx = tile_start + threadIdx.x; token_idx < tile_end; token_idx += block_dim) {
        const int8_t mask_val = masks[expert_id * token_num_padded + token_idx];
        if (mask_val >= 0) {
            count++;
        }
    }

    int total_count = BlockReduce(temp_storage).Sum(count);

    if (threadIdx.x == 0) {
        accum[expert_id * tiles + tile_id] = total_count;
    }
}

void invokeMoeScan_a2a(int*         f2n,      // [e*n] -> n
                       int*         f2E,      // [e*n] -> E
                       int*         en2f,     // [e,n] -> n*e
                       int*         offsets,  // [E+1]
                       int8_t*      masks,    // [E,n]
                       int*         accum,    // [E,tiles]
                       int          token_num,
                       int          token_num_padded,
                       int          local_expert_num,
                       cudaStream_t stream)
{
    if (token_num == 0) {
        return;
    }
    constexpr int base_log_tile = 9;

    int log_tile = base_log_tile;
    while (((token_num + (1 << log_tile) - 1) >> log_tile) > kMoeGateMaxTiles) {
        ++log_tile;
    }
    const int tiles = ceil_div(token_num, 1 << log_tile);

    constexpr int block = 256;
    dim3          grid(tiles, local_expert_num);
    MoeComputeAccumKernel<block><<<grid, block, 0, stream>>>(  //
        accum,
        masks,
        log_tile,
        tiles,
        token_num,
        token_num_padded);
    sync_check_cuda_error();

    invokeMoeScan_v2(f2n,  //
                     f2E,
                     en2f,
                     offsets,
                     masks,
                     accum,
                     log_tile,
                     tiles,
                     token_num,
                     token_num_padded,
                     local_expert_num,
                     stream);
}

template<int vec_size, int exp_k, bool has_bias, int block_dim, class T>
__global__ void MoeReduceKernel(T*           dst,     // [  n, d]
                                const T*     src,     // [e*n, d]
                                const T*     bias,    // [  E, d]
                                const float* scales,  // [  e, n]
                                const int*   en2f,    // [  e, n] :: (e,n) -> e*n
                                const int*   f2E,     // [  e* n]
                                int          dim,
                                int          tokens)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int64_t ti = blockIdx.x;

        dst += dim * ti;

        // Should be warp uniforms
        const T* src_[exp_k]{};
        const T* bias_[exp_k];

        float scale[exp_k];

        PRAGMA_UNROLL
        for (int e = 0; e < exp_k; ++e) {
            int fid = __ldg(&en2f[e * tokens + ti]);
            if (fid >= 0) {
                src_[e] = src + dim * fid;
                if constexpr (has_bias) {
                    bias_[e] = bias + __ldg(&f2E[fid]) * dim;
                }
            }
            scale[e] = scales ? __ldg(&scales[e * tokens + ti]) : 1.f;
        }

        using Vec = Array<T, vec_size>;

        for (int i = threadIdx.x * vec_size; i < dim; i += block_dim * vec_size) {
            Array<float, vec_size> accum{};
            PRAGMA_UNROLL
            for (int e = 0; e < exp_k; ++e) {
                if (src_[e] == nullptr) {
                    continue;
                }
                Vec v{};
                Load(v, src_[e] + i);
                using namespace ops;
                if constexpr (has_bias) {
                    Vec b;
                    Load(b, bias_[e] + i);
                    v = v + b;
                }
                const auto x = cast<float>(v) * scale[e];
                accum        = accum + x;
            }
            Store(&dst[i], cast<T>(accum));
        }
    }
}

void invokeMoeCombine_a2a(Ref<Tensor>   out_,
                          const Tensor& src,
                          const Tensor& bias,
                          const float*  scales,
                          const int*    en2f,
                          const int*    f2E,
                          int           experts_per_token,
                          cudaStream_t  st)
{
    auto& out = out_.get();

    const int tokens = out.shape(0);

    auto invoke = [&](auto has_bias, auto t, auto e) {
        using T                   = decltype(t);
        constexpr int threads     = 256;
        constexpr int vec_size    = 16 / sizeof(T);
        constexpr int exp_per_tok = decltype(e)::value;
        MoeReduceKernel<vec_size, exp_per_tok, has_bias.value, threads><<<tokens, threads, 0, st>>>(  //
            out.data<T>(),
            src.data<T>(),
            bias.data_or((T*)nullptr),
            scales,
            en2f,
            f2E,
            src.shape(1),
            tokens);
    };

    auto dispatch_experts_per_token = [&](auto has_bias, auto t) {
        switch (experts_per_token) {
            case 1:
                return invoke(has_bias, t, std::integral_constant<int, 1>{});
            case 2:
                return invoke(has_bias, t, std::integral_constant<int, 2>{});
            case 4:
                return invoke(has_bias, t, std::integral_constant<int, 4>{});
            case 6:
                return invoke(has_bias, t, std::integral_constant<int, 6>{});
            case 8:
                return invoke(has_bias, t, std::integral_constant<int, 8>{});
            default:
                fprintf(stderr, "Unsupported experts_per_token %d\n", experts_per_token);
                std::abort();
        }
    };

    auto dispatch_dtype = [&](auto t) {
        if (bias) {
            TM_CHECK_NOTNULL(f2E);
            return dispatch_experts_per_token(std::true_type{}, t);
        }
        else {
            return dispatch_experts_per_token(std::false_type{}, t);
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(src.dtype(), dispatch_dtype);
}

}  // namespace turbomind
