// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/moe_ep_utils.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <algorithm>
#include <limits>
#include <type_traits>

namespace turbomind {

template<int max_expert_num, int max_top_k, int items_per_thread, int block_dim, int access_size>
__global__ void MoeGateKernel(float*       topk_weights,  // [n, topk]
                              int64_t*     topk_idx,      // [n, topk]
                              const float* logits,        // [n,E]
                              int          token_num,
                              int          expert_num,
                              int          top_k,
                              bool         softmax,
                              bool         norm_topk,
                              float        routed_scale)
{
    constexpr int threads_per_token = max_expert_num / items_per_thread;

    static_assert(items_per_thread <= 32);
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
            if (e + access_size <= expert_num) {
                Ldg((Array<float, access_size>&)data[i], &logits[ti * expert_num + e]);
            }
            else {
                PRAGMA_UNROLL
                for (int j = 0; j < access_size; ++j) {
                    if (e + j < expert_num) {
                        data[i + j] = logits[ti * expert_num + e + j];
                    }
                }
            }
        }
    }

    unsigned mask = (unsigned)-1;
    float    max_logit;

    const int warp_ti_offset = warp_ti * threads_per_token;

    int sel_item[max_top_k];

    auto run = [&](int k) {
        unsigned bit     = 1;
        unsigned max_bit = 0;
        float    max_val = -std::numeric_limits<float>::infinity();
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if ((mask & bit) && data[i] > max_val) {
                max_bit = bit;
                max_val = data[i];
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }

        int   g_max_ei  = ei;
        float g_max_val = max_val;
        if constexpr (threads_per_token > 1) {
            PRAGMA_UNROLL
            for (int m = threads_per_token / 2; m >= 1; m /= 2) {
                g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
            }
            const auto active = __ballot_sync((uint32_t)-1, max_val == g_max_val);
            g_max_ei          = __ffs(active >> (unsigned)warp_ti_offset) - 1;
        }
        if (k == 0) {
            max_logit = g_max_val;
        }
        int local_item = -1;
        if (ei == g_max_ei) {
            local_item = __ffs(max_bit) - 1;
            mask -= max_bit;
        }
        sel_item[k] = local_item;
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

    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int k = 0; k < max_top_k; ++k) {
            if (k < top_k && sel_item[k] >= 0) {
                const int i                  = sel_item[k];
                topk_weights[ti * top_k + k] = data[i] * sum_prob * routed_scale;
                topk_idx[ti * top_k + k]     = idxs[i];
            }
        }
    }
}

template<int N>
inline constexpr std::integral_constant<int, N> _Int{};

void invokeMoeGateEp(float*       topk_weights,
                     int64_t*     topk_idx,
                     const float* logits,
                     int          tokens,
                     int          experts,
                     int          experts_per_token,
                     bool         softmax,
                     bool         norm_topk,
                     float        routed_scale,
                     cudaStream_t stream)
{
    if (tokens == 0) {
        return;
    }

    auto invoke = [&](auto max_expert_num, auto top_k, auto items_per_thread, auto vec_size) {
        constexpr int thrs_per_tok = max_expert_num.value / items_per_thread.value;
        constexpr int threads      = 256;
        const int     blocks       = ceil_div(tokens, threads / thrs_per_tok);

        MoeGateKernel<max_expert_num.value, top_k.value, items_per_thread.value, threads, vec_size.value>
            <<<blocks, threads, 0, stream>>>(
                topk_weights, topk_idx, logits, tokens, experts, experts_per_token, softmax, norm_topk, routed_scale);
        TM_CUDA_CHECK(cudaGetLastError());

        return true;
    };

    if (!softmax && norm_topk) {
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
            if (experts_per_token <= 10) {
                return invoke(_Int<512>, _Int<10>, _Int<16>, _Int<4>);
            }
        }
        return false;
    };

    auto success = dispatch();

    TM_CHECK(success) << "unsupported moe config: expert_num=" << experts << ", top_k=" << experts_per_token
                      << ", softmax=" << softmax << ", norm_topk=" << norm_topk;
}

template<int block_dim, int max_local_experts>
__global__ void MoeAgRsBuildMasksKernel(int8_t*        masks,     // [num_local_experts, tokens_padded], pre-set to -1
                                        int*           accum,     // [num_local_experts, tiles], pre-set to 0
                                        const int64_t* topk_idx,  // [tokens, topk]
                                        int            tokens,
                                        int            tokens_padded,
                                        int            topk,
                                        int            local_expert_offset,
                                        int            num_local_experts,
                                        int            log_tile,
                                        int            tiles)
{
    __shared__ int shared_accum[kMoeGateMaxTiles][max_local_experts + 1];

    // Clear shared_accum
    for (int i = threadIdx.x; i < tiles * num_local_experts; i += block_dim) {
        shared_accum[i / num_local_experts][i % num_local_experts] = 0;
    }
    __syncthreads();

    // Each thread processes one token
    const int ti = threadIdx.x + blockIdx.x * blockDim.x;
    if (ti < tokens) {
        for (int k = 0; k < topk; ++k) {
            const int global_eid = static_cast<int>(topk_idx[ti * topk + k]);
            const int local_eid  = global_eid - local_expert_offset;
            if (0 <= local_eid && local_eid < num_local_experts) {
                masks[local_eid * tokens_padded + ti] = static_cast<int8_t>(k);
                atomicAdd(&shared_accum[ti >> log_tile][local_eid], 1);
            }
        }
    }
    __syncthreads();

    // Flush shared_accum to global accum
    for (int i = threadIdx.x; i < tiles * num_local_experts; i += block_dim) {
        int t = i / num_local_experts;
        int e = i % num_local_experts;
        if (t < tiles && shared_accum[t][e]) {
            atomicAdd(&accum[e * tiles + t], shared_accum[t][e]);
        }
    }
}

void invokeMoeBuildAgRsRoutingMap(int*           f2n,
                                  int*           f2E,
                                  int*           en2f,
                                  int*           offsets,
                                  int8_t*        masks,
                                  int*           accum,
                                  const int64_t* topk_idx,
                                  int            tokens,
                                  int            topk,
                                  int            local_expert_offset,
                                  int            num_local_experts,
                                  cudaStream_t   stream)
{
    if (tokens == 0) {
        TM_CUDA_CHECK(cudaMemsetAsync(offsets, 0, sizeof(int) * (num_local_experts + 1), stream));
        return;
    }

    const int tokens_padded = round_up(tokens, kMoeGateVecSize);

    // Compute log_tile / tiles (must match invokeMoeScanKernel's internal computation)
    constexpr int base_log_tile = 9;
    int           log_tile      = base_log_tile;
    while (((tokens_padded + (1 << log_tile) - 1) >> log_tile) > kMoeGateMaxTiles) {
        ++log_tile;
    }
    const int tiles = ceil_div(tokens_padded, 1 << log_tile);

    // Initialise buffers
    TM_CUDA_CHECK(cudaMemsetAsync(masks, -1, sizeof(int8_t) * num_local_experts * tokens_padded, stream));
    TM_CUDA_CHECK(cudaMemsetAsync(accum, 0, sizeof(int) * num_local_experts * kMoeGateMaxTiles, stream));
    TM_CUDA_CHECK(cudaMemsetAsync(en2f, -1, sizeof(int) * tokens * topk, stream));

    constexpr int block = 256;
    const int     grid  = cdiv(tokens_padded, block);

    auto invoke = [&](auto max_local_experts) {
        MoeAgRsBuildMasksKernel<block, max_local_experts.value><<<grid, block, 0, stream>>>(masks,
                                                                                            accum,
                                                                                            topk_idx,
                                                                                            tokens,
                                                                                            tokens_padded,
                                                                                            topk,
                                                                                            local_expert_offset,
                                                                                            num_local_experts,
                                                                                            log_tile,
                                                                                            tiles);
        return true;
    };

    auto dispatch = [&] {
        if (num_local_experts <= 4) {
            return invoke(_Int<4>);
        }
        else if (num_local_experts <= 8) {
            return invoke(_Int<8>);
        }
        else if (num_local_experts <= 16) {
            return invoke(_Int<16>);
        }
        else if (num_local_experts <= 32) {
            return invoke(_Int<32>);
        }
        else if (num_local_experts <= 64) {
            return invoke(_Int<64>);
        }
        else if (num_local_experts <= 128) {
            return invoke(_Int<128>);
        }
        return false;
    };

    auto success = dispatch();
    TM_CHECK(success) << "unsupported num_local_experts: " << num_local_experts;
    TM_CUDA_CHECK(cudaGetLastError());

    invokeMoeScanKernel(f2n, f2E, en2f, offsets, masks, accum, tokens, tokens_padded, num_local_experts, stream);
}

template<int vec_size, int exp_k, bool has_bias, int block_dim, class T>
__global__ void MoeCombineKernel(T*           dst,           // [num_tokens, dim]
                                 const T*     src,           // [expert_token_num, dim]
                                 const T*     bias,          // [num_local_experts, dim]
                                 const float* topk_weights,  // [num_tokens, topk]
                                 const int*   en2f,          // [topk, num_tokens]
                                 const int*   f2E,           // [expert_token_num]
                                 int          dim,
                                 int          tokens)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        for (int ti = blockIdx.x; ti < tokens; ti += gridDim.x) {
            T* dst_row = dst + (int64_t)dim * ti;

            const T* src_[exp_k]{};
            const T* bias_[exp_k]{};
            float    weight[exp_k]{};

            PRAGMA_UNROLL
            for (int e = 0; e < exp_k; ++e) {
                const int fid = __ldg(&en2f[e * tokens + ti]);
                if (fid >= 0) {
                    src_[e]   = src + (int64_t)dim * fid;
                    weight[e] = __ldg(&topk_weights[ti * exp_k + e]);
                    if constexpr (has_bias) {
                        bias_[e] = bias + (int64_t)dim * __ldg(&f2E[fid]);
                    }
                }
            }

            using Vec = Array<T, vec_size>;

            for (int i = threadIdx.x * vec_size; i < dim; i += block_dim * vec_size) {
                Array<float, vec_size> accum{};
                PRAGMA_UNROLL
                for (int e = 0; e < exp_k; ++e) {
                    if (src_[e] == nullptr) {
                        continue;
                    }
                    Vec v;
                    Load(v, src_[e] + i);
                    if constexpr (has_bias) {
                        Vec b;
                        Load(b, bias_[e] + i);
                        PRAGMA_UNROLL
                        for (int j = 0; j < vec_size; ++j) {
                            v[j] = (T)((float)v[j] + (float)b[j]);
                        }
                    }
                    using namespace ops;
                    const auto x = cast<float>(v) * weight[e];
                    accum        = accum + x;
                }
                Store(&dst_row[i], cast<T>(accum));
            }
        }
    }
}

void invokeMoeLocalCombineEp(Ref<Tensor>   out_,
                             const Tensor& src,
                             const Tensor& bias,
                             const float*  topk_weights,
                             const int*    en2f,
                             const int*    f2E,
                             int           experts_per_token,
                             cudaStream_t  st)
{
    auto& out = out_.get();

    const int tokens = out.shape(0);

    if (tokens == 0) {
        return;
    }

    const int dim = src.shape(1);

    auto invoke = [&](auto t, auto e, auto has_bias_) {
        using T                    = decltype(t);
        constexpr int  threads     = 256;
        constexpr int  vec_size    = sizeof(uint4) / sizeof(T);
        constexpr int  exp_per_tok = decltype(e)::value;
        constexpr bool has_bias    = decltype(has_bias_)::value;

        static const int sm_count = getSMCount();
        const int        grid     = std::min<int>(tokens, sm_count * 4);

        MoeCombineKernel<vec_size, exp_per_tok, has_bias, threads><<<grid, threads, 0, st>>>(
            out.data<T>(), src.data<T>(), bias.data_or((T*)nullptr), topk_weights, en2f, f2E, dim, tokens);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    auto dispatch_topk = [&](auto has_bias, auto t) {
        switch (experts_per_token) {
            case 1:
                return invoke(t, std::integral_constant<int, 1>{}, has_bias);
            case 2:
                return invoke(t, std::integral_constant<int, 2>{}, has_bias);
            case 4:
                return invoke(t, std::integral_constant<int, 4>{}, has_bias);
            case 6:
                return invoke(t, std::integral_constant<int, 6>{}, has_bias);
            case 8:
                return invoke(t, std::integral_constant<int, 8>{}, has_bias);
            case 10:
                return invoke(t, std::integral_constant<int, 10>{}, has_bias);
            default:
                TM_CHECK(0) << "unsupported experts_per_token " << experts_per_token;
        }
    };

    auto dispatch_dtype = [&](auto t) {
        if (bias) {
            TM_CHECK_NOTNULL(f2E);
            return dispatch_topk(std::true_type{}, t);
        }
        else {
            return dispatch_topk(std::false_type{}, t);
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(src.dtype(), dispatch_dtype);
}

template<int vec_size, int block_dim, class T>
__global__ void MoeCombineOutputEpKernel(T*           dst,            // [tokens, dim]
                                         const T*     shared,         // [tokens, dim]
                                         const float* shared_scales,  // [tokens] or nullptr
                                         int          dim,
                                         float        scale)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int ti = blockIdx.x;

        float dst_scale = scale;
        if (shared_scales) {
            dst_scale = __ldg(&shared_scales[ti]);
            dst_scale = fdividef(1.f, 1.f + expf(-dst_scale));
        }

        dst += (int64_t)dim * ti;
        shared += (int64_t)dim * ti;

        using Vec = Array<T, vec_size>;

        for (int i = threadIdx.x * vec_size; i < dim; i += block_dim * vec_size) {
            Vec routed;
            Load(routed, &dst[i]);
            using namespace ops;
            Array<float, vec_size> accum = cast<float>(routed);
            {
                Vec v;
                Load(v, &shared[i]);
                accum = accum + cast<float>(v) * dst_scale;
            }
            Store(&dst[i], cast<T>(accum));
        }
    }
}

void invokeMoeCombineOutputEp(
    Ref<Tensor> output, const Tensor& shared, const float* shared_scales, float scale, cudaStream_t st)
{
    auto& out = output.get();

    TM_CHECK(shared);
    TM_CHECK_EQ(shared.shape(0), out.shape(0));
    TM_CHECK_EQ(shared.shape(1), out.shape(1));

    const int tokens = out.shape(0);
    const int dim    = out.shape(1);

    if (tokens == 0) {
        return;
    }

    if (shared_scales == nullptr && scale == 0) {
        return;
    }

    auto dispatch = [&](auto t) {
        using T                = decltype(t);
        constexpr int threads  = 256;
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        MoeCombineOutputEpKernel<vec_size, threads>
            <<<tokens, threads, 0, st>>>(out.data<T>(), shared.data<T>(), shared_scales, dim, scale);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    TM_DISPATCH_PRIMARY_DTYPES(out.dtype(), dispatch);
}

}  // namespace turbomind
