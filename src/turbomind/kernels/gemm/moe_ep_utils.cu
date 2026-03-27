// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/moe_ep_utils.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cub/block/block_scan.cuh>

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

    int sel_item[max_top_k];

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
            <<<blocks, threads, 0, stream>>>(  //
                topk_weights,
                topk_idx,
                logits,
                tokens,
                experts,
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
}

// Kernel: compute f2n, f2E, en2f from recv_topk_idx after EP dispatch.
// One CTA per local expert. Each CTA scans all received tokens in chunks,
template<int block_dim>
__global__ void MoeEpRoutingMapKernel(
    int* f2n, int* f2E, int* en2f, const int* offsets, const int64_t* recv_topk_idx, int num_tokens, int topk)
{
    using BlockScan = cub::BlockScan<int, block_dim>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int local_eid = blockIdx.x;

    int write_offset = offsets[local_eid];

    // All threads iterate the same number of chunks (base is thread-independent).
    // Threads with ti >= num_tokens contribute flag=0 to BlockScan.
    const int num_chunks = ceil_div(num_tokens, block_dim);
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int ti = chunk * block_dim + threadIdx.x;

        // Check if this token is assigned to this expert
        int match_k = -1;
        if (ti < num_tokens) {
            for (int k = 0; k < topk; ++k) {
                if (static_cast<int>(recv_topk_idx[ti * topk + k]) == local_eid) {
                    match_k = k;
                    break;
                }
            }
        }

        int flag = (match_k >= 0) ? 1 : 0;
        int prefix;
        int block_total;
        BlockScan(temp_storage).ExclusiveSum(flag, prefix, block_total);
        __syncthreads();

        if (match_k >= 0) {
            const int flat_id               = write_offset + prefix;
            f2n[flat_id]                    = ti;
            f2E[flat_id]                    = local_eid;
            en2f[match_k * num_tokens + ti] = flat_id;
        }

        write_offset += block_total;
    }
}

void invokeMoeRoutingMapEp(int*           f2n,
                           int*           f2E,
                           int*           en2f,
                           int*           offsets,
                           const int64_t* recv_topk_idx,
                           int            num_tokens,
                           int            topk,
                           int            num_local_experts,
                           cudaStream_t   stream)
{
    if (num_tokens == 0) {
        return;
    }

    constexpr int block = 256;
    check_cuda_error(cudaMemsetAsync(en2f, -1, sizeof(int) * num_tokens * topk, stream));
    // One CTA per local expert
    MoeEpRoutingMapKernel<block>
        <<<num_local_experts, block, 0, stream>>>(f2n, f2E, en2f, offsets, recv_topk_idx, num_tokens, topk);
    sync_check_cuda_error();
}

template<int vec_size, int block_dim, class T>
__global__ void MoeAddBiasKernel(T* dst, const T* bias, const int* f2E, int dim)
{
    if constexpr (TURBOMIND_ARCH_DTYPE_GUARD(data_type_v<T>)) {
        const int ti = blockIdx.x;

        dst += (int64_t)dim * ti;
        bias += (int64_t)dim * __ldg(&f2E[ti]);

        using Vec = Array<T, vec_size>;

        for (int i = threadIdx.x * vec_size; i < dim; i += block_dim * vec_size) {
            Vec x;
            Vec b;
            Load(x, dst + i);
            Load(b, bias + i);
            PRAGMA_UNROLL
            for (int j = 0; j < vec_size; ++j) {
                x[j] = (T)((float)x[j] + (float)b[j]);
            }
            Store(dst + i, x);
        }
    }
}

void invokeMoeAddBias(Ref<Tensor> out_, const Tensor& bias, const int* f2E, cudaStream_t st)
{
    auto& out = out_.get();

    if (!bias || out.shape(0) == 0) {
        return;
    }

    TM_CHECK_NOTNULL(f2E);
    TM_CHECK_EQ(out.shape(1), bias.shape(1));
    TM_CHECK_EQ(out.dtype(), bias.dtype());

    const int tokens = out.shape(0);
    const int dim    = out.shape(1);

    auto dispatch = [&](auto t) {
        using T                = decltype(t);
        constexpr int threads  = 256;
        constexpr int vec_size = 16 / sizeof(T);

        TM_CHECK_EQ(dim % vec_size, 0);

        MoeAddBiasKernel<vec_size, threads><<<tokens, threads, 0, st>>>(out.data<T>(), bias.data<T>(), f2E, dim);
        sync_check_cuda_error();
    };

    TM_DISPATCH_PRIMARY_DTYPES(out.dtype(), dispatch);
}

// Combine kernel for EP mode: one CTA per received token.
// For each token, gather expert outputs weighted by topk_weights and sum them.
// en2f[k * tokens + ti] gives the flat index in src for token ti's k-th expert slot,
// or -1 if no local expert matched that slot.
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
        const int ti = blockIdx.x;

        dst += (int64_t)dim * ti;

        // Gather source pointers and weights for this token's expert slots
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
            Store(&dst[i], cast<T>(accum));
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

    auto dispatch_topk = [&](auto has_bias, auto t) {
        using T               = decltype(t);
        constexpr int threads = 256;
        constexpr int vsize   = 16 / sizeof(T);

        auto invoke = [&](auto e) {
            constexpr int exp_per_tok = decltype(e)::value;
            MoeCombineKernel<vsize, exp_per_tok, has_bias.value, threads><<<tokens, threads, 0, st>>>(  //
                out.data<T>(),
                src.data<T>(),
                bias.data_or((T*)nullptr),
                topk_weights,
                en2f,
                f2E,
                dim,
                tokens);
            sync_check_cuda_error();
        };

        switch (experts_per_token) {
            case 1:
                return invoke(std::integral_constant<int, 1>{});
            case 2:
                return invoke(std::integral_constant<int, 2>{});
            case 4:
                return invoke(std::integral_constant<int, 4>{});
            case 6:
                return invoke(std::integral_constant<int, 6>{});
            case 8:
                return invoke(std::integral_constant<int, 8>{});
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
                                         const T*     src,            // [tokens, dim]
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
        src += (int64_t)dim * ti;

        using Vec = Array<T, vec_size>;

        for (int i = threadIdx.x * vec_size; i < dim; i += block_dim * vec_size) {
            Array<float, vec_size> accum{};
            if (dst_scale) {
                Vec v;
                Load(v, &dst[i]);
                using namespace ops;
                accum = cast<float>(v) * dst_scale;
            }
            {
                Vec v;
                Load(v, &src[i]);
                using namespace ops;
                accum = accum + cast<float>(v);
            }
            Store(&dst[i], cast<T>(accum));
        }
    }
}

void invokeMoeCombineOutputEp(
    Ref<Tensor> output, const Tensor& src, const float* shared_scales, float scale, cudaStream_t st)
{
    auto& out = output.get();

    TM_CHECK_EQ(src.shape(0), out.shape(0));
    TM_CHECK_EQ(src.shape(1), out.shape(1));

    const int tokens = out.shape(0);
    const int dim    = out.shape(1);

    if (tokens == 0) {
        return;
    }

    if (shared_scales == nullptr && scale == 0) {
        TM_CHECK_EQ(src.byte_size(), out.byte_size());
        cudaMemcpyAsync(out.raw_data(), src.raw_data(), out.byte_size(), cudaMemcpyDefault, st);
        return;
    }

    auto dispatch = [&](auto t) {
        using T                = decltype(t);
        constexpr int threads  = 256;
        constexpr int vec_size = 16 / sizeof(T);
        MoeCombineOutputEpKernel<vec_size, threads><<<tokens, threads, 0, st>>>(  //
            out.data<T>(),
            src.data<T>(),
            shared_scales,
            dim,
            scale);
        sync_check_cuda_error();
    };

    TM_DISPATCH_PRIMARY_DTYPES(src.dtype(), dispatch);
}

__global__ void MoeLLDispatchRoutingMapKernel(int* moe_recv_counter_mapped,  //
                                              int* f2n,
                                              int* f2E,
                                              const int* __restrict__ offsets)
{
    const int ei    = blockIdx.x;
    const int begin = offsets[ei];
    const int end   = offsets[ei + 1];

    if (ei == gridDim.x - 1 && threadIdx.x == 0) {
        *moe_recv_counter_mapped = end;
    }

    for (int idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
        f2n[idx] = idx;
        f2E[idx] = ei;
    }
}

__global__ void MoeLLDispatchCopyKernel(int4* out,
                                        const int4* __restrict__ x,
                                        int hidden_int4,
                                        const int* __restrict__ offsets,
                                        int num_max_tokens,
                                        int num_local_experts)
{
    int row = blockIdx.x;

    int lo = 0;
    int hi = num_local_experts;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }

    const int   src_row = row - offsets[lo];
    const int4* src     = x + (lo * num_max_tokens + src_row) * hidden_int4;
    int4*       dst     = out + row * hidden_int4;
    for (int i = threadIdx.x; i < hidden_int4; i += blockDim.x) {
        __stcg(dst + i, __ldcg(src + i));
    }
}

void invokeMoeLLDispatchPostprocess(Tensor&       out,  //
                                    int*          f2n,
                                    int*          f2E,
                                    const int*    offsets,
                                    volatile int* moe_recv_counter,
                                    int*          moe_recv_counter_mapped,
                                    Tensor&       packed_recv_x,
                                    cudaStream_t  st)
{
    const int num_local_experts = packed_recv_x.shape(0);
    const int num_max_tokens    = packed_recv_x.shape(1);
    const int hidden            = packed_recv_x.shape(2);
    const int threads           = 256;

    *moe_recv_counter = -1;
    MoeLLDispatchRoutingMapKernel<<<num_local_experts, threads, 0, st>>>(moe_recv_counter_mapped, f2n, f2E, offsets);
    sync_check_cuda_error();
    core::Context::stream().Sync();

    while (*moe_recv_counter < 0) {};
    out = Tensor({*moe_recv_counter, hidden}, packed_recv_x.dtype(), packed_recv_x.device());
    TM_CHECK_EQ(hidden * byte_size(packed_recv_x.dtype()) % sizeof(int4), 0LL);
    const int hidden_int4 = hidden * byte_size(packed_recv_x.dtype()) / sizeof(int4);
    if (*moe_recv_counter > 0) {
        MoeLLDispatchCopyKernel<<<*moe_recv_counter, threads, 0, st>>>((int4*)out.raw_data(),
                                                                       (const int4*)packed_recv_x.raw_data(),
                                                                       hidden_int4,
                                                                       offsets,
                                                                       num_max_tokens,
                                                                       num_local_experts);
    }
}

__global__ void MoeLLCombinePreprocessKernel(int4* out,
                                             const int4* __restrict__ x,
                                             int hidden_int4,
                                             const int* __restrict__ offsets,
                                             int num_max_tokens,
                                             int num_local_experts)
{
    int row = blockIdx.x;

    int lo = 0;
    int hi = num_local_experts;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }

    const int   dst_row = row - offsets[lo];
    const int4* src     = x + row * hidden_int4;
    int4*       dst     = out + (lo * num_max_tokens + dst_row) * hidden_int4;
    for (int i = threadIdx.x; i < hidden_int4; i += blockDim.x) {
        __stcg(dst + i, __ldcg(src + i));
    }
}

void invokeMoeLLCombinePreprocess(Tensor& out, const Tensor& src, const int* offsets, cudaStream_t st)
{
    const int tokens = src.shape(0);
    if (tokens == 0) {
        return;
    }

    const int num_max_tokens    = out.shape(1);
    const int num_local_experts = out.shape(0);
    const int hidden            = src.shape(1);

    TM_CHECK_EQ(hidden * byte_size(src.dtype()) % sizeof(int4), 0LL);
    const int hidden_int4 = hidden * byte_size(src.dtype()) / sizeof(int4);

    const int threads = 256;
    MoeLLCombinePreprocessKernel<<<tokens, threads, 0, st>>>(
        (int4*)out.raw_data(), (const int4*)src.raw_data(), hidden_int4, offsets, num_max_tokens, num_local_experts);
}

}  // namespace turbomind
