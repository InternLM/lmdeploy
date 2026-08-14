#include "src/turbomind/comm/nccl/deepep/moe_a2a_utils.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

#include <cub/warp/warp_scan.cuh>

#define DEVICE_ASSERT(cond)                                                                                            \
    do {                                                                                                               \
        if (not(cond)) {                                                                                               \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);                             \
            asm("trap;");                                                                                              \
        }                                                                                                              \
    } while (0)

namespace turbomind {

MoeA2AInputPartition GetMoeA2AInputPartition(const std::vector<int>& local_token_nums,  //
                                             int                     ep_size,
                                             int                     ep_rank,
                                             int                     mlp_tp_size)
{
    TM_CHECK(!local_token_nums.empty());

    const int world_size = ep_size * mlp_tp_size;
    TM_CHECK_EQ(world_size % (int)local_token_nums.size(), 0);
    const int inner_tp = world_size / local_token_nums.size();

    std::vector<int> offsets(local_token_nums.size() + 1);
    std::inclusive_scan(local_token_nums.begin(), local_token_nums.end(), offsets.begin() + 1);

    auto TaskRange = [&](int task_id) {
        const int local_id = task_id / inner_tp;
        const int num      = local_token_nums[local_id];
        const int slice    = num / inner_tp + (num % inner_tp != 0);
        const int first    = std::min(num, task_id % inner_tp * slice);
        const int last     = std::min(num, first + slice);
        return std::pair{offsets[local_id] + first, offsets[local_id] + last};
    };

    MoeA2AInputPartition res{};
    for (int rank = 0; rank < ep_size; ++rank) {
        const int first_task    = rank * mlp_tp_size;
        const int last_task     = first_task + mlp_tp_size - 1;
        const int begin         = TaskRange(first_task).first;
        const int end           = TaskRange(last_task).second;
        res.max_tokens_per_rank = std::max(res.max_tokens_per_rank, end - begin);
        if (ep_rank == rank) {
            res.begin = begin;
            res.size  = end - begin;
        }
    }
    return res;
}

int CeilPowerOfTwo(int x)
{
    if (x <= 1) {
        return 1;
    }

    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

template<int max_expert_num, int max_top_k, int items_per_thread, int block_dim, int access_size>
__global__ void MoeA2AGateKernel(float*       topk_weights,
                                 int*         topk_indices,
                                 const float* logits,
                                 int          token_num,
                                 int          expert_num,
                                 int          top_k,
                                 bool         softmax,
                                 bool         norm_topk,
                                 float        routed_scale)
{
    constexpr int threads_per_token = max_expert_num / items_per_thread;
    constexpr int tokens_per_cta    = block_dim / threads_per_token;

    static_assert(items_per_thread <= 32);
    static_assert(items_per_thread % access_size == 0);
    static_assert(threads_per_token <= WARP_SIZE);
    static_assert((threads_per_token & (threads_per_token - 1)) == 0);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ti         = thread_idx / threads_per_token;
    const int ei         = thread_idx % threads_per_token;
    const int bti        = threadIdx.x / threads_per_token;

    float data[items_per_thread];
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        data[i] = -std::numeric_limits<float>::infinity();
    }

    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;
            if (e < expert_num) {
                Ldg((Array<float, access_size>&)data[i], logits + ti * expert_num + e);
            }
        }
    }

    unsigned mask = (unsigned)-1;
    float    max_logit;

    int count{};

    const int warp_ti        = threadIdx.x % WARP_SIZE / threads_per_token;
    const int warp_ti_offset = warp_ti * threads_per_token;

    auto run = [&](int k) {
        unsigned bit     = 1;
        unsigned max_bit = 0;
        float    max_val = -std::numeric_limits<float>::infinity();

        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            const int e = ei * items_per_thread + i;
            if (e < expert_num && (mask & bit) && data[i] > max_val) {
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
        if (ei == g_max_ei) {
            mask -= max_bit;
            ++count;
        }
    };

    run(0);

    for (int k = 1; k < top_k; ++k) {
        run(k);
    }

    mask = ~mask;

    float sum_prob = 0.f;
    if (softmax) {
        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            const int  e    = ei * items_per_thread + i;
            const bool used = mask & bit;
            if (e < expert_num && (!norm_topk || used)) {
                data[i] = expf(data[i] - max_logit);
                sum_prob += data[i];
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
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

    using WarpScan = cub::WarpScan<int, threads_per_token>;
    __shared__ typename WarpScan::TempStorage temp_storage[tokens_per_cta];

    int idx{};
    WarpScan{temp_storage[bti]}.ExclusiveSum(count, idx);
    const int end = idx + count;

    if (ti < token_num) {
        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (mask & bit) {
                const int e       = ei * items_per_thread + i;
                const int dst     = ti * top_k + idx++;
                topk_indices[dst] = e;
                topk_weights[dst] = data[i] * sum_prob * routed_scale;
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }
        // Prevent illegal indices caused by NaN values in the logits.
        for (; idx < end; ++idx) {
            const int dst     = ti * top_k + idx;
            topk_indices[dst] = -1;
            topk_weights[dst] = 0.f;
        }
    }
}

template<int N>
inline constexpr std::integral_constant<int, N> Int{};

void invokeMoeA2AGate(float*       topk_weights,
                      int*         topk_indices,
                      const float* logits,
                      int          tokens,
                      int          experts,
                      int          experts_per_token,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      cudaStream_t stream)
{
    TM_CHECK(softmax || !norm_topk) << "top-k normalization requires softmax";

    if (tokens == 0) {
        return;
    }

    auto invoke = [&](auto max_expert_num, auto max_top_k, auto items_per_thread, auto access_size) {
        constexpr int threads           = 256;
        constexpr int threads_per_token = max_expert_num.value / items_per_thread.value;
        constexpr int tokens_per_cta    = threads / threads_per_token;
        const int     blocks            = (tokens + tokens_per_cta - 1) / tokens_per_cta;

        MoeA2AGateKernel<max_expert_num.value, max_top_k.value, items_per_thread.value, threads, access_size.value>
            <<<blocks, threads, 0, stream>>>(topk_weights,
                                             topk_indices,
                                             logits,
                                             tokens,
                                             experts,
                                             experts_per_token,
                                             softmax,
                                             norm_topk,
                                             routed_scale);
        TM_CUDA_CHECK(cudaGetLastError());
        return true;
    };

    bool success = false;
    if (experts <= 8) {
        if (experts_per_token <= 2) {
            success = invoke(Int<8>, Int<2>, Int<8>, Int<4>);
        }
        else if (experts_per_token <= 8) {
            success = invoke(Int<8>, Int<8>, Int<8>, Int<4>);
        }
    }
    else if (experts <= 64) {
        if (experts_per_token <= 4) {
            success = invoke(Int<64>, Int<4>, Int<16>, Int<4>);
        }
        else if (experts_per_token <= 8) {
            success = invoke(Int<64>, Int<8>, Int<16>, Int<4>);
        }
    }
    else if (experts <= 128 && experts_per_token <= 8) {
        success = invoke(Int<128>, Int<8>, Int<16>, Int<4>);
    }
    else if (experts <= 160 && experts_per_token <= 8) {
        success = invoke(Int<160>, Int<8>, Int<10>, Int<2>);
    }
    else if (experts <= 512 && experts_per_token <= 10) {
        success = invoke(Int<512>, Int<10>, Int<16>, Int<4>);
    }

    TM_CHECK(success) << "unsupported A2A gate config: expert_num=" << experts << ", top_k=" << experts_per_token;
}

template<int expert_num, int top_k, int items_per_thread, int block_dim, int access_size>
__global__ void MoeA2AGateNoAuxTCKernel(float*       topk_weights,
                                        int*         topk_indices,
                                        const float* logits,
                                        const float* correction_bias,
                                        int          token_num,
                                        bool         norm_topk,
                                        float        routed_scale,
                                        bool         use_sigmoid)
{
    constexpr int threads_per_token = expert_num / items_per_thread;
    constexpr int tokens_per_cta    = block_dim / threads_per_token;

    static_assert(expert_num % items_per_thread == 0);
    static_assert(items_per_thread <= 32);
    static_assert(items_per_thread % access_size == 0);
    static_assert(block_dim % threads_per_token == 0);
    static_assert(threads_per_token <= WARP_SIZE);
    static_assert((threads_per_token & (threads_per_token - 1)) == 0);
    static_assert(WARP_SIZE % threads_per_token == 0);

    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ti         = thread_idx / threads_per_token;
    const int ei         = thread_idx % threads_per_token;
    const int bti        = threadIdx.x / threads_per_token;

    float scores[items_per_thread];
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        scores[i] = -std::numeric_limits<float>::infinity();
    }

    if (ti < token_num) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; i += access_size) {
            const int e = ei * items_per_thread + i;
            Ldg((Array<float, access_size>&)scores[i], logits + ti * expert_num + e);
        }
    }

    if (use_sigmoid) {
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            scores[i] = fdividef(1.f, 1.f + expf(-scores[i]));
        }
    }
    else {
        float max_logit = -std::numeric_limits<float>::infinity();
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            max_logit = fmaxf(max_logit, scores[i]);
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            max_logit = fmaxf(max_logit, __shfl_xor_sync((uint32_t)-1, max_logit, m));
        }

        float sum_prob = 0.f;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            scores[i] = expf(scores[i] - max_logit);
            sum_prob += scores[i];
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            sum_prob += __shfl_xor_sync((uint32_t)-1, sum_prob, m);
        }

        const float inv_sum = fdividef(1.f, sum_prob);
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            scores[i] *= inv_sum;
        }
    }

    float choice_scores[items_per_thread];
    PRAGMA_UNROLL
    for (int i = 0; i < items_per_thread; ++i) {
        const int e = ei * items_per_thread + i;
        float     v = scores[i] + (correction_bias ? __ldg(correction_bias + e) : 0.f);
        if (!isfinite(v)) {
            v = -std::numeric_limits<float>::infinity();
        }
        choice_scores[i] = v;
    }

    unsigned selected_mask = 0;

    const int warp_ti        = threadIdx.x % WARP_SIZE / threads_per_token;
    const int warp_ti_offset = warp_ti * threads_per_token;

    int count{};

    auto run = [&] {
        unsigned max_bit = 0;
        float    max_val = -std::numeric_limits<float>::infinity();

        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            const bool  available = !(selected_mask & bit);
            const float v         = choice_scores[i];
            if (available && (!max_bit || v > max_val)) {
                max_bit = bit;
                max_val = v;
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }

        float g_max_val = max_val;
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            g_max_val = fmaxf(g_max_val, __shfl_xor_sync((uint32_t)-1, g_max_val, m));
        }

        constexpr unsigned subgroup_mask = (unsigned)-1 >> (WARP_SIZE - threads_per_token);
        const auto         active =
            (__ballot_sync((uint32_t)-1, max_bit && max_val == g_max_val) >> (unsigned)warp_ti_offset) & subgroup_mask;
        const int g_max_ei = __ffs(active) - 1;
        if (ei == g_max_ei) {
            selected_mask |= max_bit;
            ++count;
        }
    };

    run();

    PRAGMA_UNROLL
    for (int k = 1; k < top_k; ++k) {
        run();
    }

    float selected_sum = 0.f;
    if (norm_topk) {
        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (selected_mask & bit) {
                selected_sum += scores[i];
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }
        PRAGMA_UNROLL
        for (int m = threads_per_token / 2; m >= 1; m /= 2) {
            selected_sum += __shfl_xor_sync((uint32_t)-1, selected_sum, m);
        }
    }
    const float scale = norm_topk && selected_sum > 1e-20f ? routed_scale / selected_sum : routed_scale;

    using WarpScan = cub::WarpScan<int, threads_per_token>;
    __shared__ typename WarpScan::TempStorage temp_storage[tokens_per_cta];

    int output_idx{};
    WarpScan{temp_storage[bti]}.ExclusiveSum(count, output_idx);

    if (ti < token_num) {
        unsigned bit = 1;
        PRAGMA_UNROLL
        for (int i = 0; i < items_per_thread; ++i) {
            if (selected_mask & bit) {
                const int dst     = ti * top_k + output_idx++;
                topk_indices[dst] = ei * items_per_thread + i;
                topk_weights[dst] = scores[i] * scale;
            }
            asm("shl.b32 %0, %1, 1;\n" : "=r"(bit) : "r"(bit));
        }
    }
}

void invokeMoeA2AGate_NoAuxTC(float*       topk_weights,
                              int*         topk_indices,
                              const float* logits,
                              const float* correction_bias,
                              int          tokens,
                              int          experts,
                              int          experts_per_token,
                              bool         norm_topk,
                              float        routed_scale,
                              bool         use_sigmoid,
                              cudaStream_t stream)
{
    TM_CHECK_GE(tokens, 0);

    if (tokens == 0) {
        return;
    }

    auto invoke = [&](auto expert_num, auto top_k, auto items_per_thread, auto access_size) {
        constexpr int threads           = 256;
        constexpr int threads_per_token = expert_num.value / items_per_thread.value;
        constexpr int tokens_per_cta    = threads / threads_per_token;
        const int     blocks            = (tokens + tokens_per_cta - 1) / tokens_per_cta;

        MoeA2AGateNoAuxTCKernel<expert_num.value, top_k.value, items_per_thread.value, threads, access_size.value>
            <<<blocks, threads, 0, stream>>>(
                topk_weights, topk_indices, logits, correction_bias, tokens, norm_topk, routed_scale, use_sigmoid);
        TM_CUDA_CHECK(cudaGetLastError());
        return true;
    };

    bool success = false;
    if (experts == 64 && experts_per_token == 4) {
        success = invoke(Int<64>, Int<4>, Int<16>, Int<4>);
    }
    else if (experts == 160 && experts_per_token == 8) {
        success = invoke(Int<160>, Int<8>, Int<10>, Int<2>);
    }

    TM_CHECK(success) << "unsupported A2A gate config: expert_num=" << experts << ", top_k=" << experts_per_token;
}

__global__ void MoeA2AMappingKernel(int*       f2n,
                                    int*       f2E,
                                    int*       en2f,
                                    const int* recv_topk_idx,
                                    const int* actual_recv_tokens,
                                    const int* offsets,
                                    int*       expert_counters,
                                    int        recv_capacity,
                                    int        num_topk,
                                    int        num_local_experts)
{
    extern __shared__ int block_counts[];

    for (int expert_idx = threadIdx.x; expert_idx < num_local_experts; expert_idx += blockDim.x) {
        block_counts[expert_idx] = 0;
    }
    __syncthreads();

    const int token_idx       = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_recv_tokens = __ldg(actual_recv_tokens);

    // Use en2f as temporary storage for each assignment's rank within this CTA
    // and expert. This avoids K-sized per-thread arrays and their register cost.
    for (int topk_idx = 0; topk_idx < num_topk; ++topk_idx) {
        if (token_idx >= recv_capacity) {
            continue;
        }

        const int reverse_idx = topk_idx * recv_capacity + token_idx;
        en2f[reverse_idx]     = -1;

        if (token_idx >= num_recv_tokens) {
            continue;
        }

        const int expert_idx = __ldg(recv_topk_idx + token_idx * num_topk + topk_idx);
        if ((unsigned)expert_idx >= (unsigned)num_local_experts) {
            continue;
        }
        en2f[reverse_idx] = atomicAdd(block_counts + expert_idx, 1);
    }

    __syncthreads();

    // Reserve one contiguous output range per non-empty CTA/expert pair, then
    // reuse block_counts to store the corresponding expert-major base.
    for (int expert_idx = threadIdx.x; expert_idx < num_local_experts; expert_idx += blockDim.x) {
        const int count = block_counts[expert_idx];
        if (count > 0) {
            const int old_count = atomicSub(expert_counters + expert_idx, count);
            DEVICE_ASSERT(old_count >= count);
            block_counts[expert_idx] = __ldg(offsets + expert_idx + 1) - old_count;
        }
    }

    __syncthreads();

    if (token_idx >= num_recv_tokens || token_idx >= recv_capacity) {
        return;
    }

    // Convert the CTA-local ranks into expert-major flat indices.
    for (int topk_idx = 0; topk_idx < num_topk; ++topk_idx) {
        const int reverse_idx = topk_idx * recv_capacity + token_idx;
        const int local_rank  = en2f[reverse_idx];
        if (local_rank < 0) {
            continue;
        }

        const int expert_idx = __ldg(recv_topk_idx + token_idx * num_topk + topk_idx);
        const int flat_idx   = block_counts[expert_idx] + local_rank;
        f2n[flat_idx]        = token_idx;
        f2E[flat_idx]        = expert_idx;
        en2f[reverse_idx]    = flat_idx;
    }
}

void invokeMoeA2AMapping(int*         f2n,
                         int*         f2E,
                         int*         en2f,
                         const int*   recv_topk_idx,
                         const int*   actual_recv_tokens,
                         const int*   offsets,
                         int*         expert_counters,
                         int          recv_capacity,
                         int          num_topk,
                         int          num_local_experts,
                         cudaStream_t stream)
{
    if (recv_capacity == 0) {
        return;
    }

    constexpr int threads      = 256;
    const int     blocks       = (recv_capacity - 1) / threads + 1;
    const int     shared_bytes = num_local_experts * sizeof(int);
    MoeA2AMappingKernel<<<blocks, threads, shared_bytes, stream>>>(f2n,
                                                                   f2E,
                                                                   en2f,
                                                                   recv_topk_idx,
                                                                   actual_recv_tokens,
                                                                   offsets,
                                                                   expert_counters,
                                                                   recv_capacity,
                                                                   num_topk,
                                                                   num_local_experts);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T, int vec_size>
__global__ void MoeA2ASharedCombineKernel(T*           output,  //
                                          const T*     routed,
                                          const float* shared_scales,
                                          int          hidden_dim,
                                          float        shared_scale)
{
    const int token_idx = blockIdx.x;

    output += (int64_t)token_idx * hidden_dim;
    routed += (int64_t)token_idx * hidden_dim;

    if (shared_scales) {
        shared_scale *= fdividef(1.f, 1.f + expf(-__ldg(shared_scales + token_idx)));
    }

    using Vec = Array<T, vec_size>;

    for (int i = threadIdx.x * vec_size; i < hidden_dim; i += blockDim.x * vec_size) {
        Vec routed_vec;
        Load(routed_vec, routed + i);
        auto result = cast<float>(routed_vec);

        if (shared_scale != 0.f) {
            Vec shared_vec;
            Load(shared_vec, output + i);
            using namespace ops;
            result = result + cast<float>(shared_vec) * shared_scale;
        }
        Store(output + i, cast<T>(result));
    }
}

void invokeMoeA2ASharedCombine(core::Tensor&       output,
                               const core::Tensor& routed,
                               const float*        shared_scales,
                               float               shared_scale,
                               cudaStream_t        stream)
{
    const int tokens = output.shape(0);
    if (tokens == 0) {
        return;
    }

    const int hidden_dim = output.shape(1);
    auto      invoke     = [&](auto t) {
        using T                 = decltype(t);
        constexpr int vec_size  = 16 / sizeof(T);
        constexpr int block_dim = 256;
        MoeA2ASharedCombineKernel<T, vec_size><<<tokens, block_dim, 0, stream>>>(
            output.data<T>(), routed.data<T>(), shared_scales, hidden_dim, shared_scale);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    TM_DISPATCH_PRIMARY_DTYPES(output.dtype(), invoke);
}

}  // namespace turbomind
