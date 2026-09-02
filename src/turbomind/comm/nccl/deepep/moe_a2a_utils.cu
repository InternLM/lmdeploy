#include "src/turbomind/comm/nccl/deepep/moe_a2a_utils.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <utility>

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
