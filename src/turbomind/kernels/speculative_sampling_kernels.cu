#include <cfloat>
#include <climits>
#include <cstddef>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11000)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include <curand_kernel.h>

#include "src/turbomind/kernels/sampling_device.cuh"
#include "src/turbomind/kernels/speculative_sampling_kernels.h"

namespace turbomind {
namespace {

constexpr int kBlockSize = 256;

struct ArgMax {
    float probability;
    int   index;
};

struct ArgMaxOp {
    __device__ ArgMax operator()(ArgMax lhs, ArgMax rhs) const
    {
        if (rhs.probability > lhs.probability || (rhs.probability == lhs.probability && rhs.index < lhs.index)) {
            return rhs;
        }
        return lhs;
    }
};

struct RecoveryStats {
    float mass;
    int   last_index;
};

struct RecoveryStatsOp {
    __device__ RecoveryStats operator()(RecoveryStats lhs, RecoveryStats rhs) const
    {
        return {lhs.mass + rhs.mass, max(lhs.last_index, rhs.last_index)};
    }
};

struct MaxFloatOp {
    __device__ float operator()(float lhs, float rhs) const
    {
        return max(lhs, rhs);
    }
};

struct MinIndexOp {
    __device__ int operator()(int lhs, int rhs) const
    {
        return min(lhs, rhs);
    }
};

template<int BlockSize>
struct VerifySharedStorage {
    typename cub::BlockReduce<ArgMax, BlockSize>::TempStorage        argmax;
    typename cub::BlockReduce<float, BlockSize>::TempStorage         float_reduce;
    typename cub::BlockReduce<RecoveryStats, BlockSize>::TempStorage recovery;
    typename cub::BlockScan<float, BlockSize>::TempStorage           recovery_scan;
    typename cub::BlockReduce<int, BlockSize>::TempStorage           index_reduce;
    ProcessedDistributionSampleStorage<BlockSize>                    sample;

    float probability_draft;
    float recovery_threshold;
    int   recovery_token;
    bool  recovery_found;
};

template<int BlockSize>
__device__ void VerifyOneProcessedDraft(const float*                    row,
                                        const int*                      token_row,
                                        int                             kept,
                                        int                             draft,
                                        bool                            greedy,
                                        curandState_t*                  random_state,
                                        int&                            selected,
                                        bool&                           accepted,
                                        VerifySharedStorage<BlockSize>& storage)
{
    const int thread_idx = threadIdx.x;

    if (greedy) {
        ArgMax local{-FLT_MAX, INT_MAX};
        for (int i = thread_idx; i < kept; i += BlockSize) {
            local = ArgMaxOp{}(local, ArgMax{row[i], i});
        }

        const ArgMax target = cub::BlockReduce<ArgMax, BlockSize>(storage.argmax).Reduce(local, ArgMaxOp{});
        if (thread_idx == 0) {
            selected = token_row[target.index];
            accepted = selected == draft;
        }
        __syncthreads();
        return;
    }

    float local_draft_probability = 0.f;
    for (int i = thread_idx; i < kept; i += BlockSize) {
        if (token_row[i] == draft) {
            local_draft_probability = max(local_draft_probability, row[i]);
        }
    }
    const float reduced_draft_probability =
        cub::BlockReduce<float, BlockSize>(storage.float_reduce).Reduce(local_draft_probability, MaxFloatOp{});
    if (thread_idx == 0) {
        storage.probability_draft = reduced_draft_probability;
    }
    __syncthreads();

    if (thread_idx == 0) {
        const float acceptance_uniform = curand_uniform(random_state);
        accepted = storage.probability_draft == 1.f
                   || (storage.probability_draft > 0.f && acceptance_uniform <= storage.probability_draft);
        if (accepted) {
            selected = draft;
        }
    }
    __syncthreads();
    if (accepted) {
        return;
    }

    RecoveryStats local_recovery{0.f, -1};
    for (int i = thread_idx; i < kept; i += BlockSize) {
        const float probability = row[i];
        if (token_row[i] != draft && probability > 0.f) {
            local_recovery.mass += probability;
            local_recovery.last_index = max(local_recovery.last_index, i);
        }
    }
    const RecoveryStats recovery =
        cub::BlockReduce<RecoveryStats, BlockSize>(storage.recovery).Reduce(local_recovery, RecoveryStatsOp{});
    if (thread_idx == 0) {
        storage.recovery_threshold = curand_uniform(random_state) * recovery.mass;
        storage.recovery_token     = token_row[recovery.last_index];
        storage.recovery_found     = false;
    }
    __syncthreads();

    BlockPrefixCallbackOp prefix_op{0.f};
    for (int base = 0; base < kept; base += BlockSize) {
        const int i           = base + thread_idx;
        float     probability = 0.f;
        if (i < kept && token_row[i] != draft && row[i] > 0.f) {
            probability = row[i];
        }

        float inclusive_mass;
        cub::BlockScan<float, BlockSize>(storage.recovery_scan).InclusiveSum(probability, inclusive_mass, prefix_op);
        __syncthreads();

        const int candidate =
            probability > 0.f && inclusive_mass > storage.recovery_threshold ? i : INT_MAX;
        const int first_crossing =
            cub::BlockReduce<int, BlockSize>(storage.index_reduce).Reduce(candidate, MinIndexOp{});
        if (thread_idx == 0 && first_crossing != INT_MAX) {
            storage.recovery_token = token_row[first_crossing];
            storage.recovery_found = true;
        }
        __syncthreads();
        if (storage.recovery_found) {
            break;
        }
    }

    if (thread_idx == 0) {
        selected = storage.recovery_token;
        accepted = false;
    }
    __syncthreads();
}

template<int BlockSize>
__device__ void SampleOneProcessedDistribution(const float*                    row,
                                               const int*                      token_row,
                                               int                             kept,
                                               curandState_t*                  random_state,
                                               int&                            selected,
                                               VerifySharedStorage<BlockSize>& storage)
{
    const int selected_index =
        SampleProcessedDistribution<float, BlockSize>(row, kept, random_state, storage.sample);
    if (threadIdx.x == 0) {
        selected = token_row[selected_index];
    }
    __syncthreads();
}

template<int BlockSize>
__global__ void VerifyTargetBlock(VerifyTargetBlockParams p)
{
    const int b       = blockIdx.x;
    const int g_begin = p.request_to_generation_offsets[b];
    const int g_end   = p.request_to_generation_offsets[b + 1];
    if (g_begin == g_end) {
        return;
    }

    const int g = g_begin;
    __shared__ VerifySharedStorage<BlockSize> storage;
    __shared__ int                            selected;
    __shared__ int                            span_len;
    __shared__ int                            accepted_drafts;
    __shared__ bool                           accepted;
    __shared__ bool                           continue_verification;

    curandState_t* random_state = p.random_states + p.random_state_indices[g];
    if (threadIdx.x == 0) {
        span_len              = 0;
        accepted_drafts       = 0;
        continue_verification = p.logits_active[g];
    }
    __syncthreads();

    if (!p.speculative_row[b]) {
        if (continue_verification) {
            const float* row = p.probabilities + g * p.probability_stride;
            const int*   ids = p.probability_token_ids + g * p.token_id_stride;
            SampleOneProcessedDistribution<BlockSize>(
                row, ids, p.kept_count[g], random_state, selected, storage);
            __syncthreads();

            if (threadIdx.x == 0) {
                p.request_token_ids_ptrs[b][p.entry_sequence_length[b]] = selected;
                p.selected_span_ids[b * p.selected_span_stride]         = selected;
                span_len                                                = 1;
            }
        }
    }
    else {
        for (int position = 0; position < p.position_count - 1; ++position) {
            const int flat = position * p.generation_count + g;
            if (continue_verification && p.logits_active[flat]) {
                const float* row   = p.probabilities + flat * p.probability_stride;
                const int*   ids   = p.probability_token_ids + flat * p.token_id_stride;
                const int    draft = p.verification_draft_ids[position * p.draft_row_stride + g];

                VerifyOneProcessedDraft<BlockSize>(row,
                                                   ids,
                                                   p.kept_count[flat],
                                                   draft,
                                                   p.greedy[flat],
                                                   random_state,
                                                   selected,
                                                   accepted,
                                                   storage);
                __syncthreads();

                if (threadIdx.x == 0) {
                    p.request_token_ids_ptrs[b][p.entry_sequence_length[b] + span_len] = selected;
                    p.selected_span_ids[b * p.selected_span_stride + span_len]         = selected;
                    ++span_len;
                    if (accepted) {
                        ++accepted_drafts;
                    }
                    else {
                        continue_verification = false;
                    }
                }
                __syncthreads();
            }
        }

        const int bonus_flat = (p.position_count - 1) * p.generation_count + g;
        if (continue_verification && p.logits_active[bonus_flat]) {
            const float* row = p.probabilities + bonus_flat * p.probability_stride;
            const int*   ids = p.probability_token_ids + bonus_flat * p.token_id_stride;
            SampleOneProcessedDistribution<BlockSize>(
                row, ids, p.kept_count[bonus_flat], random_state, selected, storage);
            __syncthreads();

            if (threadIdx.x == 0) {
                p.request_token_ids_ptrs[b][p.entry_sequence_length[b] + span_len] = selected;
                p.selected_span_ids[b * p.selected_span_stride + span_len]         = selected;
                ++span_len;
            }
        }
    }

    if (threadIdx.x == 0) {
        p.accept_len[b] = span_len;
        if (p.accepted_draft_count && p.speculative_row[b] && p.logits_active[g]) {
            p.accepted_draft_count[b] = accepted_drafts;
        }
    }
}

}  // namespace

void invokeVerifyTargetBlock(const VerifyTargetBlockParams& params, cudaStream_t stream)
{
    if (params.request_count == 0) {
        return;
    }
    VerifyTargetBlock<kBlockSize><<<params.request_count, kBlockSize, 0, stream>>>(params);
}

}  // namespace turbomind
