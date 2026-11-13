// Copyright (c) OpenMMLab. All rights reserved.

#include <climits>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/speculative_sequence_kernels.h"

namespace turbomind {

__global__ void build_target_inputs(int*              target_input_ids,
                                    int*              target_key_lengths,
                                    const int* const* request_token_ids_ptrs,
                                    const int*        target_q_offsets,
                                    const int*        sequence_length,
                                    const bool*       target_ids_from_row,
                                    const bool*       finished,
                                    int               request_count)
{
    const int b = blockIdx.x;
    if (b >= request_count) {
        return;
    }

    const int  q_begin  = target_q_offsets[b];
    const int  q_len    = target_q_offsets[b + 1] - q_begin;
    const int  S        = sequence_length[b];
    const bool from_row = target_ids_from_row[b];

    if (threadIdx.x == 0) {
        target_key_lengths[b] = S + (from_row ? q_len - 1 : 0);
    }

    if (!from_row) {
        return;
    }

    if (finished[b]) {
        for (int j = threadIdx.x; j < q_len; j += blockDim.x) {
            target_input_ids[q_begin + j] = 0;
        }
        return;
    }

    const int token_begin = S - 1;

    for (int j = threadIdx.x; j < q_len; j += blockDim.x) {
        target_input_ids[q_begin + j] = request_token_ids_ptrs[b][token_begin + j];
    }
}

void invokeBuildTargetInputs(int*              target_input_ids,
                             int*              target_key_lengths,
                             const int* const* request_token_ids_ptrs,
                             const int*        target_q_offsets,
                             const int*        sequence_length,
                             const bool*       target_ids_from_row,
                             const bool*       finished,
                             int               request_count,
                             cudaStream_t      stream)
{
    if (request_count == 0) {
        return;
    }

    build_target_inputs<<<request_count, 256, 0, stream>>>(target_input_ids,
                                                           target_key_lengths,
                                                           request_token_ids_ptrs,
                                                           target_q_offsets,
                                                           sequence_length,
                                                           target_ids_from_row,
                                                           finished,
                                                           request_count);
}

template<int BLOCK_SIZE>
__global__ void BuildDraftExtensionKeyOffsetsKernel(int*       k_offsets,
                                                    const int* q_offsets,
                                                    const int* entry_sequence_length,
                                                    const int* accept_len,
                                                    int        batch_size,
                                                    int        extension_index)
{
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    const int end = ((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    int prefix = 0;

    for (int b = threadIdx.x; b < end; b += BLOCK_SIZE) {
        if (b >= BLOCK_SIZE) {
            __syncthreads();
        }

        int key_len = 0;

        if (b < batch_size) {
            const int q_width = q_offsets[b + 1] - q_offsets[b];

            if (q_width == 1) {
                key_len = entry_sequence_length[b] + accept_len[b] + extension_index;
            }
        }

        int tile_sum = 0;
        BlockScan{scan_storage}.ExclusiveSum(key_len, key_len, tile_sum);

        if (b < batch_size) {
            k_offsets[b] = prefix + key_len;
        }

        prefix += tile_sum;
    }

    if (threadIdx.x == 0) {
        k_offsets[batch_size] = prefix;
    }
}

void invokeBuildDraftExtensionKeyOffsets(int*         k_offsets,
                                         const int*   q_offsets,
                                         const int*   entry_sequence_length,
                                         const int*   accept_len,
                                         int          batch_size,
                                         int          extension_index,
                                         cudaStream_t stream)
{
    constexpr int block_size = 256;

    BuildDraftExtensionKeyOffsetsKernel<block_size><<<1, block_size, 0, stream>>>(
        k_offsets, q_offsets, entry_sequence_length, accept_len, batch_size, extension_index);
}

__global__ void InitializeTargetVerificationKernel(bool*             block_logits_active,
                                                   int*              effective_history,
                                                   int*              verification_draft_ids,
                                                   const int* const* request_token_ids_ptrs,
                                                   const int*        entry_sequence_length,
                                                   const bool*       finished_on_entry,
                                                   const bool*       speculative_row,
                                                   int*              accepted_draft_count,
                                                   const int*        request_to_generation_row_offsets,
                                                   int               request_count,
                                                   int               generation_count,
                                                   int               position_count)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= request_count) {
        return;
    }

    if (accepted_draft_count) {
        accepted_draft_count[b] = speculative_row[b] && !finished_on_entry[b] ? 0 : -1;
    }

    const int g_begin = request_to_generation_row_offsets[b];
    const int g_end   = request_to_generation_row_offsets[b + 1];

    if (g_end == g_begin) {
        return;
    }

    const int  g      = g_begin;
    const int  S      = entry_sequence_length[b];
    const bool active = !finished_on_entry[b];
    const bool verify = active && speculative_row[b];

    for (int position = 0; position < position_count; ++position) {
        const int flat             = position * generation_count + g;
        block_logits_active[flat]  = active && (position == 0 || verify);
        effective_history[flat]    = S + position;

        if (position + 1 < position_count) {
            verification_draft_ids[position * generation_count + g] =
                verify ? request_token_ids_ptrs[b][S + position] : 0;
        }
    }
}

void invokeInitializeTargetVerification(bool*             block_logits_active,
                                        int*              effective_history,
                                        int*              verification_draft_ids,
                                        const int* const* request_token_ids_ptrs,
                                        const int*        entry_sequence_length,
                                        const bool*       finished_on_entry,
                                        const bool*       speculative_row,
                                        int*              accepted_draft_count,
                                        const int*        request_to_generation_row_offsets,
                                        int               request_count,
                                        int               generation_count,
                                        int               position_count,
                                        cudaStream_t      stream)
{
    if (request_count == 0) {
        return;
    }

    static_cast<void>(generation_count);

    constexpr int block_size = 128;
    const int     grid_size  = cdiv(request_count, block_size);

    InitializeTargetVerificationKernel<<<grid_size, block_size, 0, stream>>>(block_logits_active,
                                                                             effective_history,
                                                                             verification_draft_ids,
                                                                             request_token_ids_ptrs,
                                                                             entry_sequence_length,
                                                                             finished_on_entry,
                                                                             speculative_row,
                                                                             accepted_draft_count,
                                                                             request_to_generation_row_offsets,
                                                                             request_count,
                                                                             generation_count,
                                                                             position_count);
}

__global__ void build_draft_refresh_inputs(int*              draft_input_ids,
                                           int*              selected_token_pos,
                                           bool*             candidate_active,
                                           const int* const* token_ids_ptrs,
                                           const int*        q_offsets,
                                           const int*        k_offsets,
                                           const int*        extension_q_offsets,
                                           const int*        accept_len,
                                           const bool*       limit_to_accept_len,
                                           const bool*       finished,
                                           int               batch_size)
{
    const int b = blockIdx.x;

    if (b >= batch_size) {
        return;
    }

    const int q_begin = q_offsets[b];
    const int q_end   = q_offsets[b + 1];
    const int q_len   = q_end - q_begin;

    if (threadIdx.x == 0) {
        const int candidate_begin = extension_q_offsets[b];
        const int candidate_end   = extension_q_offsets[b + 1];

        if (candidate_end != candidate_begin) {
            const int candidate = candidate_begin;

            bool active       = false;
            int  selected_pos = 0;

            if (!finished[b]) {
                if (limit_to_accept_len[b]) {
                    const int committed = accept_len[b];

                    if (committed > 0) {
                        active       = true;
                        selected_pos = q_begin + committed - 1;
                    }
                }
                else if (q_len > 0) {
                    active       = true;
                    selected_pos = q_end - 1;
                }
            }

            selected_token_pos[candidate] = selected_pos;
            candidate_active[candidate]   = active;
        }
    }

    if (q_len == 0) {
        return;
    }

    const int k_len = k_offsets[b + 1] - k_offsets[b];

    const int token_begin = k_len - q_len + 1;

    int valid_len = q_len;

    if (limit_to_accept_len[b]) {
        valid_len = accept_len[b];
    }

    for (int j = threadIdx.x; j < q_len; j += blockDim.x) {
        int token = 0;

        if (j < valid_len) {
            token = token_ids_ptrs[b][token_begin + j];
        }

        draft_input_ids[q_begin + j] = token;
    }
}

void invokeBuildDraftRefreshInputs(int*              draft_input_ids,
                                   int*              selected_token_pos,
                                   bool*             candidate_active,
                                   const int* const* token_ids_ptrs,
                                   const int*        refresh_q_offsets,
                                   const int*        refresh_k_offsets,
                                   const int*        extension_q_offsets,
                                   const int*        accept_len,
                                   const bool*       limit_to_accept_len,
                                   const bool*       finished,
                                   int               draft_input_count,
                                   int               batch_size,
                                   int               candidate_count,
                                   cudaStream_t      stream)
{
    if (batch_size == 0) {
        return;
    }

    if (draft_input_count == 0) {
        return;
    }

    constexpr int block_size = 256;

    build_draft_refresh_inputs<<<batch_size, block_size, 0, stream>>>(draft_input_ids,
                                                                      selected_token_pos,
                                                                      candidate_active,
                                                                      token_ids_ptrs,
                                                                      refresh_q_offsets,
                                                                      refresh_k_offsets,
                                                                      extension_q_offsets,
                                                                      accept_len,
                                                                      limit_to_accept_len,
                                                                      finished,
                                                                      batch_size);
}

struct DraftArgMax {
    float value;
    int   token_id;
};

struct DraftArgMaxOp {
    __device__ DraftArgMax operator()(const DraftArgMax& a, const DraftArgMax& b) const
    {
        if (b.value > a.value) {
            return b;
        }
        if (b.value == a.value && b.token_id < a.token_id) {
            return b;
        }
        return a;
    }
};

template<class T, int BLOCK_SIZE>
__global__ void DraftArgmaxAndStoreTokenKernel(const T*    logits,
                                               int         logits_stride,
                                               int         vocab_size,
                                               int*        proposal_ids,
                                               int* const* token_ids_ptrs,
                                               const int*  extension_q_offsets,
                                               const bool* candidate_active,
                                               const int*  entry_sequence_length,
                                               const int*  accept_len,
                                               int         batch_size,
                                               int         proposal_index)
{
    const int b = blockIdx.x;

    if (b >= batch_size) {
        return;
    }

    const int candidate_begin = extension_q_offsets[b];
    const int candidate_end   = extension_q_offsets[b + 1];

    if (candidate_begin == candidate_end) {
        return;
    }

    const int c = candidate_begin;

    if (!candidate_active[c]) {
        if (threadIdx.x == 0) {
            proposal_ids[c] = 0;
        }
        return;
    }

    const T* row = logits + static_cast<ssize_t>(c) * logits_stride;

    DraftArgMax local{-CUDART_INF_F, INT_MAX};

    for (int token_id = threadIdx.x; token_id < vocab_size; token_id += BLOCK_SIZE) {
        float value = static_cast<float>(row[token_id]);

        if (isnan(value)) {
            value = -CUDART_INF_F;
        }

        local = DraftArgMaxOp{}(local, DraftArgMax{value, token_id});
    }

    using BlockReduce = cub::BlockReduce<DraftArgMax, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage storage;

    const DraftArgMax best = BlockReduce(storage).Reduce(local, DraftArgMaxOp{});

    if (threadIdx.x == 0) {
        proposal_ids[c] = best.token_id;

        const int token_position = entry_sequence_length[b] + accept_len[b] + proposal_index;

        token_ids_ptrs[b][token_position] = best.token_id;
    }
}

void invokeDraftArgmaxAndStoreToken(const Tensor& logits,
                                    int*          proposal_ids,
                                    int* const*   token_ids_ptrs,
                                    const int*    extension_q_offsets,
                                    const bool*   candidate_active,
                                    const int*    entry_sequence_length,
                                    const int*    accept_len,
                                    int           batch_size,
                                    int           candidate_count,
                                    int           proposal_index,
                                    int           vocab_size,
                                    cudaStream_t  stream)
{
    if (batch_size == 0 || candidate_count == 0) {
        return;
    }

    constexpr int block_size    = 256;
    const int     logits_stride = static_cast<int>(logits.shape(1));

    auto dispatch = [&](auto t) {
        using T = decltype(t);
        DraftArgmaxAndStoreTokenKernel<T, block_size><<<batch_size, block_size, 0, stream>>>(logits.data<T>(),
                                                                                             logits_stride,
                                                                                             vocab_size,
                                                                                             proposal_ids,
                                                                                             token_ids_ptrs,
                                                                                             extension_q_offsets,
                                                                                             candidate_active,
                                                                                             entry_sequence_length,
                                                                                             accept_len,
                                                                                             batch_size,
                                                                                             proposal_index);
    };

    TM_DISPATCH_DTYPES(logits.dtype(), dispatch, half_t, bfloat16_t);
}

__global__ void
AdvanceSequenceByAcceptedSpan(int* sequence_length, const int* accept_len, int* accepted_draft_count, int batch_size)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size) {
        return;
    }

    sequence_length[b] += accept_len[b];

    if (accepted_draft_count && accepted_draft_count[b] >= 0) {
        accepted_draft_count[b] = min(accepted_draft_count[b], accept_len[b]);
    }
}

void invokeAdvanceSequenceByAcceptedSpan(
    int* sequence_length, const int* accept_len, int* accepted_draft_count, int batch_size, cudaStream_t stream)
{
    if (batch_size == 0) {
        return;
    }

    constexpr int threads = 256;
    const int     blocks  = (batch_size + threads - 1) / threads;

    AdvanceSequenceByAcceptedSpan<<<blocks, threads, 0, stream>>>(
        sequence_length, accept_len, accepted_draft_count, batch_size);
}

}  // namespace turbomind
