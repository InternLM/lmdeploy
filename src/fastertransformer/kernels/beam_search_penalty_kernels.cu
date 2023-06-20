/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>

#include "src/fastertransformer/kernels/beam_search_penalty_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"

namespace fastertransformer {

template<typename T>
__global__ void add_bias_temperature(T*          logits,
                                     const T*    bias,
                                     const int   batch_size,
                                     const int   beam_width,
                                     const int   vocab_size,
                                     const int   vocab_size_padded,
                                     const float temperature)
{
    int tid  = threadIdx.x;
    int bid  = blockIdx.x;
    int bbid = blockIdx.y;

    logits += bbid * vocab_size_padded;

    const T MASK_VAL = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
    const T inv_temp = static_cast<T>(1.0f / (temperature + 1e-6f));
    for (int i = tid + bid * blockDim.x; i < vocab_size_padded; i += blockDim.x * gridDim.x) {
        if (i < vocab_size) {
            T bias_val = bias == nullptr ? (T)(0.0f) : bias[i];
            logits[i]  = (logits[i] + bias_val) * inv_temp;
        }
        else {
            logits[i] = MASK_VAL;
        }
    }
}

template<>
__global__ void add_bias_temperature(half2*       logits,
                                     const half2* bias,
                                     const int    batch_size,
                                     const int    beam_width,
                                     const int    vocab_size,
                                     const int    vocab_size_padded,
                                     const float  temperature)
{
    assert(vocab_size % 2 == 0);
    assert(vocab_size_padded % 2 == 0);

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int bbid = blockIdx.y;

    const half2 mask_val = __float2half2_rn(-HALF_FLT_MAX);
    const half2 inv_temp = __float2half2_rn(1.0f / (temperature + 1e-6f));

    const int half_vocab_size        = vocab_size / 2;
    const int half_vocab_size_padded = vocab_size_padded / 2;

    logits += bbid * half_vocab_size_padded;
    for (int index = tid + bid * blockDim.x; index < half_vocab_size_padded; index += blockDim.x * gridDim.x) {
        int   vocab_idx = index % half_vocab_size_padded;
        half2 logit     = vocab_idx < half_vocab_size ? __ldg(&logits[index]) : mask_val;
        if (vocab_idx < half_vocab_size) {
            if (bias != nullptr) {
                logit = __hadd2(logit, bias[vocab_idx]);
            }
            logit = __hmul2(logit, inv_temp);
        }
        logits[index] = logit;
    }
}

template<typename T, bool IS_ADDITIVE>
__global__ void apply_repetition_penalty(T*          logits,
                                         const int   batch_size,
                                         const int   beam_width,
                                         const int   vocab_size,
                                         const int   vocab_size_padded,
                                         const int   step,
                                         const int*  current_ids,
                                         const int*  previous_ids,
                                         const int*  parent_ids,
                                         const int*  input_lengths,
                                         const int   max_input_length,
                                         const float repetition_penalty)
{
    assert(step > 0);

    const int tid      = threadIdx.x;
    const int bbid     = blockIdx.x;
    const int batch_id = bbid / beam_width;
    const int bbsize   = batch_size * beam_width;

    logits += bbid * vocab_size_padded;
    extern __shared__ char sbuf[];
    T*                     penalty_logits = reinterpret_cast<T*>(sbuf);
    // prevent misaligment when sizeof(T) = 2
    int*      penalty_indices = reinterpret_cast<int*>(sbuf + (sizeof(T) * step + 31) / 32 * 32);
    const int input_length    = (input_lengths != nullptr) ? input_lengths[bbid] : max_input_length;
    if (tid == 0) {
        T   repet_penalty         = static_cast<T>(repetition_penalty);
        int prev_id               = current_ids[bbid];
        T   prev_logit            = logits[prev_id];
        penalty_indices[step - 1] = prev_id;

        if (IS_ADDITIVE) {
            penalty_logits[step - 1] = prev_logit - repet_penalty;
        }
        else {
            penalty_logits[step - 1] = prev_logit > T(0) ? prev_logit / repet_penalty : prev_logit * repet_penalty;
        }
        if (step > 1) {
            int parent_beam = bbid % beam_width;
            for (int i = step - 2; i >= 0; --i) {
                // Skip the padded tokens.
                if (i >= input_length && i < max_input_length) {
                    continue;
                }
                parent_beam        = parent_ids[i * bbsize + batch_id * beam_width + parent_beam];
                prev_id            = previous_ids[i * bbsize + batch_id * beam_width + parent_beam];
                prev_logit         = logits[prev_id];
                penalty_indices[i] = prev_id;
                if (IS_ADDITIVE) {
                    penalty_logits[i] = prev_logit - repet_penalty;
                }
                else {
                    penalty_logits[i] = prev_logit > T(0) ? prev_logit / repet_penalty : prev_logit * repet_penalty;
                }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < step; i += blockDim.x) {
        if (i >= input_length && i < max_input_length) {
            continue;
        }
        logits[penalty_indices[i]] = penalty_logits[i];
    }
}

template<typename T>
__global__ void apply_min_length_penalty(T*         logits,
                                         const int  min_length,
                                         const int* end_ids,
                                         const int* sequence_lengths,
                                         const int  max_input_length,
                                         const int  beam_width,
                                         const int  vocab_size_padded)
{
    int bbid = threadIdx.x + blockIdx.x * blockDim.x;  // batch-beam index
    int bid  = bbid / beam_width;                      // batch index
    // We need +1 because sequence_lengths = max_input_length + num_gen_tokens - 1,
    // which is equal to the length of k/v caches.
    if (sequence_lengths[bbid] + 1 - max_input_length < min_length) {
        T mask_val                                      = (std::is_same<T, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
        logits[bbid * vocab_size_padded + end_ids[bid]] = mask_val;
    }
}

template<typename T>
void invokeAddBiasApplyPenalties(int                         step,
                                 T*                          logits,
                                 const int*                  current_ids,
                                 const int*                  previous_ids,
                                 const int*                  parent_ids,
                                 const int*                  input_lengths,
                                 const int*                  sequence_lengths,
                                 const T*                    bias,
                                 const int                   ite,
                                 const int                   max_input_length,
                                 const int                   local_batch_size,
                                 const int                   batch_size,
                                 const int                   beam_width,
                                 const int                   vocab_size,
                                 const int                   vocab_size_padded,
                                 const int*                  end_ids,
                                 const float                 temperature,
                                 const float                 repetition_penalty,
                                 const RepetitionPenaltyType repetition_penalty_type,
                                 const int                   min_length,
                                 cudaStream_t                stream)
{
    if (bias != nullptr || temperature != 1.0f || vocab_size != vocab_size_padded) {
        dim3 block(512);
        if (std::is_same<T, half>::value && vocab_size % 2 == 0 && vocab_size_padded % 2 == 0) {
            dim3 grid((vocab_size_padded / 2 + block.x - 1) / block.x, beam_width * local_batch_size);
            add_bias_temperature<<<grid, block, 0, stream>>>(reinterpret_cast<half2*>(logits),
                                                             reinterpret_cast<const half2*>(bias),
                                                             batch_size,
                                                             beam_width,
                                                             vocab_size,
                                                             vocab_size_padded,
                                                             temperature);
        }
        else {
            dim3 grid((vocab_size_padded + block.x - 1) / block.x, beam_width * local_batch_size);
            add_bias_temperature<<<grid, block, 0, stream>>>(
                logits, bias, batch_size, beam_width, vocab_size, vocab_size_padded, temperature);
        }
    }

    if (repetition_penalty_type != RepetitionPenaltyType::None && step > 0) {
        if (repetition_penalty != getDefaultPenaltyValue(repetition_penalty_type)) {
            size_t smem_size = (sizeof(T) * step + 31) / 32 * 32 + sizeof(int) * step;
            dim3   block(256);
            dim3   grid(beam_width * local_batch_size);
            if (repetition_penalty_type == RepetitionPenaltyType::Multiplicative) {
                apply_repetition_penalty<T, false>
                    <<<grid, block, smem_size, stream>>>(logits,
                                                         batch_size,
                                                         beam_width,
                                                         vocab_size,
                                                         vocab_size_padded,
                                                         step,
                                                         current_ids,
                                                         previous_ids,
                                                         // TODO(jaedeokk):
                                                         //   Remove (+ite ...) by getting parent_ids with offset
                                                         //   and then remove 'ite' argument from the function.
                                                         parent_ids + ite * beam_width * local_batch_size,
                                                         input_lengths,
                                                         max_input_length,
                                                         repetition_penalty);
            }
            else if (repetition_penalty_type == RepetitionPenaltyType::Additive) {
                apply_repetition_penalty<T, true>
                    <<<grid, block, smem_size, stream>>>(logits,
                                                         batch_size,
                                                         beam_width,
                                                         vocab_size,
                                                         vocab_size_padded,
                                                         step,
                                                         current_ids,
                                                         previous_ids,
                                                         parent_ids + ite * beam_width * local_batch_size,
                                                         input_lengths,
                                                         max_input_length,
                                                         repetition_penalty);
            }
        }
    }

    if (step - max_input_length < min_length) {
        FT_CHECK_WITH_INFO(sequence_lengths != nullptr, "Need sequence_lengths to apply min length penlaty");
        FT_CHECK_WITH_INFO(end_ids != nullptr, "Need end_id to apply min length penlaty");

        const int block_size = min(local_batch_size * beam_width, 1024);
        const int grid_size  = (local_batch_size * beam_width + block_size - 1) / block_size;
        apply_min_length_penalty<<<grid_size, block_size, 0, stream>>>(
            logits, min_length, end_ids, sequence_lengths, max_input_length, beam_width, vocab_size_padded);
    }
}

template void invokeAddBiasApplyPenalties(int                         step,
                                          float*                      logits,
                                          const int*                  current_ids,
                                          const int*                  previous_ids,
                                          const int*                  parent_ids,
                                          const int*                  input_lengths,
                                          const int*                  sequence_lengths,
                                          const float*                bias,
                                          const int                   ite,
                                          const int                   max_input_length,
                                          const int                   local_batch_size,
                                          const int                   batch_size,
                                          const int                   beam_width,
                                          const int                   vocab_size,
                                          const int                   vocab_size_padded,
                                          const int*                  end_ids,
                                          const float                 temperature,
                                          const float                 repetition_penalty,
                                          const RepetitionPenaltyType repetition_penalty_type,
                                          const int                   min_length,
                                          cudaStream_t                stream);

template void invokeAddBiasApplyPenalties(int                         step,
                                          half*                       logits,
                                          const int*                  current_ids,
                                          const int*                  previous_ids,
                                          const int*                  parent_ids,
                                          const int*                  input_lengths,
                                          const int*                  sequence_lengths,
                                          const half*                 bias,
                                          const int                   ite,
                                          const int                   max_input_length,
                                          const int                   local_batch_size,
                                          const int                   batch_size,
                                          const int                   beam_width,
                                          const int                   vocab_size,
                                          const int                   vocab_size_padded,
                                          const int*                  end_ids,
                                          const float                 temperature,
                                          const float                 repetition_penalty,
                                          const RepetitionPenaltyType repetition_penalty_type,
                                          const int                   min_length,
                                          cudaStream_t                stream);

}  // namespace fastertransformer
