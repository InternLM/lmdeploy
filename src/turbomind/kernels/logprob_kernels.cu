/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/logprob_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
__global__ void log_probs_kernel(float*       log_probs,
                                 const T*     logits,
                                 const int*   ids,
                                 const int*   lengths,
                                 const size_t max_input_length,
                                 const size_t batch_size,
                                 const size_t vocab_size,
                                 const size_t vocab_size_padded,
                                 bool         batch_first)
{
    // Calculate the log probability from logits.
    //   log_probs[t, :] = log(softmax(logits))[ids[t + 1, :]]
    //
    // log_probs: [max_length - 1, batch_size] or [batch_size, max_length -1],
    //     log probabilities of each token.
    // logits: [max_length, batch_size, vocab_size_padded] or [batch_size, max_length, vocab_size_padded]
    // lengths: [batch_size], sequence lengths
    // ids: [max_length, batch_size], token ids.
    // batch_size: [1], batch_size. in case of beam > 1, batch x beam.
    // vocab_size: [1], vocab_size,
    // vocab_size: [1], vocab_size_padded, padded vocab size.

    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    int tidx = threadIdx.x;                            // vocab dim
    int bidx = batch_first ? blockIdx.x : blockIdx.y;  // batch dim
    int step = batch_first ? blockIdx.y : blockIdx.x;  // step dim

    __shared__ float s_max_logit;

    if (bidx < batch_size && step < lengths[bidx] - 1) {
        // reposition logits to data for the current batch.
        int step_offset  = batch_first ? step * vocab_size_padded : step * batch_size * vocab_size_padded;
        int batch_offset = batch_first ? bidx * max_input_length * vocab_size_padded : bidx * vocab_size_padded;
        logits += step_offset + batch_offset;

        // Find max(logits).
        float local_max = -MAX_T_VAL;
        float val       = -MAX_T_VAL;
        for (int i = tidx; i < vocab_size; i += blockDim.x) {
            val       = static_cast<float>(logits[i]);
            local_max = fmax(local_max, val);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (tidx == 0) {
            s_max_logit = max_val;
        }
        __syncthreads();

        // Calculate the denominator: sum_i exp(logits[i])
        float local_sum_exp = 0.0f;
        for (int i = tidx; i < vocab_size; i += blockDim.x) {
            val = __expf(static_cast<float>(logits[i]) - s_max_logit);
            local_sum_exp += val;
        }

        float sum_exp = blockDim.x <= 32 ? warpReduceSum(local_sum_exp) : blockReduceSum<float>(local_sum_exp);
        if (tidx == 0) {
            int idx = batch_first ? step + bidx * (max_input_length - 1) : step * batch_size + bidx;
            // log_probs[step, ...] is the log probability of a token at step t + 1.
            int token_idx  = batch_first ? step + 1 + bidx * max_input_length : (step + 1) * batch_size + bidx;
            log_probs[idx] = static_cast<float>(logits[ids[token_idx]]) - s_max_logit - __logf(sum_exp + 1e-9f);
        }
    }
}

__global__ void accumulate_log_probs(float*       cum_log_probs,
                                     const float* log_probs,
                                     const int*   lengths,
                                     const size_t max_input_length,
                                     const size_t batch_size,
                                     const bool   batch_first)
{
    // Accumulate the log probability along with the sequence dimension.
    //   cum_log_probs[j] = sum_i log(softmax(logits))[ids[i,j]]
    //
    // cum_log_probs: [batch_size], cumulative log probability
    // log_probs: [max_length - 1, batch_size] or [batch_size, max_length - 1],
    //   log probability of each token
    // lengths: [batch_size], sequence lengths
    // batch_size: [1], batch_size. in case of beam > 1, batch x beam.

    int bidx = blockIdx.x;   // batch dim
    int tidx = threadIdx.x;  // step dim

    if (bidx < batch_size) {
        int length = lengths[bidx];
        // reposition logits to data for the current batch.
        log_probs += batch_first ? bidx * (max_input_length - 1) : bidx;
        int   stride      = batch_first ? 1 : batch_size;  // stride along with seq dim.
        float local_accum = 0.0f;
        for (int step = tidx; step < length - 1; step += blockDim.x) {
            local_accum += static_cast<float>(log_probs[step * stride]);
        }
        float accum = blockDim.x <= 32 ? warpReduceSum(local_accum) : blockReduceSum<float>(local_accum);
        if (tidx == 0) {
            cum_log_probs[bidx] = accum;
        }
    }
}

template<typename T>
void invokeLogProbFromLogits(float*       cum_log_probs,
                             const T*     logits,
                             const int*   input_ids,
                             const int*   input_lengths,
                             const size_t max_input_length,
                             const size_t batch_size,
                             const size_t vocab_size,
                             const size_t vocab_size_padded,
                             void*        workspace,
                             const size_t workspace_size,
                             cudaStream_t stream,
                             const bool   batch_first)
{
    // A batched version of log prob computation.
    //
    // cum_log_probs: [batch_size]
    // logits: [max_input_length, batch_size, vocab_size] or [batch_size, max_input_length, vocab_size]
    // input_ids: [max_input_length, batch_size] or [max_input_length, batch_size]
    // input_lengths: [batch_size]
    // workspace: workspace buffer of size at least sizeof(float) * max_input_length * batch_size.

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // block_size should be multiple of 32 to use warpReduceMax.
    const int block_size = vocab_size < 1024 ? (vocab_size + 31) / 32 * 32 : 1024;
    assert(block_size % 32 == 0);
    assert(workspace != nullptr && workspace_size >= sizeof(float) * max_input_length * batch_size);
    assert(vocab_size <= vocab_size_padded);

    float* log_probs = reinterpret_cast<float*>(workspace);
    int    gx        = batch_first ? batch_size : max_input_length - 1;
    int    gy        = batch_first ? max_input_length - 1 : batch_size;
    dim3   grid(gx, gy);
    log_probs_kernel<T><<<grid, block_size, 0, stream>>>(log_probs,
                                                         logits,
                                                         input_ids,
                                                         input_lengths,
                                                         max_input_length,
                                                         batch_size,
                                                         vocab_size,
                                                         vocab_size_padded,
                                                         batch_first);
    accumulate_log_probs<<<batch_size, block_size, 0, stream>>>(
        cum_log_probs, log_probs, input_lengths, max_input_length, batch_size, batch_first);
}

template void invokeLogProbFromLogits(float*       cum_log_probs,
                                      const float* logits,
                                      const int*   input_ids,
                                      const int*   input_lengths,
                                      const size_t max_input_length,
                                      const size_t batch_size,
                                      const size_t vocab_size,
                                      const size_t vocab_size_padded,
                                      void*        workspace,
                                      const size_t workspace_size,
                                      cudaStream_t stream,
                                      const bool   batch_first);

template void invokeLogProbFromLogits(float*       cum_log_probs,
                                      const half*  logits,
                                      const int*   input_ids,
                                      const int*   input_lengths,
                                      const size_t max_input_length,
                                      const size_t batch_size,
                                      const size_t vocab_size,
                                      const size_t vocab_size_padded,
                                      void*        workspace,
                                      const size_t workspace_size,
                                      cudaStream_t stream,
                                      const bool   batch_first);
}  // end of namespace fastertransformer
