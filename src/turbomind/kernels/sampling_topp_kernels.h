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
#pragma once

#include <curand_kernel.h>

namespace turbomind {

void invokeTopPSortInitialize(const int    vocab_size_padded,
                              const int    vocab_size,
                              const size_t batch_size,
                              const int*   top_ks,
                              int*         topp_id_val_buf,
                              int*         begin_offet_buf,
                              int*         end_offset_buf,
                              cudaStream_t stream);

template<typename T>
void invokeSoftmax(T*           logits,
                   const int    vocab_size_padded,
                   const int    vocab_size,
                   const int    batch_size,
                   const int*   kept,
                   cudaStream_t stream);

struct BlockPrefixCallbackOp {
    // Running prefix
    float running_total;
    // Constructor
    __device__ BlockPrefixCallbackOp(float running_total): running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ float operator()(float block_aggregate)
    {
        float old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

struct TopPSortParams {
    void*  workspace;
    size_t workspace_size;
    void*  logits;
    void*  sorted_logits;
    int*   sorted_indices;
    int*   kept;
    int*   top_ks;
    float* top_ps;
    int    batch_size;
    int    vocab_size;
    int    vocab_size_padded;
};

template<typename T>
void invokeTopPSort(TopPSortParams& params, cudaStream_t stream);

struct TopPMinPFilterParams {
    void*  sorted_logits;
    int*   sorted_indices;
    int*   kept;
    float* top_ps;
    float* min_ps;
    int    batch_size;
    int    vocab_size;
    int    vocab_size_padded;
};

template<typename T>
void invokeTopPMinPFilter(TopPMinPFilterParams& params, cudaStream_t stream);

}  // namespace turbomind
