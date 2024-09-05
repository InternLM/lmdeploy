/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/turbomind/utils/logger.h"
#include <curand_kernel.h>
namespace turbomind {

template<typename T>
void invokeBatchTopKSampling(void*          workspace,
                             size_t&        workspace_size,
                             const T*       log_probs,
                             int*           ids,
                             int*           sequence_length,
                             bool*          finished,
                             float*         cum_log_probs,
                             float*         output_log_probs,
                             float*         sampled_logprobs,
                             uint32_t*      sampled_indexes,
                             uint32_t*      sampled_nums,
                             curandState_t* curandstate,
                             const int      max_top_k,
                             const int*     top_ks,
                             const float    top_p,
                             const float*   top_ps,
                             const int      vocab_size_padded,
                             const int*     end_ids,
                             cudaStream_t   stream,
                             const int      batch_size,
                             const bool*    skip_decode);

void invokeCurandInitialize(curandState_t*     state,
                            const size_t       batch_size,
                            unsigned long long random_seed,
                            cudaStream_t       stream);

void invokeCurandBatchInitialize(curandState_t*            states,
                                 const size_t              batch_size,
                                 const unsigned long long* random_seeds,
                                 cudaStream_t              stream);

struct TopKSortFilterParams {
    void*  workspace;
    size_t workspace_size;
    void*  logits;
    void*  sorted_logits;
    int*   sorted_indices;
    int*   kept;
    int*   top_ks;
    int    max_top_k;
    int    batch_size;
    int    vocab_size;
    int    vocab_size_padded;
};

template<typename T>
void invokeTopKSortFilter(TopKSortFilterParams& params, cudaStream_t stream);

}  // namespace turbomind
