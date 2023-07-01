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

#pragma once

#include <assert.h>
#include <cuda_fp16.h>

#include <iostream>

#include "src/turbomind/utils/cuda_utils.h"

#define CUSTOM_AR_SIZE_THRESHOLD 50331648
#define MAX_ALL_REDUCE_BLOCKS 24
#define FLAG(a) ((uint32_t)((a) % 0x146))
#define RANKS_PER_NODE 8
#define WARP_SIZE 32
#define DEFAULT_BLOCK_SIZE 1024
#define DEFALUT_ALGO_AR_SIZE_THRESHOLD 393216

namespace fastertransformer {

#ifdef ENABLE_BF16
typedef struct bf168 {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
} bf168;
#endif

template<typename T>
struct AllReduceParams {
    size_t    elts_total;
    size_t    elts_per_rank;
    size_t    elts_per_block;
    size_t    rank_offset;
    size_t    rank, local_rank, node_id;
    uint32_t  barrier_flag;
    uint32_t* peer_barrier_ptrs[RANKS_PER_NODE];
    T*        peer_comm_buffer_ptrs[RANKS_PER_NODE];
    T*        local_output_buffer_ptr;
};

template<typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams<T>& param, cudaStream_t stream);

void kernelLaunchConfig(int& blocks_per_grid, int& threads_per_block, size_t elts, int kernel_algo);

}  // namespace fastertransformer
