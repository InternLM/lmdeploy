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
#include <vector>

#include "src/turbomind/layers/BaseLayer.h"
#include "src/turbomind/layers/attention_layers/BaseAttentionLayer.h"
#include "src/turbomind/layers/attention_layers_fp8/AttentionFP8Weight.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasFP8MMWrapper.h"
#include "src/turbomind/utils/memory_utils.h"

namespace fastertransformer {

// template<typename T>
// AttentionType getAttentionType(size_t size_per_head, const int sm, const bool remove_padding, const int max_seq_len,
// const bool is_fuse = true)
// {
//     if (std::is_same<T, half>::value && (sm == kSM_70 || sm == kSM_86 || sm == kSM_80 || sm == kSM_75 || sm ==
//     kSM_72)
//         && size_per_head == 64 && max_seq_len <= 384 && is_fuse == true) {
//         return remove_padding ? AttentionType::FUSED_MHA : AttentionType::FUSED_PADDED_MHA;
//     }
//     else {
//         return remove_padding ? AttentionType::UNFUSED_MHA : AttentionType::UNFUSED_PADDED_MHA;
//     }
// }

template<typename T1, typename T2>
class BaseAttentionFP8Layer: public BaseLayer {

public:
    virtual void forward(TensorMap*                        output_tensors,
                         TensorMap*                        input_tensors,
                         const AttentionFP8Weight<T1, T2>* attention_weights) = 0;

    BaseAttentionFP8Layer(cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward,
                          bool             sparse = false):
        BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse)
    {
    }
    virtual ~BaseAttentionFP8Layer() = default;
};

}  // namespace fastertransformer
