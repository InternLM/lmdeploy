/*
 * Copyright (c) OpenMMLab. All rights reserved.
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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/FfnLayer.cc

#pragma once

// #include "src/turbomind/layers/FfnLayer.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/nccl_utils.h"
#include <functional>

namespace turbomind {

template<typename T>
class LlamaFfnLayer {
public:
    LlamaFfnLayer(size_t           head_num,
                  size_t           size_per_head,
                  size_t           inter_size,
                  NcclParam        tensor_para,
                  cudaStream_t     stream,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator*      allocator,
                  bool             is_free_buffer_after_forward):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size / tensor_para.world_size_),
        hidden_units_(head_num * size_per_head),
        stream_(stream),
        linear_(cublas_wrapper, stream),
        allocator_(allocator),
        tensor_para_(tensor_para),
        is_free_buffer_after_forward_(is_free_buffer_after_forward)
    {
    }

    ~LlamaFfnLayer()
    {
        freeBuffer();
    }

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaFfnWeight<T>* weights);

private:
    void allocateBuffer(size_t token_num);

    void freeBuffer();

    void activation(int num_token);

    size_t         head_num_;
    size_t         size_per_head_;
    size_t         inter_size_;
    size_t         hidden_units_;
    cudaStream_t   stream_;
    LlamaLinear<T> linear_;
    IAllocator*    allocator_;
    bool           is_free_buffer_after_forward_;

    T* gating_buf_{};
    T* inter_buf_{};

    NcclParam tensor_para_;

    bool is_allocate_buffer_{};
};

}  // namespace turbomind
