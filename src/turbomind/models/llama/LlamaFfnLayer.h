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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.cc

#pragma once

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class LlamaFfnLayer {
public:
    LlamaFfnLayer(const ModelParam& model, const NcclParam& tp, const Context<T>& ctx, bool all_reduce):
        hidden_units_(model.hidden_units),
        tensor_para_(tp),
        stream_(ctx.stream),
        linear_(ctx.linear.get()),
        allocator_(ctx.allocator.get()),
        all_reduce_(all_reduce)
    {
    }

    ~LlamaFfnLayer()
    {
        freeBuffer();
    }

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaFfnWeight<T>* weights);

private:
    void allocateBuffer(size_t token_num, int inter_size, const LlamaDenseWeight<T>*, const LlamaDenseWeight<T>*);

    void freeBuffer();

    void activation(int token_num, int inter_size, bool is_chunked);

    const size_t          hidden_units_;
    const NcclParam       tensor_para_;
    cudaStream_t const    stream_;
    LlamaLinear<T>* const linear_;
    IAllocator* const     allocator_;
    const bool            all_reduce_;
    bool                  is_free_buffer_after_forward_{};

    T* gating_buf_{};
    T* inter_buf_{};

    bool is_allocate_buffer_{};
};

}  // namespace turbomind
