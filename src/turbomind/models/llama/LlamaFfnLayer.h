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

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

template<typename T>
class LlamaFfnLayer {
public:
    LlamaFfnLayer(const ModelParam& model, const Context<T>& ctx):
        hidden_units_(model.hidden_units),
        stream_(ctx.stream),
        linear_(ctx.linear.get()),
        allocator_(ctx.allocator.get())
    {
    }

    struct ForwardParam {
        core::Tensor             input;
        core::Tensor             output;
        const LlamaFfnWeight<T>* weights;
        int                      layer_id;
    };

    void forward(ForwardParam&& param);

private:
    void activation(core::Tensor& gating, core::Tensor& inter);

private:
    const size_t          hidden_units_;
    cudaStream_t const    stream_;
    LlamaLinear<T>* const linear_;
    IAllocator* const     allocator_;
};

}  // namespace turbomind
