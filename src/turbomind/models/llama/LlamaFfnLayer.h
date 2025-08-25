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

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class LlamaFfnLayer {
public:
    LlamaFfnLayer(const ModelParam& model, const Context& ctx): hidden_units_(model.hidden_units), linear_(*ctx.linear)
    {
    }

    struct ForwardParam {
        Tensor                input;
        Tensor                output;
        const LlamaFfnWeight* weights;
        int                   layer_id;
    };

    void forward(ForwardParam param);

private:
    const size_t hidden_units_;
    LlamaLinear& linear_;
};

}  // namespace turbomind
