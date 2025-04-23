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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.h

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"

namespace turbomind {

void LlamaFfnLayer::activation(Tensor& gating, Tensor& inter, cudaStream_t stream)
{
    // Code for dispatching activation types
    invokeGenericActivation_v3<SiluActivation>(gating, inter, stream);
}

void LlamaFfnLayer::forward(ForwardParam param)
{
    NvtxScope scope("ffn");

    const auto& mlp = *param.weights;

    const int token_num  = param.input.shape(0);
    const int inter_size = mlp.inter_size;
    const int layer_id   = param.layer_id;

    const auto stream = core::Context::stream().handle();

    Tensor gating;
    Tensor inter;

    if (mlp.fused_gating_intermediate.weight) {
        const auto type = mlp.is_fused_silu ? LlamaLinear::kFusedSiluFfn : LlamaLinear::kGemm;

        auto mix = linear_.forward(param.input, mlp.fused_gating_intermediate, type);
        sync_check_cuda_error();

        gating = mix.slice({0, 0}, {(int)token_num, inter_size});
        if (!mlp.is_fused_silu) {
            inter = mix.slice({0, inter_size}, {(ssize_t)token_num, inter_size});
        }
    }
    else {
        gating = linear_.forward(param.input, mlp.gating, LlamaLinear::kGemm);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(gating, Concat("w1", layer_id), 3);

        inter = linear_.forward(param.input, mlp.intermediate, LlamaLinear::kGemm);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(inter, Concat("w3", layer_id), 3);
    }

    if (!mlp.is_fused_silu) {
        // silu(w1(x)) * w3(x)
        activation(gating, inter, stream);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(gating, Concat("act", layer_id), 3);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        linear_.forward(gating, mlp.output, LlamaLinear::kGemm, param.output);
        sync_check_cuda_error();
    }
}

}  // namespace turbomind
