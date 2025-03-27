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
#include "src/turbomind/utils/nvtx_utils.h"

namespace turbomind {

template<typename T>
void LlamaFfnLayer<T>::activation(core::Tensor& gating, core::Tensor& inter)
{
    FT_CHECK(gating.shape() == inter.shape());
    FT_CHECK(gating.ndim() == 2);
    FT_CHECK(gating.stride(0) == inter.stride(0));
    const auto stride     = gating.stride(0);
    const auto [num, dim] = gating.shapes(0, 1);
    invokeGenericActivation_v2<SiluActivation>(gating.data<T>(), inter.data<T>(), stride, num, dim, stream_);
}

template<typename T>
void LlamaFfnLayer<T>::forward(ForwardParam&& param)
{
    NvtxScope scope("ffn");

    const auto token_num     = param.input.shape(0);
    const auto weights       = param.weights;
    const int  layer_id      = param.layer_id;
    const int  inter_size    = weights->inter_size;
    const bool is_fused_silu = weights->fused_gating_intermediate.kernel && weights->is_fused_silu;

    core::Tensor gating;
    core::Tensor inter;

    if (weights->fused_gating_intermediate.kernel) {
        NvtxScope scope("fused_silu_ffn");

        const auto type = weights->is_fused_silu ? LlamaLinear<T>::kFusedSiluFfn : LlamaLinear<T>::kGemm;

        auto mix = linear_->forward(param.input, weights->fused_gating_intermediate, type);
        sync_check_cuda_error();

        gating = mix.slice({0, 0}, {(int)token_num, inter_size});

        if (!weights->is_fused_silu) {
            inter = mix.slice({0, inter_size}, {(ssize_t)token_num, inter_size});
        }
    }
    else {
        FT_CHECK(weights->gating.lora.r + weights->intermediate.lora.r == 0);

        {  // w1(x)
            NvtxScope scope("w1");
            gating = linear_->forward(param.input, weights->gating, LlamaLinear<T>::kGemm);
            sync_check_cuda_error();
        }
        count_and_fix(gating.data<T>(), token_num * weights->gating.output_dims, Concat("w1", layer_id), 3);

        {  // w3(x)
            NvtxScope scope("w3");
            inter = linear_->forward(param.input, weights->intermediate, LlamaLinear<T>::kGemm);
            sync_check_cuda_error();
        }
        count_and_fix(inter.data<T>(), token_num * weights->intermediate.output_dims, Concat("w3", layer_id), 3);
    }

    if (!weights->is_fused_silu) {
        // silu(w1(x)) * w3(x)
        activation(gating, inter);
        sync_check_cuda_error();
        count_and_fix(gating.data<T>(), token_num * weights->output.input_dims, Concat("act", layer_id), 3);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        (void)linear_->forward(gating, weights->output, LlamaLinear<T>::kGemm, &param.output);
        sync_check_cuda_error();
    }

    count_and_fix((T*)param.output.raw_data(), token_num * weights->output.output_dims, Concat("w2", layer_id), 3);
}

#ifdef ENABLE_FP32
template class LlamaFfnLayer<float>;
#endif
template class LlamaFfnLayer<half>;
#ifdef ENABLE_BF16
template class LlamaFfnLayer<__nv_bfloat16>;
#endif

}  // namespace turbomind
