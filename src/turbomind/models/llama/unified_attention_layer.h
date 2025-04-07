/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

namespace attention {

struct ForwardParam;

void Initialize(ForwardParam& p, core::TensorMap& args, const core::Tensor& input, core::Tensor& output);

void SetLayer(ForwardParam& p, const void* weights, int layer_id);

void Finalize(ForwardParam& p);

const int* d_cu_q_len(ForwardParam& p);

}  // namespace attention

class UnifiedAttentionLayer {
public:
    using WeightType = LlamaAttentionWeight;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    using ForwardParam = attention::ForwardParam;

    std::shared_ptr<ForwardParam> CreateForwardParam(int max_batch_size);

    ~UnifiedAttentionLayer();

    UnifiedAttentionLayer(const ModelParam&     model,
                          const AttentionParam& attn,
                          const LoraParam&      lora,
                          int                   tp_size,
                          const Context&        context);

    void forward(ForwardParam& param);

private:
    core::Tensor forward_mla(const core::Tensor& hidden_state, const WeightType& weights);

    /// TODO: dropping the `T` here requires deep refactor of attention dispatch
    template<class T>
    core::Tensor core_attention(core::Tensor& qkv, const ForwardParam& p, const WeightType& weights);

    void qk_norm(core::Tensor& qkv, const WeightType& weights);

private:
    const int head_num_;
    const int kv_head_num_;
    const int size_per_head_;
    const int hidden_units_;
    const int local_head_num_;
    const int local_kv_head_num_;

    const AttentionParam param_;
    const ModelParam     model_param_;
    const LoraParam      lora_param_;
    const Context&       context_;

    cudaStream_t const stream_;
    LlamaLinear&       linear_;
    const int          arch_{};

    cudaStream_t aux_stream_;
    cudaEvent_t  qkv_event_;
    cudaEvent_t  aux_event_;

    std::array<cudaStream_t, 2> streams_;

    RNG rng_;

    RopeKernelParam rope_param_{};

    core::Tensor_<float> partial_M_;
    core::Tensor_<float> partial_L_;
    core::Tensor_<float> partial_O_;
    core::Tensor_<int>   split_cnt_;
    core::Tensor_<int>   barriers_;  // always zero
};

}  // namespace turbomind
