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

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/kernels/attention/cp_utils.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct AttentionData;

class UnifiedAttentionLayer {
public:
    using WeightType = LlamaAttentionWeight;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    struct ForwardParam {
        int               phase;
        Tensor            input;
        Tensor            output;
        const WeightType* weights;
        int               layer_id;
    };

    ~UnifiedAttentionLayer();

    UnifiedAttentionLayer(const ModelParam&     model,
                          const AttentionParam& attn,
                          const EngineParam&    engine,
                          int                   tp_size,
                          const Context&        context,
                          int                   phases,
                          bool                  init);

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

private:
    void Setup(int phase, TensorMap& env);

    Tensor forward_mla(const Tensor& hidden_state, const WeightType& weights);

    /// TODO: dropping the `T` here requires deep refactor of attention dispatch
    template<class T>
    Tensor core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights);

    void qk_norm(Tensor& qkv, const WeightType& weights);

private:
    const int head_num_;
    const int kv_head_num_;
    const int size_per_head_;
    const int hidden_units_;
    const int local_head_num_;
    const int local_kv_head_num_;

    const AttentionParam param_;
    const EngineParam    engine_param_;
    const ModelParam     model_param_;
    const Context&       context_;

    int& is_warm_up_;

    LlamaLinear& linear_;
    const int    arch_{};

    cudaStream_t aux_stream_;
    cudaEvent_t  qkv_event_;
    cudaEvent_t  aux_event_;

    RNG rng_;

    RopeKernelParam rope_param_{};

    std::vector<std::shared_ptr<AttentionData>> data_;

    ///////////////////////////////////////////////////////
    /// temp runtime buffers
    Tensor_<float> partial_O_;
    Tensor_<float> partial_ML_;
    Tensor_<int>   split_cnt_;
    Tensor         tmp_attn_;

    Buffer_<float> rope_base_buf_;
    Buffer_<int>   mrope_position_delta_buf_;
    Buffer_<int>   mrope_length_buf_;

    CpPostContext cp_fn_ctx_;  // context parallel
};

}  // namespace turbomind
