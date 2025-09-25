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

#include <array>

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

class UnifiedAttentionLayer {
public:
    using WeightType = LlamaAttentionWeight;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    struct ForwardParam {
        Tensor            input;
        Tensor            output;
        const WeightType* weights;
        int               layer_id;
    };

    ~UnifiedAttentionLayer();

    UnifiedAttentionLayer(const ModelParam&     model,
                          const AttentionParam& attn,
                          const EngineParam&    engine,
                          const LoraParam&      lora,
                          int                   tp_size,
                          const Context&        context);

    void Forward(ForwardParam p);

    void Initialize(TensorMap& args);

    void Finalize();

    const int* d_cu_q_len()
    {
        return d_cu_q_len_;
    }

private:
    Tensor forward_mla(const Tensor& hidden_state, const WeightType& weights);

    /// TODO: dropping the `T` here requires deep refactor of attention dispatch
    template<class T>
    Tensor core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights);

    template<class T>
    void cp_postprocess(Tensor& attn);

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
    const LoraParam      lora_param_;
    const Context&       context_;

    cudaStream_t const stream_;
    LlamaLinear&       linear_;
    const int          arch_{};

    cudaStream_t aux_stream_;
    cudaEvent_t  qkv_event_;
    cudaEvent_t  aux_event_;

    const int                   attn_cp_group_;
    comm::DeviceCommImpl* const d_comm_;

    std::array<cudaStream_t, 2> streams_;

    RNG rng_;

    RopeKernelParam rope_param_{};

    ///////////////////////////////////////////////////////
    /// runtime states
    int decode_num_;
    int prefil_num_;

    Tensor_<float> partial_M_;
    Tensor_<float> partial_L_;
    Tensor_<float> partial_O_;
    Tensor_<int>   split_cnt_;
    Tensor_<int>   barriers_;  // always zero

    // context parallel
    Tensor_<float> cp_ML_;

    Event event_;

    Buffer_<int> h_q_len_;
    Buffer_<int> h_k_len_;

    Buffer_<int> d_cu_x_len_;
    Buffer_<int> h_cu_x_len_;

    // references into d/h_cu_x_len_
    int* d_cu_q_len_;
    int* d_cu_k_len_;
    int* h_cu_q_len_;
    int* h_cu_k_len_;

    Buffer_<bool>  finished_;
    Buffer_<float> rope_base_;

    Buffer_<int>       cu_block_nums_;
    Buffer_<uintptr_t> kv_block_ptrs_;
    ///////////////////////////////////////////////////////
};

}  // namespace turbomind
