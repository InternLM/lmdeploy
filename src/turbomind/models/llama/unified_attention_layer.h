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

template<typename T>
class UnifiedAttentionLayer {
public:
    using WeightType = LlamaAttentionWeight<T>;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    using ForwardParam = attention::ForwardParam;

    std::shared_ptr<ForwardParam> CreateForwardParam(int max_batch_size);

    void allocateWorkspace();
    void freeWorkspace();

    ~UnifiedAttentionLayer()
    {
        freeWorkspace();

        for (auto& s : streams_) {
            s = {};
        }

        check_cuda_error(cudaEventDestroy(aux_event_));
        check_cuda_error(cudaEventDestroy(qkv_event_));
        check_cuda_error(cudaStreamDestroy(aux_stream_));

        aux_event_ = qkv_event_ = {};
        aux_stream_             = {};
    }

    UnifiedAttentionLayer(const ModelParam&     model,
                          const AttentionParam& attn,
                          const LoraParam&      lora,
                          size_t                tp_size,
                          const Context<T>&     context);

    void forward(ForwardParam& param);

    void prefill(T*                output,
                 T*                tmp_kv_buffer,
                 const T*          qkv,
                 void**            block_ptrs,
                 const int*        cu_q_len,
                 const int*        cu_k_len,
                 const int*        input_length,
                 const int*        context_length,
                 const int*        cu_block_count,
                 const bool*       is_finished,
                 const float*      rope_theta,
                 int               pf_batch_size,
                 int               pf_num_token,
                 size_t            layer_offset,
                 int               pf_max_q_len,
                 int               pf_max_k_len,
                 int               pf_session_len,
                 const WeightType* weights);

    void decode(T*                output,
                const T*          qkv,
                void**            block_ptrs,
                const int*        cu_q_len,
                const int*        cu_block_count,
                const int*        input_length,
                const int*        context_length,
                const bool*       is_finished,
                const float*      rope_theta,
                size_t            layer_offset,
                int               batch_size,
                int               dc_sum_seq_len,
                int               dc_max_seq_len,
                int               max_split_k,
                const WeightType* weights);

private:
    core::Tensor forward_mla(const core::Tensor& hidden_state, const WeightType& weights);

    void qk_norm(core::Tensor& qkv, const WeightType& weights);

private:
    const size_t head_num_;
    const size_t kv_head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_kv_head_num_;

    const AttentionParam param_;
    const ModelParam     model_param_;
    const LoraParam      lora_param_;
    const Context<T>&    context_;

    cudaStream_t const    stream_;
    LlamaLinear<T>* const linear_;
    IAllocator* const     allocator_;
    const int             arch_{};

    cudaStream_t aux_stream_;
    cudaEvent_t  qkv_event_;
    cudaEvent_t  aux_event_;

    std::array<cudaStream_t, 2> streams_;

    RNG rng_;

    RopeKernelParam rope_param_{};

    float* partial_M_{};
    float* partial_L_{};
    float* partial_O_{};
    int*   split_cnt_{};
    int*   barriers_{};  // always zero

    bool is_allocate_workspace_ = false;
};

}  // namespace turbomind
