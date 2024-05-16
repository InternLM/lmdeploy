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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/attention_layers/GptContextAttentionLayer.h

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class UnifiedAttentionLayer {
public:
    using WeightType = LlamaAttentionWeight<T>;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    void freeBuffer();
    void allocateBuffer(size_t q_count, size_t k_count, size_t batch_size, const WeightType* weights);

    void allocateWorkspace();
    void freeWorkspace();

    ~UnifiedAttentionLayer() {
        freeBuffer();
        freeWorkspace();
    }

    UnifiedAttentionLayer(size_t               head_num,
                          size_t               kv_head_num,
                          size_t               size_per_head,
                          LlamaAttentionParams attn_params,
                          NcclParam            tensor_para,
                          LoraParams           lora_params,
                          cudaStream_t         stream,
                          cublasMMWrapper*     cublas_wrapper,
                          IAllocator*          allocator,
                          bool                 is_free_buffer_after_forward,
                          int                  cache_block_seq_len,
                          int                  quant_policy):
        head_num_(head_num),
        size_per_head_(size_per_head),
        hidden_units_(head_num * size_per_head),
        local_head_num_(head_num / tensor_para.world_size_),
        local_kv_head_num_(kv_head_num / tensor_para.world_size_),
        head_n_rep_(head_num / kv_head_num),
        params_(attn_params),
        tensor_para_(tensor_para),
        lora_params_(lora_params),
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        linear_(cublas_wrapper, stream),
        allocator_(allocator),
        kv_cache_block_len_(cache_block_seq_len),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        quant_policy_(quant_policy)
    {
        FT_CHECK(head_num % kv_head_num == 0);
        arch_ = getSMVersion();

        check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
        check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
        check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

        streams_[0] = stream_;
        streams_[1] = aux_stream_;

        allocateWorkspace();
    }

    void forward(TensorMap* outputs, const TensorMap* inputs, const LlamaAttentionWeight<T>* weights);

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
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_kv_head_num_;
    const size_t local_head_num_;
    const size_t head_n_rep_;
    const size_t kv_cache_block_len_;
    const bool   is_free_buffer_after_forward_;

    const LlamaAttentionParams params_;

    const int quant_policy_;

    NcclParam tensor_para_;

    LoraParams lora_params_;

    cudaStream_t     stream_;
    IAllocator*      allocator_;
    cublasMMWrapper* cublas_wrapper_;
    LlamaLinear<T>   linear_;

    cudaStream_t aux_stream_;
    cudaEvent_t  qkv_event_;
    cudaEvent_t  aux_event_;

    std::array<cudaStream_t, 2> streams_;

    int arch_{};

    T*     qkv_buf_{};
    T*     q_buf_2_{};
    T*     k_buf_2_{};
    T*     v_buf_2_{};
    T*     k_cache_buf_{};
    T*     v_cache_buf_{};
    T*     qk_buf_{};
    float* qk_buf_float_{};
    T*     qkv_buf_2_{};
    T*     qkv_buf_3_{};

    float* partial_M_{};
    float* partial_L_{};
    float* partial_O_{};
    int*   split_cnt_{};
    int*   barriers_{};  // always zero

    T* tmp_kv_buf_{};

    bool is_allocate_buffer_    = false;
    bool is_allocate_workspace_ = false;
};

}  // namespace turbomind
