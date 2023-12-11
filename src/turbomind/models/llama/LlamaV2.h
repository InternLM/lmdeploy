/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGpt.h

#pragma once

#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/models/llama/Barrier.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/instance_comm.h"
#include "src/turbomind/utils/nccl_utils.h"
#include <unordered_map>

using ffi_api_lock_ctrl_t = std::function<void(int)>;

namespace turbomind {

template<typename T>
class LlamaV2 {
public:
    struct SharedState {
        std::vector<std::shared_ptr<Request>> infer_requests;
        std::vector<std::shared_ptr<Request>> stop_requests;
        RequestQueue                          request_queue;
        std::shared_ptr<Barrier>              barrier;
        bool                                  abort;
    };

    ~LlamaV2();

    LlamaV2(size_t                       head_num,
            size_t                       kv_head_num,
            size_t                       size_per_head,
            size_t                       inter_size,
            size_t                       num_layer,
            size_t                       vocab_size,
            float                        norm_eps,
            const LlamaAttentionParams&  attn_params,
            int                          start_id,
            int                          end_id,
            int                          cache_block_seq_len,
            int                          quant_policy,
            bool                         use_context_fmha,
            const EngineParams&          engine_params,
            std::shared_ptr<SharedState> shared_state,
            LlamaWeight<T>*              weights,
            NcclParam                    tensor_para,
            cudaStream_t                 stream,
            cublasMMWrapper*             cublas_wrapper,
            IAllocator*                  allocator,
            bool                         is_free_buffer_after_forward,
            cudaDeviceProp*              cuda_device_prop);

    struct Control {
        AbstractInstanceComm* comm;
        Request::Callback     callback;
    };

    void forward(std::unordered_map<std::string, Tensor>*       outputs,
                 const std::unordered_map<std::string, Tensor>* inputs,
                 Control                                        control);

    void stop(const std::vector<uint64_t>& seq_ids);

    size_t vocab_size() const noexcept
    {
        return vocab_size_;
    }

    void setFfiLock(ffi_api_lock_ctrl_t func)
    {
        ffi_lock_ = func;
    }

private:
    friend class Batch;

    void initialize(const LlamaAttentionParams& attn_params,
                    size_t                      kv_head_num,
                    bool                        use_context_fmha,
                    int                         cache_block_seq_len,
                    int                         quant_policy);

    void embeddingLookup(T* embeddings, const int* token_ids_buf, int batch_size, int step);

    void forwardUnified(T*           out,
                        T*           decoder_output,
                        T*           decoder_input,
                        void**       k_block_ptrs,
                        void**       v_block_ptrs,
                        const int*   input_ids,
                        const int*   cu_block_cnts,
                        const float* rope_theta,
                        const bool*  dc_finished,
                        const int*   pf_input_length,
                        const int*   pf_context_length,
                        T**          pf_tmp_k_ptrs,
                        T**          pf_tmp_v_ptrs,
                        size_t       token_num,
                        int          dc_batch_size,
                        int          dc_step,
                        int          dc_sum_seq_len,
                        int          dc_max_seq_len,
                        int          pf_batch_size,
                        int          pf_max_input_len,
                        int          pf_max_context_len,
                        int          pf_session_len);

    void postDecodeEmbedding(float* logits, float* local_logits, const T* decoder_output, int batch_size);

    void dynamicDecode(int*            token_ids,
                       bool*           finished,
                       int*            sequence_length,
                       bool*           should_stop,
                       curandState_t*  curand_state,
                       TensorMap*      inputs,
                       TensorMap*      outputs,
                       const float*    logits,
                       const uint32_t* seq_limit_len,
                       const int*      context_length,
                       const int*      end_ids,
                       int             step,
                       int             ite,
                       size_t          max_context_len,
                       size_t          token_ids_len,
                       size_t          batch_size);

private:
    friend class LlamaBatch<T>;

    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t num_layer_;
    const size_t vocab_size_;
    size_t       vocab_size_padded_;
    float        rmsnorm_eps_ = 1e-6f;

    const LlamaAttentionParams attn_params_;

    static constexpr bool neox_rotary_style_ = false;

    const int    start_id_;
    const int    end_id_;
    const size_t hidden_units_;

    const size_t local_head_num_;
    const size_t local_kv_head_num_;
    NcclParam    tensor_para_;

    cudaStream_t     stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator*      allocator_;
    bool             is_free_buffer_after_forward_;
    cudaDeviceProp*  cuda_device_prop_;

    const bool debug_{false};

    LlamaWeight<T>* weights_{};

    std::unique_ptr<UnifiedDecoder<T>> unified_decoder_;
    DynamicDecodeLayer<float>*         dynamic_decode_layer_{};

    std::shared_ptr<SharedState>   shared_state_;
    ffi_api_lock_ctrl_t            ffi_lock_;
    std::unique_ptr<LlamaBatch<T>> batch_;
};

}  // namespace turbomind
