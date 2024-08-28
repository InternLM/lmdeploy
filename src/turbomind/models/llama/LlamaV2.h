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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h

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
#include <limits>
#include <unordered_map>

namespace turbomind {

template<typename T>
class LlamaV2 {
public:
    ~LlamaV2();

    LlamaV2(const ModelParam&               model,
            const AttentionParam&           attn,
            const LoraParam&                lora,
            const NcclParam&                tp,
            const Context<T>&               ctx,
            int                             max_batch_size,
            std::shared_ptr<LlamaWeight<T>> weights);

    void tune();

    size_t vocab_size() const noexcept
    {
        return vocab_size_;
    }

private:
    void embeddingLookup(T* embeddings, const int* token_ids_buf, int batch_size, int step);

    void updateEmbedding(T*               decoder_input,
                         const int        bsz,
                         const int*       h_input_length,
                         const Sequence** sequences,
                         int              token_num,
                         int*             lora_mask,
                         bool*            have_embeddings);

    void forwardUnified(T*               out,
                        T*               decoder_output,
                        T*               decoder_input,
                        void**           block_ptrs,
                        const int*       cu_block_cnts,
                        const int*       input_ids,
                        const int*       h_input_length,
                        const int*       h_context_length,
                        const float*     rope_theta,
                        const bool*      finished,
                        size_t           token_num,
                        int              dc_batch_size,
                        int              pf_batch_size,
                        int*             lora_mask,
                        const Sequence** sequences);

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

    const ModelParam     param_;
    const AttentionParam attn_param_;
    const LoraParam      lora_param_;

    const size_t    head_num_;
    const size_t    size_per_head_;
    const size_t    hidden_units_;
    const size_t    inter_size_;
    const size_t    layer_num_;
    const size_t    vocab_size_;
    const size_t    vocab_size_padded_;
    const float     rmsnorm_eps_;
    const int       start_id_;
    const int       end_id_;
    const NcclParam tensor_para_;
    const size_t    local_head_num_;
    const size_t    local_kv_head_num_;

    const std::shared_ptr<LlamaWeight<T>> weights_{};

    // Refs into `Context<T>`, make the pointer constant (not the pointed objects)
    cudaStream_t const     stream_;
    cublasMMWrapper* const cublas_wrapper_;
    IAllocator* const      allocator_;
    IAllocator* const      peer_allcator_;
    LlamaLinear<T>* const  linear_;

    const bool is_free_buffer_after_forward_;
    const bool debug_;

    std::unique_ptr<UnifiedDecoder<T>>         unified_decoder_;
    std::unique_ptr<DynamicDecodeLayer<float>> dynamic_decode_layer_;
};

}  // namespace turbomind
