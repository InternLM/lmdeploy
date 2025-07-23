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

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/layers/DynamicDecodeLayer.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_decoder.h"

namespace turbomind {

class LlamaBatch;

class LlamaV2 {
public:
    LlamaV2(DataType                     dtype,
            const ModelParam&            model,
            const EngineParam&           engine,
            const AttentionParam&        attn,
            const MoeParam&              moe,
            const LoraParam&             lora,
            const Context&               ctx,
            int                          max_batch_size,
            std::shared_ptr<LlamaWeight> weights);

    size_t vocab_size() const noexcept
    {
        return vocab_size_;
    }

private:
    void updateEmbedding(char*            decoder_input,
                         const int        bsz,
                         const int*       h_input_length,
                         const Sequence** sequences,
                         int              token_num,
                         int*             lora_mask,
                         bool*            have_embeddings);

    void Forward(Buffer_<int>     input_ids,
                 Tensor           hidden_states_out,
                 Tensor           decoder_out,
                 Buffer           kv_block_ptrs,
                 Buffer           cu_block_nums,
                 Buffer_<int>     h_input_length,
                 Buffer_<int>     h_context_length,
                 Buffer           rope_base,
                 MropeRope*       mrope,
                 Buffer           finished,
                 Buffer           local_token_nums,
                 Buffer           lora_mask,
                 int              decode_num,
                 int              prefil_num,
                 const Sequence** sequences);

    Tensor postDecodeEmbedding(const Tensor& features, Buffer local_logits);

    void dynamicDecode(Buffer token_ids,
                       Buffer finished,
                       Buffer sequence_length,
                       Tensor curand_state,
                       Tensor logits,
                       Buffer seq_limit_len,
                       Buffer init_context_length,
                       Buffer context_length,
                       Buffer prompt_length,
                       Buffer sampled_logprobs,  // <- indicator
                       Buffer sampled_indexes,
                       Buffer sampled_nums,
                       int    step,
                       int    max_context_len);

private:
    friend class LlamaBatch;

    const DataType dtype_;

    const ModelParam     param_;
    const AttentionParam attn_param_;
    const LoraParam      lora_param_;

    const Communicators* const comm_;

    const int    tp_size_;
    const int    tp_rank_;
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const size_t vocab_size_padded_;
    const float  rmsnorm_eps_;
    const size_t local_head_num_;
    const size_t local_kv_head_num_;

    const std::shared_ptr<LlamaWeight> weights_;

    // Refs into `Context`, make the pointer constant (not the pointed objects)
    cudaStream_t const stream_;
    LlamaLinear&       linear_;

    bool use_allgather_2d_{false};

    const bool debug_;

    std::unique_ptr<UnifiedDecoder>     unified_decoder_;
    std::unique_ptr<DynamicDecodeLayer> dynamic_decode_;
};

}  // namespace turbomind
