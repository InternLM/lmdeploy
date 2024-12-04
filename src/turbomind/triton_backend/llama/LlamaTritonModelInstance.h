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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h

#pragma once

#include <memory>

#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"

namespace turbomind {

template<typename T>
struct LlamaTritonModelInstance: AbstractTransformerModelInstance {

    LlamaTritonModelInstance(Engine<T>&                                      instance,
                             std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator,
                             int                                             device_id);
    ~LlamaTritonModelInstance() override;

    virtual std::shared_ptr<std::unordered_map<std::string, Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, Tensor>> input_tensors) override;

private:
    Engine<T>*                                            instance_;
    const std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator_;

    void allocateBuffer(const size_t request_batch_size,
                        const size_t max_input_len,
                        const size_t beam_width,
                        const size_t session_len,
                        const bool   is_return_logits);
    void freeBuffer();

    int device_id_;

    int*   d_input_ids_                = nullptr;
    int*   d_input_lengths_            = nullptr;
    int*   d_input_bad_words_          = nullptr;
    int*   d_input_stop_words_         = nullptr;
    int*   d_request_prompt_lengths_   = nullptr;
    T*     d_request_prompt_embedding_ = nullptr;
    float* d_top_p_decay_              = nullptr;
    float* d_top_p_min_                = nullptr;
    int*   d_top_p_reset_ids_          = nullptr;

    int*   d_output_ids_       = nullptr;
    int*   d_sequence_lengths_ = nullptr;
    float* d_output_log_probs_ = nullptr;
    float* d_cum_log_probs_    = nullptr;
    float* d_output_logits_    = nullptr;

    float*    h_logprob_vals_    = nullptr;
    uint32_t* h_logprob_indexes_ = nullptr;
    uint32_t* h_logprob_nums_    = nullptr;

    uint32_t*          h_total_output_lengths_ = nullptr;
    std::exception_ptr h_exception_            = nullptr;
};

}  // namespace turbomind
