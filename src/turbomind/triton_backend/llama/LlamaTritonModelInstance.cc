/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/triton_backend/llama/LlamaTritonModelInstance.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace turbomind {

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, Tensor>* outputs, void* ctx)
{
    LlamaTritonModelInstance<T>* model = reinterpret_cast<LlamaTritonModelInstance<T>*>(ctx);
    model->stream_cb_(std::make_shared<std::unordered_map<std::string, Tensor>>(*outputs), model->stream_ctx_);
}

template<typename T>
LlamaTritonModelInstance<T>::LlamaTritonModelInstance(Engine<T>&                                      instance,
                                                      std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator,
                                                      int                                             device_id):
    device_id_{device_id}, instance_(&instance), allocator_(std::move(allocator))
{
}

template<typename T>
std::string format_vector(const std::vector<T>& vec)
{
    std::stringstream ss;
    ss << "[";
    bool first = true;
    for (const auto& x : vec) {
        ss << (first ? "" : ", ") << x;
        first = false;
    }
    ss << "]";
    return ss.str();
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, Tensor>>
LlamaTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, Tensor>> inputs)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // In some cases, this is needed to trigger the creation of CUDA context, or later `cudaMallocAsync` will die
    check_cuda_error(cudaSetDevice(device_id_));

    FT_CHECK_WITH_INFO(inputs->at("input_ids").shape.size() == 2, "inputs->at(\"input_ids\").shape.size() == 2");
    FT_CHECK_WITH_INFO(inputs->at("input_lengths").shape.size() == 1,
                       "inputs->at(\"input_lengths\").shape.size() == 1");

    const uint32_t request_batch_size     = inputs->at("input_ids").shape[0];
    const uint32_t max_request_output_len = (size_t)*std::max_element((int*)inputs->at("request_output_len").data,
                                                                      (int*)inputs->at("request_output_len").data
                                                                          + inputs->at("request_output_len").shape[0]);
    // const uint32_t total_output_len = max_request_output_len + input_tensors->at("input_ids").shape[1];
    const uint32_t beam_width = inputs->count("beam_width") ? (size_t)(*(uint*)inputs->at("beam_width").data) : 1;
    FT_CHECK_WITH_INFO(beam_width == 1, "Beam search is not implemented");

    h_total_output_lengths_ =
        (uint32_t*)std::realloc((void*)h_total_output_lengths_, request_batch_size * sizeof(uint32_t));

    const size_t max_input_len    = inputs->at("input_ids").shape[1];
    const bool   is_return_logits = inputs->count("is_return_logits") && *(bool*)inputs->at("is_return_logits").data;

    const size_t vocab_size = instance_->model().vocab_size();

    allocateBuffer(request_batch_size, max_input_len, beam_width, instance_->session_len(), is_return_logits);

    std::unordered_map<std::string, Tensor> outputs{
        {"output_ids",
         Tensor{MEMORY_CPU,
                TYPE_UINT32,
                std::vector<size_t>{request_batch_size, beam_width, (size_t)instance_->session_len()},
                d_output_ids_}},
        {"sequence_length",
         Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths_}}};

    if (inputs->count("is_return_log_probs") && *((bool*)inputs->at("is_return_log_probs").data)) {
        outputs.insert({"output_log_probs",
                        Tensor{MEMORY_GPU,
                               TYPE_FP32,
                               std::vector<size_t>{request_batch_size, beam_width, max_request_output_len},
                               d_output_log_probs_}});
        outputs.insert(
            {"cum_log_probs",
             Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{request_batch_size, beam_width}, d_cum_log_probs_}});
    }

    if (inputs->count("logprobs")) {
        size_t max_logprob_length = std::min((int)max_request_output_len, instance_->session_len()) + 1;
        h_logprob_vals_           = (float*)std::realloc(
            h_logprob_vals_, sizeof(float) * request_batch_size * beam_width * max_logprob_length * kMaxLogProb);
        h_logprob_indexes_ = (uint32_t*)std::realloc(
            h_logprob_indexes_, sizeof(uint32_t) * request_batch_size * beam_width * max_logprob_length * kMaxLogProb);
        h_logprob_nums_ = (uint32_t*)std::realloc(
            h_logprob_nums_, sizeof(uint32_t) * request_batch_size * beam_width * max_logprob_length);

        outputs.insert({{"logprob_vals",
                         Tensor{MEMORY_CPU,
                                TYPE_FP32,
                                std::vector<size_t>{request_batch_size, beam_width, max_logprob_length, kMaxLogProb},
                                h_logprob_vals_}}});

        outputs.insert({{"logprob_indexes",
                         Tensor{MEMORY_CPU,
                                TYPE_UINT32,
                                std::vector<size_t>{request_batch_size, beam_width, max_logprob_length, kMaxLogProb},
                                h_logprob_indexes_}}});

        outputs.insert({{"logprob_nums",
                         Tensor{MEMORY_CPU,
                                TYPE_UINT32,
                                std::vector<size_t>{request_batch_size, beam_width, max_logprob_length},
                                h_logprob_nums_}}});
    }

    if (is_return_logits) {
        outputs.insert(
            {{"logits", {MEMORY_GPU, TYPE_FP32, {request_batch_size, max_input_len, vocab_size}, d_output_logits_}}});
    }

    try {
        Request::Callback callback;

        if (stream_cb_) {
            callback = [this](std::unordered_map<std::string, Tensor>* outputs) {
                triton_stream_callback<T>(outputs, this);
            };
        }

        check_cuda_error(cudaStreamSynchronize(allocator_->returnStream()));

        instance_->Submit(&outputs, inputs.get(), {callback});
        // ! stream synced by the model before returning
    }
    catch (...) {
        h_exception_ = std::current_exception();
        outputs.insert({"error_message", Tensor{MEMORY_CPU, TYPE_BYTES, {1}, &h_exception_}});
    }

    return std::make_shared<std::unordered_map<std::string, Tensor>>(std::move(outputs));
}

template<typename T>
LlamaTritonModelInstance<T>::~LlamaTritonModelInstance()
{
    freeBuffer();
}

template<typename T>
void LlamaTritonModelInstance<T>::allocateBuffer(const size_t request_batch_size,
                                                 const size_t max_input_len,
                                                 const size_t beam_width,
                                                 const size_t session_len,
                                                 const bool   is_return_logits)
{
    d_output_ids_ = (int*)std::realloc(d_output_ids_, sizeof(int) * request_batch_size * beam_width * session_len);
    d_sequence_lengths_ = (int*)std::realloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width);

    if (is_return_logits) {
        d_output_logits_ = (float*)allocator_->reMalloc(d_output_logits_,
                                                        sizeof(float) * request_batch_size * max_input_len
                                                            * instance_->model().vocab_size(),
                                                        false);
    }
}

template<typename T>
void LlamaTritonModelInstance<T>::freeBuffer()
{
    std::free(d_output_ids_);
    std::free(d_sequence_lengths_);
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_cum_log_probs_));
    std::free(h_total_output_lengths_);
    std::free(h_logprob_vals_);
    std::free(h_logprob_indexes_);
    std::free(h_logprob_nums_);
}

#ifdef ENABLE_FP32
template struct LlamaTritonModelInstance<float>;
#endif
template struct LlamaTritonModelInstance<half>;
#ifdef ENABLE_BF16
template struct LlamaTritonModelInstance<__nv_bfloat16>;
#endif

}  // namespace turbomind
