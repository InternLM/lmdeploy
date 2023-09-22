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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h

#include "src/turbomind/triton_backend/llama/LlamaTritonModelInstance.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/triton_backend/triton_utils.hpp"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ft = turbomind;

template<typename T>
void triton_stream_callback(std::unordered_map<std::string, ft::Tensor>* output_tensors, void* ctx)
{
    LlamaTritonModelInstance<T>* model  = reinterpret_cast<LlamaTritonModelInstance<T>*>(ctx);
    auto                         result = LlamaTritonModelInstance<T>::convert_outputs(*output_tensors);

    model->stream_cb_(result, model->stream_ctx_);
}

template<typename T>
LlamaTritonModelInstance<T>::LlamaTritonModelInstance(
    std::shared_ptr<LlamaTritonSharedModelInstance<T>>      instance,
    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator):
    instance_(std::move(instance)), allocator_(std::move(allocator))
{
}

template<typename T>
std::unordered_map<std::string, ft::Tensor> LlamaTritonModelInstance<T>::convert_inputs(
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    move_tensor_H2D(input_tensors->at("input_ids"), d_input_ids_, &allocator_);
    move_tensor_H2D(input_tensors->at("input_lengths"), d_input_lengths_, &allocator_);

    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t input_data_len     = input_tensors->at("input_ids").shape[1];
    h_total_output_lengths_ =
        (uint32_t*)std::realloc((void*)h_total_output_lengths_, request_batch_size * sizeof(uint32_t));

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"input_ids", as_GPU_tensor(input_tensors->at("input_ids"), d_input_ids_)},
        // {"input_lengths", as_GPU_tensor(input_tensors->at("input_lengths"), d_input_lengths_)},
    };

    if (input_tensors->find("bad_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("bad_words_list"), d_input_bad_words_, &allocator_);
        ft_input_tensors.insert(
            {"bad_words_list", as_GPU_tensor(input_tensors->at("bad_words_list"), d_input_bad_words_)});
    }

    if (input_tensors->find("stop_words_list") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("stop_words_list"), d_input_stop_words_, &allocator_);
        ft_input_tensors.insert(
            {"stop_words_list", as_GPU_tensor(input_tensors->at("stop_words_list"), d_input_stop_words_)});
    }

    if (input_tensors->count("request_prompt_embedding") && input_tensors->count("request_prompt_lengths")
        && input_tensors->count("request_prompt_type")) {

        move_tensor_H2D(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_lengths",
             as_GPU_tensor(input_tensors->at("request_prompt_lengths"), d_request_prompt_lengths_)});

        move_tensor_H2D(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_, &allocator_);
        ft_input_tensors.insert(
            {"request_prompt_embedding",
             as_GPU_tensor(input_tensors->at("request_prompt_embedding"), d_request_prompt_embedding_)});
    }

    if (input_tensors->find("top_p_decay") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_decay"), d_top_p_decay_, &allocator_);
        ft_input_tensors.insert({"top_p_decay", as_GPU_tensor(input_tensors->at("top_p_decay"), d_top_p_decay_)});
    }
    if (input_tensors->find("top_p_min") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_min"), d_top_p_min_, &allocator_);
        ft_input_tensors.insert({"top_p_min", as_GPU_tensor(input_tensors->at("top_p_min"), d_top_p_min_)});
    }
    if (input_tensors->find("top_p_reset_ids") != input_tensors->end()) {
        move_tensor_H2D(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_, &allocator_);
        ft_input_tensors.insert(
            {"top_p_reset_ids", as_GPU_tensor(input_tensors->at("top_p_reset_ids"), d_top_p_reset_ids_)});
    }

    for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
        if (t->first.find("input_ids") == std::string::npos  // && t->first.find("input_lengths") == std::string::npos
            && t->first.find("output_seq_len") == std::string::npos
            && t->first.find("prefix_soft_prompt_embedding") == std::string::npos
            && t->first.find("prefix_soft_prompt_lengths") == std::string::npos) {
            if (ft_input_tensors.count(t->first) == 0) {
                ft_input_tensors.insert({t->first, t->second.convertTritonTensorToFt()});
            }
        }
    }

    return ft_input_tensors;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
LlamaTritonModelInstance<T>::convert_outputs(const std::unordered_map<std::string, ft::Tensor>& output_tensors)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::unordered_map<std::string, triton::Tensor>* outputs_mapping =
        new std::unordered_map<std::string, triton::Tensor>();

    for (auto it = output_tensors.begin(); it != output_tensors.end(); it++) {
        outputs_mapping->insert({it->first, triton::Tensor::convertFtTensorToTriton(it->second)});
    }

    return std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(outputs_mapping);
}

template<typename T>
std::shared_ptr<std::vector<triton::Tensor>>
LlamaTritonModelInstance<T>::forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
}

template<typename T>
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
LlamaTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors)
{
    ft::FT_CHECK(false);
    return nullptr;
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
std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
LlamaTritonModelInstance<T>::forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors,
                                     ft::AbstractInstanceComm*                                        instance_comm)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // for (const auto& kv : *input_tensors) {
    //     TM_LOG_INFO("%s: %s", kv.first.c_str(), format_vector(kv.second.shape).c_str());
    // }

    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape.size() == 2,
                       "input_tensors->at(\"input_ids\").shape.size() == 2");
    FT_CHECK_WITH_INFO(input_tensors->at("input_lengths").shape.size() == 1,
                       "input_tensors->at(\"input_lengths\").shape.size() == 1");

    const uint32_t request_batch_size     = input_tensors->at("input_ids").shape[0];
    const uint32_t max_request_output_len = (size_t)*std::max_element(
        (int*)input_tensors->at("request_output_len").data,
        (int*)input_tensors->at("request_output_len").data + input_tensors->at("request_output_len").shape[0]);
    // const uint32_t total_output_len = max_request_output_len + input_tensors->at("input_ids").shape[1];
    const uint32_t beam_width =
        input_tensors->count("beam_width") ? (size_t)(*(uint*)input_tensors->at("beam_width").data) : 1;
    FT_CHECK_WITH_INFO(beam_width == 1, "Beam search is not implemented");

    std::unordered_map<std::string, ft::Tensor> ft_input_tensors = convert_inputs(input_tensors);

    const size_t max_input_len = input_tensors->at("input_ids").shape[1];
    const bool   is_return_logits =
        input_tensors->count("is_return_logits") && *(bool*)input_tensors->at("is_return_logits").data;

    const size_t vocab_size = instance_->llm->vocab_size();

    allocateBuffer(request_batch_size, max_input_len, beam_width, instance_->session_len, is_return_logits);

    std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
        {"output_ids",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_UINT32,
                    std::vector<size_t>{request_batch_size, beam_width, (size_t)instance_->session_len},
                    d_output_ids_}},
        {"sequence_length",
         ft::Tensor{ft::MEMORY_GPU,
                    ft::TYPE_UINT32,
                    std::vector<size_t>{request_batch_size, beam_width},
                    d_sequence_lengths_}}};

    if (input_tensors->count("is_return_log_probs") && *((bool*)input_tensors->at("is_return_log_probs").data)) {
        output_tensors.insert({"output_log_probs",
                               ft::Tensor{ft::MEMORY_GPU,
                                          ft::TYPE_FP32,
                                          std::vector<size_t>{request_batch_size, beam_width, max_request_output_len},
                                          d_output_log_probs_}});
        output_tensors.insert({"cum_log_probs",
                               ft::Tensor{ft::MEMORY_GPU,
                                          ft::TYPE_FP32,
                                          std::vector<size_t>{request_batch_size, beam_width},
                                          d_cum_log_probs_}});
    }

    if (is_return_logits) {
        output_tensors.insert(
            {"logits",
             {ft::MEMORY_GPU, ft::TYPE_FP32, {request_batch_size, max_input_len, vocab_size}, d_output_logits_}});
    }

    try {
        ft::Request::Callback callback;

        if (stream_cb_) {
            callback = [this](std::unordered_map<std::string, ft::Tensor>* outputs) {
                triton_stream_callback<T>(outputs, this);
            };
        }

        ft::check_cuda_error(cudaStreamSynchronize(allocator_->returnStream()));
        instance_->llm->forward(&output_tensors, &ft_input_tensors, {instance_comm, callback});
        // ! stream synced by the model before returning
    }
    catch (...) {
        h_exception_ = std::current_exception();
        output_tensors.insert({"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
    }

    return convert_outputs(output_tensors);
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
    d_output_ids_ =
        (int*)(allocator_->reMalloc(d_output_ids_, sizeof(int) * request_batch_size * beam_width * session_len, false));
    d_sequence_lengths_ =
        (int*)(allocator_->reMalloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_output_log_probs_ = (float*)(allocator_->reMalloc(
        d_output_log_probs_, sizeof(float) * request_batch_size * beam_width * session_len, false));
    d_cum_log_probs_ =
        (float*)(allocator_->reMalloc(d_cum_log_probs_, sizeof(float) * request_batch_size * beam_width, false));
    if (is_return_logits) {
        d_output_logits_ = (float*)allocator_->reMalloc(
            d_output_logits_, sizeof(float) * request_batch_size * max_input_len * instance_->llm->vocab_size(), false);
    }
}

template<typename T>
void LlamaTritonModelInstance<T>::freeBuffer()
{
    allocator_->free((void**)(&d_output_ids_));
    allocator_->free((void**)(&d_sequence_lengths_));
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_cum_log_probs_));
    std::free(h_total_output_lengths_);
}

template struct LlamaTritonModelInstance<float>;
template struct LlamaTritonModelInstance<half>;
