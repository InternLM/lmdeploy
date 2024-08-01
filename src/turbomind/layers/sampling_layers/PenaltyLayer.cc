/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/layers/sampling_layers/PenaltyLayer.h"
#include "src/turbomind/kernels/sampling_penalty_kernels.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
void init_host_buffer(TensorMap* runtime_args, const std::string& key, size_t size, T* dst, T default_value)
{
    const Tensor src      = runtime_args->isExist(key) ? runtime_args->at(key) : Tensor();
    const size_t src_size = src.size();
    if (src_size > 0) {
        std::copy_n(src.getPtr<T>(), src_size, dst);
    }
    else {
        std::fill_n(dst, size, default_value);
    }
}

template<typename T>
void PenaltyLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void PenaltyLayer<T>::allocateBuffer(const size_t batch_size)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    repetition_penalty_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false));
    min_lengths_buf_ = reinterpret_cast<int*>(allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false));
    temperature_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false));

    repetition_penalty_.resize(batch_size);
    min_lengths_.resize(batch_size);
    context_length_.resize(batch_size);
    temperature_.resize(batch_size);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void PenaltyLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    repetition_penalty_ = {};
    min_lengths_        = {};
    context_length_     = {};
    temperature_        = {};

    allocator_->free((void**)&repetition_penalty_workspace_);
    allocator_->free((void**)&repetition_penalty_buf_);
    allocator_->free((void**)&min_lengths_buf_);
    allocator_->free((void**)&temperature_buf_);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
PenaltyLayer<T>::~PenaltyLayer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void PenaltyLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const int batch_size       = output_tensors->at("output_ids").shape[1];
    const int local_batch_size = input_tensors->at("logits").shape[0];
    const int step             = input_tensors->at("step").getVal<int>();
    const int max_input_length = input_tensors->at("max_input_length").getVal<int>();
    T*        logits           = input_tensors->at("logits").getPtr<T>();

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

    // repetition penalty
    if (step > 1 && repetition_penalty_type_ != RepetitionPenaltyType::None) {
        float default_value = getDefaultPenaltyValue(repetition_penalty_type_);
        if (!ALL_OF(repetition_penalty_.begin(), local_batch_size, float, default_value)) {
            repetition_penalty_workspace_ = reinterpret_cast<int*>(allocator_->reMalloc(
                repetition_penalty_workspace_, batch_size * step * (sizeof(int) + sizeof(float)), false));
            invokeBatchApplyRepetitionPenalty(
                logits,
                repetition_penalty_buf_,
                repetition_penalty_workspace_,
                output_tensors->at("output_ids").getPtr<int>(),
                batch_size,
                local_batch_size,
                args_.vocab_size_padded,
                input_tensors->at("input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {}, nullptr}).getPtr<int>(),
                max_input_length,
                step,
                repetition_penalty_type_,
                stream_);
            sync_check_cuda_error();
        }
    }

    // min length
    {
        const int        num_generated_tokens = step - max_input_length;
        const int*       min_lengths          = min_lengths_.data();
        std::vector<int> index(batch_size);
        std::iota(index.begin(), index.end(), 0);
        const bool invoke_min_length_penalty = std::any_of(index.begin(), index.end(), [&](int i) {
            return min_lengths[i] > context_length_[i] + num_generated_tokens;
        });
        if (invoke_min_length_penalty) {
            FT_CHECK_WITH_INFO(input_tensors->isExist("end_id"), "Need end_id to apply min length penlaty");
            invokeMinLengthPenalty(logits,
                                   min_lengths_buf_,
                                   input_tensors->getPtr<const int>("end_id"),
                                   output_tensors->getPtr<const int>("sequence_length"),
                                   max_input_length,
                                   local_batch_size,
                                   args_.vocab_size_padded,
                                   stream_);
            sync_check_cuda_error();
        }
    }

    // temperature
    {
        if (!ALL_OF(temperature_.begin(), local_batch_size, float, 1.f)) {
            invokeBatchApplyTemperaturePenalty(logits,
                                               (T*)nullptr,
                                               temperature_buf_,
                                               local_batch_size,
                                               args_.vocab_size,
                                               args_.vocab_size_padded,
                                               stream_);
            sync_check_cuda_error();
        }
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void PenaltyLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    allocateBuffer(batch_size);

    if (runtime_args->isExist("repetition_penalty")) {
        init_host_buffer(runtime_args, "repetition_penalty", batch_size, repetition_penalty_.data(), 1.f);
        repetition_penalty_type_ = RepetitionPenaltyType::Multiplicative;
    }

    init_host_buffer(runtime_args, "min_length", batch_size, min_lengths_.data(), 0);
    init_host_buffer(runtime_args, "context_length", batch_size, context_length_.data(), 0);
    init_host_buffer(runtime_args, "temperature", batch_size, temperature_.data(), 1.f);

    std::transform(
        min_lengths_.begin(), min_lengths_.end(), context_length_.begin(), min_lengths_.begin(), std::plus<int>());

    cudaAutoCpy(temperature_buf_, temperature_.data(), batch_size, stream_);
    cudaAutoCpy(repetition_penalty_buf_, repetition_penalty_.data(), batch_size, stream_);
    cudaAutoCpy(min_lengths_buf_, min_lengths_.data(), batch_size, stream_);

    sync_check_cuda_error();

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class PenaltyLayer<float>;

}  // namespace turbomind
