/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/layers/sampling_layers/StopCriteriaLayer.h"
#include "src/turbomind/kernels/stop_criteria_kernels.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
void StopCriteriaLayer<T>::allocateBuffer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    h_pinned_finished_sum_ = (int*)allocator_->reMalloc(h_pinned_finished_sum_, sizeof(int), true, true);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void StopCriteriaLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    allocator_->free((void**)(&h_pinned_finished_sum_), true);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
StopCriteriaLayer<T>::~StopCriteriaLayer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void StopCriteriaLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const size_t batch_size = input_tensors->at("logits").shape[0];
    const int    step       = input_tensors->at("step").getVal<int>();

    if (input_tensors->isExist("stop_words_list")) {
        const Tensor stop_words_list = input_tensors->at("stop_words_list");
        FT_CHECK(stop_words_list.shape.size() == 3);  // [batch, 2, len]
        size_t stop_words_len = stop_words_list.shape[2];
        invokeStopWordsCriterion(output_tensors->at("output_ids").getPtr<const int>(),
                                 nullptr,
                                 stop_words_list.getPtr<const int>(),
                                 output_tensors->at("finished").getPtr<bool>(),
                                 0,
                                 stop_words_len,
                                 batch_size,
                                 1,
                                 step,
                                 stream_);

        sync_check_cuda_error();
    }

    if (input_tensors->isExist("sequence_limit_length")) {
        invokeLengthCriterion(output_tensors->at("finished").getPtr<bool>(),
                              output_tensors->getPtr<bool>("should_stop", nullptr),
                              h_pinned_finished_sum_,
                              input_tensors->at("sequence_limit_length").getPtr<const uint32_t>(),
                              batch_size,
                              1,
                              step,
                              stream_);

        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void StopCriteriaLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;

}  // namespace turbomind
