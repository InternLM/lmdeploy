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

#include "src/turbomind/layers/sampling_layers/BanBadWordsLayer.h"
#include "src/turbomind/kernels/ban_bad_words.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
void BanBadWordsLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void BanBadWordsLayer<T>::allocateBuffer(const size_t batch_size, const size_t list_size)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void BanBadWordsLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
BanBadWordsLayer<T>::~BanBadWordsLayer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void BanBadWordsLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    if (input_tensors->isExist("bad_words_list")) {
        FT_CHECK(input_tensors->at("logits").shape.size() == 3);
        const Tensor bad_words = input_tensors->at("bad_words_list");
        FT_CHECK(bad_words.shape.size() == 3);
        const size_t bad_words_len    = bad_words.shape[2];
        const size_t batch_size       = input_tensors->at("logits").shape[0];
        const int    step             = input_tensors->at("step").getVal<int>();
        const size_t local_batch_size = (size_t)input_tensors->at("local_batch_size").getVal<int>();

        invokeBanBadWords(input_tensors->at("logits").getPtr<T>(),
                          output_tensors->at("output_ids").getPtr<const int>(),
                          nullptr,
                          batch_size,
                          local_batch_size,
                          1,
                          bad_words.getPtr<const int>(),
                          false,
                          bad_words_len,
                          0,
                          args_.vocab_size_padded,
                          step,
                          stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void BanBadWordsLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class BanBadWordsLayer<float>;
}  // namespace turbomind
