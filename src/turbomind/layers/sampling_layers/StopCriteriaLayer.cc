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
#include "src/turbomind/layers/sampling_layers/utils.h"
#include "src/turbomind/macro.h"

namespace turbomind {

template<typename T>
StopCriteriaLayer<T>::StopCriteriaLayer(const BaseParam& param): BaseDynamicDecodeLayer{param}
{
    stop_words_     = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kCPUpinned};
    stop_words_buf_ = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kDEVICE};
}

template<typename T>
void StopCriteriaLayer<T>::Setup(const std::vector<const Request*>& rs, const TensorMap&)
{
    stop_words_ten_ = {};
    init_stop_bad_words(&GenerationConfig::stop_ids,  //
                        "stop_words",
                        rs,
                        stop_words_.data(),
                        stop_words_buf_.data(),
                        stop_words_ten_);
}

template<typename T>
void StopCriteriaLayer<T>::Forward(TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const int batch_size = args.at("logits").shape(0);
    const int step       = *args.at("step").data<int>();

    if (auto& stop_words = stop_words_ten_) {
        TM_CHECK_EQ(stop_words.ndim(), 3);  // [batch, 2, len]
        size_t stop_words_len = stop_words.shape(2);
        invokeStopWordsCriterion(args.at("output_ids").data<int>(),
                                 nullptr,
                                 stop_words.data(),
                                 args.at("finished").data<bool>(),
                                 0,
                                 stop_words_len,
                                 batch_size,
                                 1,
                                 step,
                                 stream_);
        sync_check_cuda_error();
    }

    if (auto seq_lim_len = args.try_("sequence_limit_length")) {
        invokeLengthCriterion(args.at("finished").data<bool>(),  //
                              seq_lim_len->data<int>(),
                              batch_size,
                              1,
                              step,
                              stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;

}  // namespace turbomind
