/*
 * Copyright (c) 2025-2025, OpenMMLab.  All rights reserved.
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

#include "src/turbomind/layers/sampling_layers/GuidedDecodeMaskLayer.h"

namespace turbomind {

template<typename T>
GuidedDecodeMaskLayer<T>::GuidedDecodeMaskLayer(const BaseParam& param): BaseDynamicDecodeLayer{param}
{
}

template<typename T>
void GuidedDecodeMaskLayer<T>::Setup(const std::vector<const Request*>& rs, const TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    matchers_.clear();
    for (const auto& r : rs) {
        matchers_.push_back(r->matcher);
    }
}

template<typename T>
void GuidedDecodeMaskLayer<T>::Forward(TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
}

template<>
void GuidedDecodeMaskLayer<float>::Forward(TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const auto           bitmask_size = xgrammar::GetBitmaskSize(vocab_size_padded_);
    std::vector<int32_t> result(bitmask_size);
    std::vector<int64_t> shape = {bitmask_size};

    Tensor_<float> logits = args.at("logits");
    const size_t   bsz    = logits.shape(0);

    FT_CHECK(bsz == matchers_.size());

    for (size_t i = 0; i < bsz; ++i) {
        const auto& matcher = matchers_[i];

        if (matcher) {
            DLTensor bitmask_dltensor{result.data(),
                                      DLDevice{kDLCPU, 0},
                                      static_cast<int32_t>(shape.size()),
                                      xgrammar::GetBitmaskDLType(),
                                      shape.data(),
                                      nullptr,
                                      0};

            matcher->FillNextTokenBitmask(&bitmask_dltensor);

            DLTensor logits_dltensor{logits.slice(i).data<float>(),
                                     DLDevice{kDLCPU, 0},
                                     shape.size(),
                                     DLDataType{kDLFloat, 32, 1},
                                     shape.data(),
                                     nullptr,
                                     0};

            xgrammar::ApplyTokenBitmaskInplaceCPU(&logits_dltensor, bitmask_dltensor, vocab_size_padded_, std::nullopt);
        }
    }
}

template class GuidedDecodeMaskLayer<float>;

}  // namespace turbomind
