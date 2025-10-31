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

#include "src/turbomind/layers/sampling_layers/GuidedDecodeUpdateLayer.h"

namespace turbomind {

template<typename T>
GuidedDecodeUpdateLayer<T>::GuidedDecodeUpdateLayer(const BaseParam& param): BaseDynamicDecodeLayer{param}
{
}

template<typename T>
void GuidedDecodeUpdateLayer<T>::Setup(const std::vector<const Request*>& rs, const TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    matchers_.clear();
    for (const auto& r : rs) {
        matchers_.push_back(r->matcher);
    }
}

template<typename T>
void GuidedDecodeUpdateLayer<T>::Forward(TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    Tensor_<T>    logits     = args.at("logits");
    Tensor_<int>  output_ids = args.at("output_ids");
    const int     step       = *args.at("step").data<int>();
    const ssize_t bsz        = logits.shape(0);
    Tensor_<int>  output_ids_buf{{bsz}, kCPU};

    FT_CHECK(bsz == matchers_.size());
    Copy(output_ids.slice(step * bsz, bsz), output_ids_buf);

    for (size_t i = 0; i < bsz; ++i) {
        const auto& matcher = matchers_[i];
        if (matcher) {
            matcher->AcceptToken(output_ids_buf.data()[i]);
        }
    }
}

template class GuidedDecodeUpdateLayer<float>;
}  // namespace turbomind
