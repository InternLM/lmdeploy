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

    Tensor_<float> logits = args.at("logits");
    const size_t   bsz    = logits.shape(0);
    Tensor_<float> logits_buf{logits.shape(), kCPU};
    std::vector<int64_t> logits_shape = {bsz, logits.shape(1)};

    FT_CHECK(bsz == matchers_.size());

    const auto           bitmask_size = xgrammar::GetBitmaskSize(vocab_size_padded_);
    Tensor_<int32_t>     bitmask{{bsz, bitmask_size}, kCPU};
    std::vector<int64_t> bitmap_shape = {bsz, bitmask_size};

    DLTensor bitmask_dltensor{bitmask.data(), DLDevice{kDLCPU, 0}, static_cast<int32_t>(bitmap_shape.size()), xgrammar::GetBitmaskDLType(), bitmap_shape.data(), nullptr, 0};
    DLTensor logits_dltensor{logits_buf.data(), DLDevice{kDLCPU, 0}, static_cast<int32_t>(logits_shape.size()), DLDataType{kDLFloat, 32, 1}, logits_shape.data(), nullptr, 0};

    std::cerr << ">> logits shape: "<< logits.shape() << std::endl << ">> vocab_size_padded_: " << vocab_size_padded_ << std::endl;
    std::cerr << ">> logits device: "<< to_string(logits.device().type) << std::endl;

    // TODO: Speedup

    for (size_t i = 0; i < bsz; ++i) {
        const auto& matcher = matchers_[i];
        if (matcher) {
            matcher->FillNextTokenBitmask(&bitmask_dltensor, i, true);
        }
    }

    Copy(logits, logits_buf);

    xgrammar::ApplyTokenBitmaskInplaceCPU(&logits_dltensor, bitmask_dltensor, vocab_size_, std::nullopt);

    Copy(logits_buf, logits);
}


template class GuidedDecodeMaskLayer<float>;

}  // namespace turbomind
