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
#include "src/turbomind/kernels/apply_token_bitmask_inplace_cuda.h"

namespace turbomind {

template<typename T>
GuidedDecodeMaskLayer<T>::GuidedDecodeMaskLayer(const BaseParam& param): BaseDynamicDecodeLayer{param}
{
    const auto bitmask_size = xgrammar::GetBitmaskSize(vocab_size_padded_);
    bitmask_buf_            = {{max_batch_size_, bitmask_size}, kCPU};
    bitmask_                = {{max_batch_size_, bitmask_size}, kDEVICE};
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
    const ssize_t  bsz    = logits.shape(0);

    TM_CHECK(bsz == matchers_.size());

    const auto           bitmask_size  = bitmask_buf_.shape(1);
    std::vector<int64_t> bitmask_shape = {bsz, bitmask_size};

    DLTensor bitmask_dltensor{bitmask_buf_.data(),
                              DLDevice{kDLCPU, 0},
                              bitmask_buf_.ndim(),
                              xgrammar::GetBitmaskDLType(),
                              bitmask_shape.data(),
                              nullptr,
                              0};
    bool     need_apply = false;
    for (size_t i = 0; i < bsz; ++i) {
        const auto& matcher = matchers_[i];
        if (matcher) {
            matcher->FillNextTokenBitmask(&bitmask_dltensor, i);
            need_apply = true;
        }
    }

    if (need_apply) {
        Copy(bitmask_buf_, bitmask_);
        ApplyTokenBitmaskInplace(logits, bitmask_.slice(0, bsz));
    }
}

template class GuidedDecodeMaskLayer<float>;

}  // namespace turbomind
