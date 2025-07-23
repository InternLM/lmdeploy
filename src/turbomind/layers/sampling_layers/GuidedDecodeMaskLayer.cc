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
}

template<typename T>
void GuidedDecodeMaskLayer<T>::Forward(TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
}

template class GuidedDecodeMaskLayer<float>;

}  // namespace turbomind
