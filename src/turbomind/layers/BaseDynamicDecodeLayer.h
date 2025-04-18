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

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/request.h"

namespace turbomind {

class BaseDynamicDecodeLayer {
public:
    struct BaseParam {
        int                   max_batch_size;
        int                   vocab_size;
        int                   vocab_size_padded;
        cudaStream_t          stream;
        const cudaDeviceProp* device_prop;
    };

    virtual ~BaseDynamicDecodeLayer() = default;

    explicit BaseDynamicDecodeLayer(const BaseParam& param)
    {
        max_batch_size_    = param.max_batch_size;
        vocab_size_        = param.vocab_size;
        vocab_size_padded_ = param.vocab_size_padded;
        stream_            = param.stream;
        device_prop_       = param.device_prop;
    };

    virtual void Setup(const std::vector<const Request*>& rs, const TensorMap& args) = 0;

    virtual void Forward(TensorMap& args) = 0;

protected:
    int                   max_batch_size_;
    int                   vocab_size_;
    int                   vocab_size_padded_;
    cudaStream_t          stream_;
    const cudaDeviceProp* device_prop_;
};

}  // namespace turbomind
