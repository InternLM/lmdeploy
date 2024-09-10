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

#include <string>
#include <unordered_map>

#include "src/turbomind/layers/BaseLayer.h"

namespace turbomind {

struct DynamicDecodeCommonArgs {
    size_t vocab_size;
    size_t vocab_size_padded;
};

class DynamicDecodeBaseLayer: public BaseLayer {
protected:
    DynamicDecodeCommonArgs args_;

    virtual void allocateBuffer() = 0;
    virtual void freeBuffer()     = 0;

public:
    DynamicDecodeBaseLayer(cudaStream_t            stream,
                           IAllocator*             allocator,
                           bool                    is_free_buffer_after_forward,
                           DynamicDecodeCommonArgs args):
        BaseLayer(stream, nullptr, allocator, is_free_buffer_after_forward, nullptr), args_(args){};
    ~DynamicDecodeBaseLayer() = default;

    virtual void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args) = 0;

    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors) = 0;
};

}  // namespace turbomind
