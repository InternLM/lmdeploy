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

#include "src/fastertransformer/layers/BaseLayer.h"

namespace fastertransformer {

class DynamicDecodeBaseLayer: public BaseLayer {
protected:
    virtual void allocateBuffer() = 0;
    virtual void freeBuffer()     = 0;

public:
    DynamicDecodeBaseLayer(cudaStream_t     stream,
                           cublasMMWrapper* cublas_wrapper,
                           IAllocator*      allocator,
                           bool             is_free_buffer_after_forward,
                           cudaDeviceProp*  cuda_device_prop):
        BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop){};
    ~DynamicDecodeBaseLayer() = default;
    DynamicDecodeBaseLayer(DynamicDecodeBaseLayer const& dynamic_decode_layer): BaseLayer(dynamic_decode_layer){};

    virtual void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args) = 0;
    virtual void forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors)             = 0;
    virtual void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                         const std::unordered_map<std::string, Tensor>* input_tensors)            = 0;
    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors)                     = 0;
};

}  // namespace fastertransformer
