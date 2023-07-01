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

#include <assert.h>

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"

namespace turbomind {

class BaseLayer {
public:
    BaseLayer(cudaStream_t     stream,
              cublasMMWrapper* cublas_wrapper,
              IAllocator*      allocator,
              bool             is_free_buffer_after_forward,
              cudaDeviceProp*  cuda_device_prop = nullptr,
              bool             sparse           = false):
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        allocator_(allocator),
        cuda_device_prop_(cuda_device_prop),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        sparse_(sparse){};
    virtual ~BaseLayer() = default;

    virtual cudaStream_t getStream()
    {
        return stream_;
    }

    virtual void setStream(cudaStream_t stream)
    {
        stream_ = stream;
    }

protected:
    virtual void allocateBuffer() = 0;
    virtual void freeBuffer()     = 0;

    // device environments
    cudaStream_t     stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator*      allocator_;
    cudaDeviceProp*  cuda_device_prop_ = nullptr;

    bool is_free_buffer_after_forward_;
    bool is_allocate_buffer_ = false;  // TODO (bhsueh) to be deprecated
    bool sparse_;
};

}  // namespace turbomind
