/*
 * Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/kernels/beam_search_topk_kernels.h"
#include "src/turbomind/layers/BaseLayer.h"
#include "src/turbomind/layers/DynamicDecodeBaseLayer.h"
#include "src/turbomind/layers/sampling_layers/TopPSamplingLayer.h"

namespace turbomind {

template<typename T>
class DynamicDecodeLayer: public BaseLayer {
protected:
    void allocateBuffer() override;
    void freeBuffer() override;
    void initialize();
    bool hasDiffRuntimeArgs(TensorMap* input_tensors);

    DynamicDecodeBaseLayer* online_beamsearch_decode_;
    DynamicDecodeBaseLayer* beamsearch_decode_;
    DynamicDecodeBaseLayer* topk_decode_;
    DynamicDecodeBaseLayer* topp_decode_;

    size_t          vocab_size_;
    size_t          vocab_size_padded_;
    cudaDeviceProp* cuda_device_prop_;

    // List of argument names which can have different values in runtime
    // and does not support a batched version of kernel in beam search.
    const std::vector<std::string> runtime_arg_names_ = {"beam_search_diversity_rate",
                                                         "temperature",
                                                         "len_penalty",
                                                         "repetition_penalty",
                                                         "presence_penalty",
                                                         "min_length"};

    bool has_diff_runtime_args_ = false;
    int* h_pinned_finished_sum_ = nullptr;

public:
    curandState_t* topk_curandstate_buf()
    {
        return static_cast<BaseSamplingLayer<T>*>(topk_decode_)->curandstate_buf();
    }
    curandState_t* topp_curandstate_buf()
    {
        return static_cast<BaseSamplingLayer<T>*>(topp_decode_)->curandstate_buf();
    }

    DynamicDecodeLayer(size_t           vocab_size,
                       size_t           vocab_size_padded,
                       int              end_id,
                       cudaStream_t     stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       cudaDeviceProp*  cuda_device_prop);

    ~DynamicDecodeLayer();
    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer);

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args);
    void forward(TensorMap* output_tensors, TensorMap* input_tensors);
    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors);
};

}  // namespace turbomind
