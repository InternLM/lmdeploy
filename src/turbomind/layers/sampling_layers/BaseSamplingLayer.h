/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <curand_kernel.h>

#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/layers/DynamicDecodeBaseLayer.h"

namespace turbomind {

template<typename T>
class BaseSamplingLayer: public DynamicDecodeBaseLayer {
private:
    bool isValidBatchSize(size_t batch_size);

protected:
    size_t vocab_size_;
    size_t vocab_size_padded_;

    int* repetition_penalty_workspace_;

    size_t sampling_workspace_size_;
    void*  sampling_workspace_ = nullptr;

    float* temperature_buf_        = nullptr;
    float* repetition_penalty_buf_ = nullptr;
    int*   min_lengths_buf_        = nullptr;
    bool*  skip_decode_buf_        = nullptr;
    T*     runtime_logits_buf_     = nullptr;
    int*   prompt_lengths_buf_     = nullptr;

    float* temperature_        = nullptr;
    float* repetition_penalty_ = nullptr;
    int*   min_lengths_        = nullptr;
    bool*  skip_decode_        = nullptr;
    bool   skip_any_           = false;
    int*   context_length_     = nullptr;

    RepetitionPenaltyType repetition_penalty_type_ = RepetitionPenaltyType::None;

    virtual void runSampling(TensorMap* output_tensors, TensorMap* input_tensors) = 0;

    virtual void freeBuffer();
    virtual void allocateBuffer() = 0;
    virtual void allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p);

public:
    BaseSamplingLayer(size_t             max_batch_size,
                      size_t             vocab_size,
                      size_t             vocab_size_padded,
                      int                end_id,
                      size_t             top_k,
                      float              top_p,
                      unsigned long long random_seed,  // TODO(bhsueh) delete
                      float              temperature,
                      float              len_penalty,
                      float              repetition_penalty,
                      cudaStream_t       stream,
                      cublasMMWrapper*   cublas_wrapper,
                      IAllocator*        allocator,
                      bool               is_free_buffer_after_forward,
                      cudaDeviceProp*    cuda_device_prop);

    BaseSamplingLayer(BaseSamplingLayer const& sampling_layer);

    ~BaseSamplingLayer();

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args) override;
    void forward(std::vector<turbomind::Tensor>*       output_tensors,
                 const std::vector<turbomind::Tensor>* input_tensors) override;
    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors) override;
    void forward(TensorMap* output_tensors, TensorMap* input_tensors) override;
};

}  // namespace turbomind
