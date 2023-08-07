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

#include "src/turbomind/layers/sampling_layers/BaseSamplingLayer.h"
#include "src/turbomind/utils/memory_utils.h"
#include "src/turbomind/windows/marco.h"

namespace turbomind {

template<typename T>
class TopKSamplingLayer: public BaseSamplingLayer<T> {
private:
    void runSampling(TensorMap* output_tensors, TensorMap* input_tensors) override;

    void freeBuffer() override;
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p) override;

    uint   runtime_max_top_k_ = 1;
    uint*  runtime_top_k_buf_ = nullptr;
    float* runtime_top_p_buf_ = nullptr;
    using BaseSamplingLayer<T>::vocab_size_;
    using BaseSamplingLayer<T>::vocab_size_padded_;

    using BaseSamplingLayer<T>::sampling_workspace_size_;
    using BaseSamplingLayer<T>::sampling_workspace_;
    using BaseSamplingLayer<T>::curandstate_buf_;
    using BaseSamplingLayer<T>::random_seeds_buf_;
    using BaseSamplingLayer<T>::skip_decode_buf_;
    using BaseSamplingLayer<T>::skip_decode_;
    using BaseSamplingLayer<T>::skip_any_;
    using BaseSamplingLayer<T>::runtime_logits_buf_;

    using BaseSamplingLayer<T>::stream_;
    using BaseSamplingLayer<T>::allocator_;
    using BaseSamplingLayer<T>::is_allocate_buffer_;

protected:
public:
    TopKSamplingLayer(size_t             max_batch_size,
                      size_t             vocab_size,
                      size_t             vocab_size_padded,
                      int                end_id,
                      size_t             top_k,
                      unsigned long long random_seed,
                      float              temperature,
                      float              len_penalty,
                      float              repetition_penalty,
                      cudaStream_t       stream,
                      cublasMMWrapper*   cublas_wrapper,
                      IAllocator*        allocator,
                      bool               is_free_buffer_after_forward);
    TopKSamplingLayer(TopKSamplingLayer<T> const& top_k_sampling_layer);
    ~TopKSamplingLayer();

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args) override;
};

}  // namespace turbomind
