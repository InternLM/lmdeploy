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

#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/layers/DynamicDecodeBaseLayer.h"

namespace turbomind {

template<typename T>
class BaseBeamSearchLayer: public DynamicDecodeBaseLayer {
private:
    void freeBuffer();

protected:
    // meta data
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t topk_softmax_workspace_size_;
    void*  topk_softmax_workspace_ = nullptr;

    virtual void allocateBuffer()                                                   = 0;
    virtual void allocateBuffer(size_t batch_size, size_t beam_width)               = 0;
    virtual void invokeSoftMax(TensorMap* output_tensors, TensorMap* input_tensors) = 0;

public:
    BaseBeamSearchLayer(size_t           max_batch_size,
                        size_t           head_num,
                        size_t           size_per_head,
                        size_t           beam_width,
                        size_t           vocab_size,
                        size_t           vocab_size_padded,
                        int              end_id,
                        float            diversity_rate,
                        float            temperature,
                        float            len_penalty,
                        float            repetition_penalty,
                        cudaStream_t     stream,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator*      allocator,
                        bool             is_free_buffer_after_forward);

    BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer);

    ~BaseBeamSearchLayer();

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args) override;
    void forward(std::vector<turbomind::Tensor>*       output_tensors,
                 const std::vector<turbomind::Tensor>* input_tensors) override;
    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors) override;
    void forward(TensorMap* output_tensors, TensorMap* input_tensors) override;
};

void update_indir_cache_kernelLauncher(int*         tgt_indir_cache,
                                       const int*   src_indir_cache,
                                       const int*   beam_ids,
                                       const bool*  finished,
                                       int          batch_dim,
                                       int          beam_width,
                                       int          max_seq_len,
                                       int          ite,
                                       cudaStream_t stream);

}  // namespace turbomind
