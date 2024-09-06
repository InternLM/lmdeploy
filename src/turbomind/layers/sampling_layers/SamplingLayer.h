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

#include "src/turbomind/layers/DynamicDecodeBaseLayer.h"
#include "src/turbomind/macro.h"
#include <vector>

namespace turbomind {

template<typename T>
class SamplingLayer: public DynamicDecodeBaseLayer {
public:
    using DynamicDecodeBaseLayer::DynamicDecodeBaseLayer;
    using DynamicDecodeBaseLayer::args_;

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* params) override;

    void forward(TensorMap* output_tensors, TensorMap* input_tensors) override;

    ~SamplingLayer();

private:
    void allocateBuffer() override;

    void freeBuffer() override;

    void allocateBuffer(const size_t batch_size);

    // host buffer
    std::vector<int>   kept_n_;
    std::vector<int>   runtime_top_k_;
    std::vector<float> runtime_top_p_;
    std::vector<float> runtime_min_p_;
    int                max_topk_;
    int                min_topk_;
    float              min_topp_;
    float              max_minp_;

    // device buffer
    int*   runtime_top_k_buf_{};
    float* runtime_top_p_buf_{};
    float* runtime_min_p_buf_{};

    void*  topk_ws_{};
    size_t topk_ws_size_;

    void*  topp_ws_{};
    size_t topp_ws_size_;

    T*   logits_{};   // sorted logits
    int* indices_{};  // sorted indices
    int* kept_{};     // kept sample
};

}  // namespace turbomind
