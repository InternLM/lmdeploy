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

#include <memory>
#include <vector>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"

#include "src/turbomind/core/tensor.h"

namespace turbomind {

class DynamicDecodeLayer {
public:
    DynamicDecodeLayer(DataType              data_type,
                       int                   max_batch_size,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       cudaStream_t          stream,
                       const cudaDeviceProp* device_prop,
                       int                   tp_rank);

    ~DynamicDecodeLayer();

    void Setup(const std::vector<const Request*>& rs, const TensorMap& args);

    void Forward(TensorMap& args);

private:
    int                                                  tp_rank_;
    std::vector<std::unique_ptr<BaseDynamicDecodeLayer>> layers_;
};

}  // namespace turbomind
