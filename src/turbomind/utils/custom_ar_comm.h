/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/turbomind/kernels/custom_ar_kernels.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace fastertransformer {

class AbstractCustomComm {
public:
    AbstractCustomComm()                                                             = default;
    virtual ~AbstractCustomComm()                                                    = default;
    virtual void customAllReduce(size_t elts, cudaStream_t stream)                   = 0;
    virtual void enableP2P(int ngpus)                                                = 0;
    virtual bool swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts) = 0;
    virtual void
    allocateAndExchangePeerAccessPointer(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms) = 0;
};

template<typename T>
class CustomAllReduceComm: public AbstractCustomComm {
public:
    CustomAllReduceComm(size_t rank_size, size_t rank);
    ~CustomAllReduceComm();

    void customAllReduce(size_t elts, cudaStream_t stream);

    void allocateAndExchangePeerAccessPointer(
        std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms) override;

    bool swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts) override;

    void enableP2P(int ngpus) override;

private:
    AllReduceParams<T>   param_;
    std::vector<Tensor>* output_tensor_;
    T*                   tmp_tensor_data_;
    size_t               rank_size_;
    size_t               rank_;
};

template<typename T>
void initCustomAllReduceComm(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                             int                                               enable_custom_all_reduce,
                             size_t                                            rank_size);

template<typename T>
struct CustomARCommTypeConverter {
    using Type = uint32_t;
};

template<>
struct CustomARCommTypeConverter<half> {
    using Type = uint16_t;
};

#ifdef ENABLE_BF16
template<>
struct CustomARCommTypeConverter<__nv_bfloat16> {
    using Type = __nv_bfloat16;
};
#endif

}  // namespace fastertransformer
