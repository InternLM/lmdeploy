/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/triton_backend/transformer_triton_backend.hpp

#pragma once

#include <functional>
#include <memory>
#include <vector>

#ifdef __linux__
#include <sys/time.h>
#endif

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/nccl_utils.h"

#include "src/turbomind/engine/model_request.h"

namespace turbomind {

using triton_stream_cb_t = std::function<void(std::shared_ptr<std::unordered_map<std::string, Tensor>>, void*)>;

struct AbstractTransformerModel;
struct AbstractTransformerModelInstance;

struct AbstractTransformerModelInstance {
    virtual ~AbstractTransformerModelInstance() = default;

    virtual std::shared_ptr<std::unordered_map<std::string, Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, Tensor>> input_tensors) = 0;

    void registerCallback(triton_stream_cb_t cb, void* ctx)
    {
        stream_cb_  = cb;
        stream_ctx_ = ctx;
    }

    void unRegisterCallback()
    {
        stream_cb_  = nullptr;
        stream_ctx_ = nullptr;
    }

    triton_stream_cb_t stream_cb_  = nullptr;
    void*              stream_ctx_ = nullptr;
};

struct AbstractTransformerModel {

    virtual ~AbstractTransformerModel() = default;

    virtual std::pair<std::vector<NcclParam>, std::vector<NcclParam>>
    createNcclParams(const int node_id, const int device_id_start = 0, const bool multi_node = false);

    virtual void destroyNcclParams(std::pair<std::vector<NcclParam>, std::vector<NcclParam>> params);

    virtual void createCustomComms(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                                   int                                               world_size) = 0;

    virtual std::unique_ptr<ModelRequest> createModelInstance(int deviceId) = 0;

    virtual void createSharedWeights(int deviceId, int rank) = 0;

    virtual std::unordered_map<std::string, Tensor> getParams(int deviceId, int rank) = 0;

    virtual void processWeights(int deviceId, int rank) = 0;

    virtual void createEngine(int                                                       device_id,
                              int                                                       rank,
                              std::pair<std::vector<NcclParam>, std::vector<NcclParam>> nccl_params,
                              std::shared_ptr<AbstractCustomComm>) = 0;

    virtual std::string toString()            = 0;
    virtual int         getTensorParaSize()   = 0;
    virtual int         getPipelineParaSize() = 0;
};

}  // namespace turbomind
