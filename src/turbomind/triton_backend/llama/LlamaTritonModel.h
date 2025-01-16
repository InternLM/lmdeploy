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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h

#pragma once

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/nccl_utils.h"
#include <cuda_fp16.h>

namespace turbomind {

template<typename T>
struct LlamaTritonModel: public AbstractTransformerModel {
    LlamaTritonModel(size_t                                 tensor_para_size,
                     size_t                                 pipeline_para_size,
                     int                                    enable_custom_all_reduce,
                     std::string                            model_dir,
                     std::string                            config,
                     std::function<std::shared_ptr<void>()> ffi_ctx_factory);

    ~LlamaTritonModel() override;

    std::unique_ptr<ModelRequest> createModelInstance(int deviceId) override;

    void createSharedWeights(int deviceId, int rank) override;

    std::unordered_map<std::string, Tensor> getParams(int deviceId, int rank) override;

    void processWeights(int deviceId, int rank) override;

    void createEngine(int                                                       device_id,
                      int                                                       rank,
                      std::pair<std::vector<NcclParam>, std::vector<NcclParam>> nccl_params,
                      std::shared_ptr<AbstractCustomComm>) override;

    void createCustomComms(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                           int                                               world_size) override;

    void handleMissingParams();

    std::string toString() override;
    int         getTensorParaSize() override;
    int         getPipelineParaSize() override;

private:
    std::unique_ptr<Engine<T>>
    createSharedModelInstance(int                                                       deviceId,
                              int                                                       rank,
                              std::pair<std::vector<NcclParam>, std::vector<NcclParam>> nccl_params,
                              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr);

    ModelParam     model_param_;
    AttentionParam attn_param_;
    MoeParam       moe_param_;
    LoraParam      lora_param_;
    EngineParam    engine_param_;
    size_t         tensor_para_size_;
    size_t         pipeline_para_size_;

    std::shared_ptr<SharedState> shared_state_;
    std::shared_ptr<Gateway>     gateway_;

    // Weights & engine instances for the ranks
    std::vector<std::shared_ptr<LlamaWeight<T>>> weights_;
    std::vector<std::shared_ptr<Engine<T>>>      engines_;

    bool is_fp16_;
    int  enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;
};

}  // namespace turbomind
