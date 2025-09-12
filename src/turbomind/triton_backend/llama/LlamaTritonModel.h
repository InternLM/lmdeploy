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

#include <cuda_fp16.h>
#include <string>
#include <unordered_map>

#include <xgrammar/xgrammar.h>

#include "src/turbomind/comm/device_comm.h"

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/model_request.h"

#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class LlamaTritonModel {
public:
    LlamaTritonModel(std::string model_dir, std::string config, std::function<std::shared_ptr<void>()> ffi_ctx_factory);

    ~LlamaTritonModel();

    std::unique_ptr<ModelRequest> createModelInstance(int deviceId);

    void createSharedWeights(int deviceId, int rank);

    TensorMap getParams(int deviceId, int rank);

    void processWeights(int deviceId, int rank);

    void createEngine(int device_id, int rank);

    void sleep(int device_id, int level);

    void wakeup(int device_id, const std::vector<std::string>& tags, int rank);

    void setGrammar(const xgrammar::CompiledGrammar& grammar);

    std::string toString();

    int getTensorParaSize();
    int getPipelineParaSize();

    ScheduleMetrics getScheduleMetrics(int device_id, int rank);

private:
    void handleMissingParams();

    Communicators createCommSplits(int rank);

private:
    DataType       dtype_;
    ModelParam     model_param_;
    AttentionParam attn_param_;
    MoeParam       moe_param_;
    LoraParam      lora_param_;
    EngineParam    engine_param_;
    size_t         comm_size_;

    std::vector<EngineParam> engine_params_;

    std::string communicator_;  // communicator backend

    std::vector<std::unique_ptr<comm::HostGroupId>> group_ids_;

    std::shared_ptr<Gateway>               gateway_;
    std::function<std::shared_ptr<void>()> ffi_ctx_factory_;

    // Weights & engine instances for the ranks
    std::vector<std::shared_ptr<LlamaWeight>> weights_;
    std::vector<std::shared_ptr<Engine>>      engines_;

    std::vector<std::shared_ptr<Context>> contexts_;

    bool is_fp16_;

    std::string model_name_;
    std::string model_dir_;
    std::shared_ptr<xgrammar::CompiledGrammar> grammar_;
};

}  // namespace turbomind
