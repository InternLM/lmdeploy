// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/utils/metrics.h"

namespace turbomind {

class TurboMind {
public:
    using FFICtxFactory = std::function<std::shared_ptr<void>()>;

    ~TurboMind();

    TurboMind(std::string model_dir, std::string config, FFICtxFactory ffi_ctx_factory);

    void CreateWeights(int device_id, int rank);

    TensorMap GetWeights(int device_id, int rank);

    void ProcessWeights(int device_id, int rank);

    void CreateEngine(int device_id, int rank);

    void Sleep(int device_id, int level);

    void WakeUp(int device_id, const std::vector<std::string>& tags, int rank);

    bool is_dummy_node() const noexcept;

    std::shared_ptr<ScheduleMetrics> GetScheduleMetrics(int device_id, int rank);

    std::unique_ptr<ModelRequest> CreateRequest();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind