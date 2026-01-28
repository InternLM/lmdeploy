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

    void CreateWeights(int index);

    TensorMap GetWeights(int index);

    void ProcessWeights(int index);

    void CreateEngine(int index);

    void Sleep(int index, int level);

    void WakeUp(int index, const std::vector<std::string>& tags);

    bool is_dummy_node() const noexcept;

    std::shared_ptr<ScheduleMetrics> GetScheduleMetrics(int index);

    std::unique_ptr<ModelRequest> CreateRequest();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
