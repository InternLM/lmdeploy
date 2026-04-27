// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/engine/engine_config.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/utils/metrics.h"

namespace turbomind {

class TurboMind {
public:
    using FFICtxFactory = std::function<std::shared_ptr<void>()>;

    ~TurboMind();

    TurboMind(std::string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory);

    void          CreateContext(int index);
    core::Module* CreateRoot(int index);

    /// Returns the root `Module` for GPU `index`'s weight tree.
    core::Module* root(int index);

    /// Returns the Stream and Allocator for GPU `index`'s weight tree.
    std::pair<core::Stream, core::Allocator> weight_context(int index);

    void ProcessWeights(int index);

    void CreateEngine(int index);

    void Sleep(int index, int level);

    void WakeUp(int index, const std::vector<std::string>& tags);

    bool is_dummy_node() const noexcept;

    std::shared_ptr<ScheduleMetrics> GetScheduleMetrics(int index);

    std::unique_ptr<ModelRequest> CreateRequest();

    /// Attention TP rank for GPU *index*.
    int GetAttnTpRank(int index);

    /// MLP TP rank for GPU *index*.
    int GetMlpTpRank(int index);

    /// Model-level TP rank (rank within d_tp_group) for GPU *index*.
    int GetModelTpRank(int index);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
