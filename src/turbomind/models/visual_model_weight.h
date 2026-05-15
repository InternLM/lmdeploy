// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/module.h"

namespace turbomind {

class VisualModel;
struct EngineParam;
struct Context;

/// Polymorphic root for the visual sub-graph.
///
/// Sits next to ``ModelWeight`` under ``ModelRoot`` (as the
/// ``visual_model`` child). Concrete subclasses (one per VLM family)
/// declare the per-family child layout via ``TM_MODULE_DECLARE`` and
/// own their own scalars + config
class VisualModelWeight: public core::Module {
public:
    const char* type() const override
    {
        return "VisualModelWeight";
    }

    VisualModelWeight()           = default;
    ~VisualModelWeight() override = default;

    /// Construct the runtime peer for this weight tree. Each concrete
    /// subclass returns its matching ``VisualModel`` implementation,
    /// which is how the engine binds weights to runtime without arch
    /// strings or registry indirection — the weight already carries the
    /// family identity (see ``type()`` and ``ModuleRegistry`` dispatch),
    /// so virtual-method dispatch is the natural extension.
    virtual std::unique_ptr<VisualModel>
    make_model(const EngineParam& engine, const Context& ctx, int phases) const = 0;
};

}  // namespace turbomind
