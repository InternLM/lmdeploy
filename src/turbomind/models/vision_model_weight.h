// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/module.h"

namespace turbomind {

class VisionModel;
struct EngineParam;
struct Context;

/// Polymorphic root for the vision sub-graph.
///
/// Sits next to ``ModelWeight`` under ``ModelRoot`` (as the
/// ``vision_model`` child). Concrete subclasses (one per VLM family)
/// declare the per-family child layout via ``TM_MODULE_DECLARE`` and
/// own their own scalars + config
class VisionModelWeight: public core::Module {
public:
    const char* type() const override
    {
        return "VisionModelWeight";
    }

    VisionModelWeight()           = default;
    ~VisionModelWeight() override = default;

    /// Construct the runtime peer for this weight tree. Each concrete
    /// subclass returns its matching ``VisionModel`` implementation,
    /// which is how the engine binds weights to runtime without arch
    /// strings or registry indirection — the weight already carries the
    /// family identity (see ``type()`` and ``ModuleRegistry`` dispatch),
    /// so virtual-method dispatch is the natural extension.
    virtual std::unique_ptr<VisionModel>
    make_model(const EngineParam& engine, const Context& ctx, int phases) const = 0;
};

}  // namespace turbomind
