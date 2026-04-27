// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/model_weight.h"

namespace turbomind {

/// Sentinel root for the weight tree.  Lives in TurboMind::Impl::weights_
/// and owns the CUDA stream + pool-backed allocator used during weight
/// loading.  Python creates a ModelWeight via _tm.create_module and
/// attaches it as the `text_model` child via add_child_raw.
class ModelRoot: public core::Module {
public:
    const char* type() const override
    {
        return "ModelRoot";
    }

    ModelRoot();
    ~ModelRoot() override;

    void prepare() override;

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    const core::Stream& stream() const
    {
        return stream_;
    }
    const core::Allocator& allocator() const
    {
        return alloca_;
    }

    /// Convenience accessor.  Nullptr before Python attaches via
    /// `add_child_raw('text_model', ...)`.
    ModelWeight* text_model_ptr() const
    {
        return text_model.get();
    }

#define MODEL_ROOT_CHILDREN(X) X(ModelWeight, text_model)

#define MODEL_ROOT_PARAMS(X)

    TM_MODULE_DECLARE(ModelRoot, MODEL_ROOT_CHILDREN, MODEL_ROOT_PARAMS)

private:
    core::Stream    stream_{};
    core::Allocator alloca_{};
};

}  // namespace turbomind
