// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_root.h"
#include "src/turbomind/core/check.h"

namespace turbomind {

ModelRoot::ModelRoot()
{
    // CUDA device is already set by CudaDeviceGuard in TurboMind::CreateRoot.
    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, /*use_default_pool=*/true};
}

ModelRoot::~ModelRoot() = default;

void ModelRoot::prepare()
{
    TM_CHECK(text_model) << "ModelRoot::prepare: text_model not attached; did the spec "
                            "forget root.build()?";
    Module::prepare();
}

TM_MODULE_METHODS(ModelRoot, MODEL_ROOT_CHILDREN, MODEL_ROOT_PARAMS)

}  // namespace turbomind
