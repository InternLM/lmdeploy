// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/host_comm.h"

namespace turbomind::comm {

HostCommImpl::~HostCommImpl() = default;

std::unique_ptr<HostGroupId> CreateThreadGroupId();

std::unique_ptr<HostGroupId> CreateGlooGroupId();

std::unique_ptr<HostGroupId> CreateHybridGroupId();

std::unique_ptr<HostGroupId> CreateHostGroupId(const std::string& backend)
{
#ifdef BUILD_MULTI_GPU
    if (backend == "hybrid") {
        return CreateHybridGroupId();
    }
    if (backend == "gloo") {
        return CreateGlooGroupId();
    }
#endif

    return CreateThreadGroupId();
}

}  // namespace turbomind::comm
