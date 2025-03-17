// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/host_comm.h"

namespace turbomind::comm {

HostCommImpl::~HostCommImpl() = default;

std::unique_ptr<HostGroupId> CreateThreadGroupId();

std::unique_ptr<HostGroupId> CreateHostGroupId(const std::string& backend)
{
    return CreateThreadGroupId();
}

}  // namespace turbomind::comm
