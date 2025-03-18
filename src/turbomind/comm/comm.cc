// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

std::unique_ptr<GroupId> CreateNcclGroupId();

std::unique_ptr<GroupId> CreateNativeGroupId();

std::unique_ptr<GroupId> CreateGroupId(const std::string& backend)
{
#if BUILD_MULTI_GPU && USE_NCCL
    if (backend == "nccl") {
        return CreateNcclGroupId();
    }
#endif

#if BUILD_MULTI_GPU
    if (backend == "native") {
        return CreateNativeGroupId();
    }
#endif

    FT_CHECK_WITH_INFO(0, fmtstr("Unknown communication backend: %s", backend.c_str()));
    return nullptr;
}

}  // namespace turbomind::comm
