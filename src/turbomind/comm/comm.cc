

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

std::unique_ptr<GroupId> CreateNcclGroupId();

std::unique_ptr<GroupId> CreateCustomGroupId();

std::unique_ptr<GroupId> CreateGroupId(const std::string& backend)
{
    if (backend == "nccl") {
        return CreateNcclGroupId();
    }
    else if (backend == "custom") {
        return CreateCustomGroupId();
    }
    FT_CHECK_WITH_INFO(0, fmtstr("Unknown communication backend: %s", backend.c_str()));
    return nullptr;
}

}  // namespace turbomind::comm