// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

std::unique_ptr<Comm> CreateNcclCommunicator(int rank, int world_size, std::shared_ptr<HostComm> host_comm);

std::unique_ptr<Comm> CreateNativeCommunicator(int rank, int world_size, std::shared_ptr<HostComm> host_comm);

std::unique_ptr<Comm>
CreateCommunicator(const std::string& backend, int rank, int n_ranks, std::shared_ptr<HostComm> host_comm)
{
#if BUILD_MULTI_GPU && USE_NCCL
    if (backend == "nccl") {
        return CreateNcclCommunicator(rank, n_ranks, host_comm);
    }
#endif

#if BUILD_MULTI_GPU
    if (backend == "native") {
        return CreateNativeCommunicator(rank, n_ranks, host_comm);
    }
#endif

    FT_CHECK_WITH_INFO(0, fmtstr("Unknown communication backend: %s", backend.c_str()));
    return nullptr;
}

}  // namespace turbomind::comm
