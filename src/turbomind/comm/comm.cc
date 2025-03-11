// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

DeviceComm CreateNcclCommunicator(int n_ranks, int rank, HostComm h_comm);

DeviceComm CreateNativeCommunicator(int n_ranks, int rank, HostComm h_comm);

DeviceComm CreateDeviceCommunicator(const std::string& backend, int n_ranks, int rank, HostComm h_comm)
{
#if BUILD_MULTI_GPU && USE_NCCL
    if (backend == "nccl") {
        return CreateNcclCommunicator(n_ranks, rank, h_comm);
    }
#endif

#if BUILD_MULTI_GPU
    if (backend == "native") {
        return CreateNativeCommunicator(n_ranks, rank, h_comm);
    }
#endif

    FT_CHECK_WITH_INFO(0, fmtstr("Unknown communication backend: %s", backend.c_str()));
    return {};
}

}  // namespace turbomind::comm
