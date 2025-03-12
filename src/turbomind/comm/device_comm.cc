// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

DeviceCommImpl::~DeviceCommImpl() = default;

DeviceComm CreateNcclCommunicator(int n_ranks, int rank, HostComm h_comm);

DeviceComm CreateCudaIpcCommunicator(int n_ranks, int rank, HostComm h_comm);

DeviceComm CreateDeviceCommunicator(const std::string& backend, int n_ranks, int rank, HostComm h_comm)
{
#if BUILD_MULTI_GPU && USE_NCCL
    if (backend == "nccl") {
        return CreateNcclCommunicator(n_ranks, rank, h_comm);
    }
#endif

#if BUILD_MULTI_GPU
    if (backend == "native" || backend == "cuda_ipc") {
        return CreateCudaIpcCommunicator(n_ranks, rank, h_comm);
    }
#endif

    FT_CHECK_WITH_INFO(0, fmtstr("Unknown communication backend: %s", backend.c_str()));
    return {};
}

}  // namespace turbomind::comm
