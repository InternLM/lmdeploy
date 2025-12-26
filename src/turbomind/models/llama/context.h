// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

namespace turbomind {

struct Communicators {
    comm::HostComm h_comm;
    comm::HostComm h_tp_group;
    comm::HostComm h_dp_group;

    comm::DeviceComm d_comm;
    int              d_tp_group;
    int              d_cp_group;
};

// Execution context for the model
struct Context {
    core::Stream                 core_stream;
    core::Allocator              allocator;
    cudaStream_t                 stream;
    std::unique_ptr<LlamaLinear> linear;
    cudaDeviceProp               device_prop;
    Communicators                comm;  // initialize later

    Context(int device_id):
        core_stream{core::Stream::create()},
        allocator{core::Allocator(core_stream, false)},
        stream{core_stream.handle()}
    {
        core::ContextGuard guard{core_stream};
        linear = std::make_unique<LlamaLinear>();
        check_cuda_error(cudaGetDeviceProperties(&device_prop, device_id));
    }
};

inline Allocator GetSymmAllocator(const comm::DeviceComm& comm)
{
    TM_CHECK(comm);
    return core::SimpleAllocator::Create(
        [&comm](auto size) {
            auto p = comm->Allocate(size);
            comm->Register(p, size);
            return p;
        },
        [&comm](void* p, auto size) {
            comm->Deregister(p);
            comm->Free(p);
        },
        kDEVICE);
}

}  // namespace turbomind
