// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

namespace turbomind {

struct Communicators {
    comm::HostComm h_comm;
    comm::HostComm h_tp_group;
    comm::HostComm h_dp_group;
    comm::HostComm h_tp_mem_group;

    comm::DeviceComm d_comm;
    int              d_tp_group;
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
        stream{core_stream.handle()},
        linear{std::make_unique<LlamaLinear>(stream)}
    {
        check_cuda_error(cudaGetDeviceProperties(&device_prop, device_id));
    }
};

}  // namespace turbomind
