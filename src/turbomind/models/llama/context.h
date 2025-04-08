// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/allocator.h"

namespace turbomind {

struct Communicators {
    comm::HostComm h_comm;
    comm::HostComm h_tp_group;
    comm::HostComm h_dp_group;

    comm::DeviceComm d_comm;
    int              d_tp_group;
};

// Execution context for the model
struct Context {
    core::Stream                                    core_stream;
    core::Allocator                                 core_allocator;
    cudaStream_t                                    stream;
    std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator;
    std::unique_ptr<LlamaLinear>                    linear;
    Communicators                                   comm;
    cudaDeviceProp                                  cuda_device_prop;

    Context(DataType data_type, int device_id)
    {
        core_stream    = core::Stream::create();
        core_allocator = core::Allocator(core_stream, false);

        stream = core_stream.handle();

        allocator = std::make_unique<Allocator<AllocatorType::CUDA>>(device_id, false);
        allocator->setStream(stream);

        linear = std::make_unique<LlamaLinear>(stream);

        check_cuda_error(cudaGetDeviceProperties(&cuda_device_prop, device_id));
    }

    ~Context()
    {
        linear.reset();
        allocator.reset();

        // `comm` destroyed by infer threads collectively
    }
};

}  // namespace turbomind
