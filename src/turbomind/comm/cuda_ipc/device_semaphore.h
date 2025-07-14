// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/mscclpp.h"
#include "semaphore.cuh"

#include <cuda/atomic>

namespace turbomind::comm {

// Modified from
// https://github.com/microsoft/mscclpp/blob/591276f9d07d2df8e2a45a16738e27867e468ca3/include/mscclpp/semaphore_device.hpp#L40
struct DeviceSemaphore {

    __device__ void Load(mscclpp::D2DSemaphoreHandle* data)
    {
        outbound_sempahore_id         = *(uint64_t*)cvta_generic_to_global(data->outboundSemaphoreId);
        expected_inbound_semaphore_id = *(uint64_t*)cvta_generic_to_global(data->expectedInboundSemaphoreId);
        inbound_semaphore_id          = data->inboundSemaphoreId;
        remote_inbound_semaphore_id   = data->remoteInboundSemaphoreId;
    }

    __device__ void Save(mscclpp::D2DSemaphoreHandle* data)
    {
        *(uint64_t*)cvta_generic_to_global(data->outboundSemaphoreId)        = outbound_sempahore_id;
        *(uint64_t*)cvta_generic_to_global(data->expectedInboundSemaphoreId) = expected_inbound_semaphore_id;
    }

    __device__ void Wait()
    {
        Wait(cuda::memory_order_acquire);
    }

    __device__ void Wait(cuda::memory_order memory_order)
    {
        ++expected_inbound_semaphore_id;
        while (true) {
            uint64_t v{};
            if (memory_order == cuda::memory_order_relaxed) {
                asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(inbound_semaphore_id) : "memory");
            }
            else if (memory_order == cuda::memory_order_acquire) {
                asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(inbound_semaphore_id) : "memory");
            }
            if (v >= expected_inbound_semaphore_id) {
                break;
            }
        }
    }

    __device__ void Signal()
    {
        Signal(cuda::memory_order_release);
    }

    __device__ void Signal(cuda::memory_order memory_order)
    {
        auto v = ++outbound_sempahore_id;
        if (memory_order == cuda::memory_order_relaxed) {
            asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(remote_inbound_semaphore_id), "l"(v) : "memory");
        }
        else if (memory_order == cuda::memory_order_release) {
            asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(remote_inbound_semaphore_id), "l"(v) : "memory");
        }
    }

    __device__ void SignalAndWait(bool relaxed)
    {
        if (relaxed) {
            Signal(cuda::memory_order_relaxed);
            Wait(cuda::memory_order_relaxed);
        }
        else {
            Signal(cuda::memory_order_release);
            Wait(cuda::memory_order_acquire);
        }
    }

    uint64_t  outbound_sempahore_id;
    uint64_t  expected_inbound_semaphore_id;
    uint64_t* inbound_semaphore_id;
    uint64_t* remote_inbound_semaphore_id;
};

}  // namespace turbomind::comm
