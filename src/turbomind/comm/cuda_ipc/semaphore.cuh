#pragma once

#include <vector>

#include "src/turbomind/kernels/core/array.h"

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.h"

namespace turbomind::comm {

template<class T>
__device__ T* cvta_generic_to_global(T* p)
{
    uintptr_t ret;
    asm("cvta.to.global.u64 %0, %1;" : "=l"(ret) : "l"(p));
    return reinterpret_cast<T*>(ret);
}

struct SystemSemaphore {

    using T = uint64_t;

    T* outbound_;
    T* inbound_;
    T  expected_;
    // T* mc_ptr_;

    bool uc_predicate_;
    // bool mc_predicate_;

    __device__ SystemSemaphore(const SystemSemaphoreInfo* info, int ranks, int channel, int thread_idx)
    {
        uc_predicate_ = thread_idx < ranks;
        // mc_predicate_ = thread_idx == 0;

        if (uc_predicate_) {
            int index = channel * kMaxRanks + thread_idx;
            inbound_  = info->inbound[index];
            outbound_ = info->outbound[index];
            expected_ = info->expected[index];
            // mc_ptr_   = info->mc_ptr[channel];
        }
    }

    __device__ void Update(SystemSemaphoreInfo* info, int ranks, int channel, int thread_idx)
    {
        if (uc_predicate_) {
            info->expected[channel * kMaxRanks + thread_idx] = expected_;
        }
    }

    __device__ void Signal(bool relaxed)
    {
        if (uc_predicate_) {
            if (relaxed) {
                asm volatile("atom.relaxed.sys.global.add.u64 _, [%0], %1;" ::"l"(outbound_), "n"(1) : "memory");
            }
            else {
                asm volatile("atom.release.sys.global.add.u64 _, [%0], %1;" ::"l"(outbound_), "n"(1) : "memory");
            }
        }
    }

    __device__ void Wait(bool relaxed)
    {
        if (uc_predicate_) {
            ++expected_;
            T x{};
            do {
                if (relaxed) {
                    asm volatile("ld.relaxed.sys.global.u64 %0,[%1];" : "=l"(x) : "l"(inbound_) : "memory");
                }
                else {
                    asm volatile("ld.acquire.sys.global.u64 %0,[%1];" : "=l"(x) : "l"(inbound_) : "memory");
                }
            } while (x < expected_);
        }
    }

    //     __device__ void SignalMulticast(bool relaxed)
    //     {
    // #if TURBOMIND_ARCH_SM90
    //         if (mc_predicate_) {
    //             if (relaxed) {
    //                 asm volatile("multimem.red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(mc_ptr_), "n"(1) :
    //                 "memory");
    //             }
    //             else {
    //                 asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;" ::"l"(mc_ptr_), "n"(1) :
    //                 "memory");
    //             }
    //             asm volatile("fence.proxy.alias;" ::: "memory");
    //         }
    // #endif
    //     }
};

}  // namespace turbomind::comm
