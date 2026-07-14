// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

__inline__ __device__ int sem_fetch(int* lock, bool pred)
{
    int state{};
    if (pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#else
        asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#endif
    }
    return state;
}

__inline__ __device__ void sem_wait(int* lock, int status, bool pred)
{
    int state = 0;
    while (__syncthreads_and(state != status)) {
        state = sem_fetch(lock, pred);
    }

    __syncthreads();  // memory fence
}

__inline__ __device__ void sem_wait_many(int* lock, int count, bool pred)
{
    int state = 0;
    while (__syncthreads_count(state) != count) {
        state = sem_fetch(lock, pred);
    }

    __syncthreads();  // memory fence
}

__inline__ __device__ void sem_post(int* lock, int status, bool pred)
{
    __syncthreads();  // memory fence

    if (pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#else
        asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
}

}  // namespace turbomind
