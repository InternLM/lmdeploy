// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/multimem.cuh"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

__global__ void Barrier_V2(SystemSemaphoreInfo* semaphores, int ranks)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::Barrier(int group, cudaStream_t stream)
{
    const int ranks = n_ranks(group);
    Barrier_V2<<<1, ranks, 0, stream>>>(groups_.at(group).semaphore.handle(), ranks);
}

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather_Simple_Pull(
    Array<T*, kMaxRanks> uc, SystemSemaphoreInfo* semaphores, int rank, int ranks, int64_t slice, Relaxed relaxed)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    auto local = uc[rank];

    for (int i = 1; i < ranks; ++i) {
        const int p  = rank + i < ranks ? rank + i : rank + i - ranks;
        const T*  ch = cvta_generic_to_global(uc[p]);
        for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < slice; idx += blockDim.x * gridDim.x) {
            local[slice * p + idx] = ch[slice * p + idx];
        }
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather_NVLS_V2(
    T* uc, T* mc, SystemSemaphoreInfo* semaphores, int rank, int ranks, int64_t slice, Relaxed relaxed)
{
#if TURBOMIND_ARCH_SM90
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < slice; idx += blockDim.x * gridDim.x) {
        multimem_st(&mc[slice * rank + idx], uc[slice * rank + idx]);
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
#endif
}

void CudaIpcCommImpl::AllGather(
    const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream)
{
    const size_t bytesize = turbomind::byte_size(type) * sendcount;

    const int ranks = this->n_ranks(group);
    const int rank  = this->rank(group);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        const auto   symm_ptr = get_symmetric_v2((T*)recvbuff, group);
        const size_t slice    = bytesize / sizeof(T);
        const int    threads  = 1024;
        if (symm_ptr.mc) {
            const int blocks = std::min<int>(4, (slice + threads - 1) / threads);
            Allgather_NVLS_V2<T><<<blocks, threads, 0, stream>>>(
                symm_ptr.uc[rank], symm_ptr.mc, semaphore, rank, ranks, slice, std::false_type{});
        }
        else {
            const int blocks = std::min<int>(max_ctas_.apply(32), (slice + threads - 1) / threads);
            Allgather_Simple_Pull<T>
                <<<blocks, threads, 0, stream>>>(symm_ptr.uc, semaphore, rank, ranks, slice, std::false_type{});
        }
    };

    auto invoke_copy_engine = [&] {
        auto symm_ptr = get_symmetric_v2((char*)recvbuff, group);

        Barrier(group, stream);

        for (int i = 1; i < ranks; ++i) {
            const int p = (rank + i) % ranks;
            check_cuda_error(cudaMemcpyAsync(symm_ptr.uc[p] + rank * bytesize,  //
                                             (char*)recvbuff + rank * bytesize,
                                             bytesize,
                                             cudaMemcpyDefault,
                                             stream));
        }

        Barrier(group, stream);
    };

    if (bytesize < copy_threshold_) {
        if (bytesize % sizeof(uint4) == 0) {
            invoke(uint4{});
        }
        else if (bytesize % sizeof(uint2) == 0) {
            invoke(uint2{});
        }
        else if (bytesize % sizeof(uint) == 0) {
            invoke(uint{});
        }
        else {
            TM_CHECK(0) << "not implemented";
        }
    }
    else {
        invoke_copy_engine();
    }
}

template<class T, int log2_block_dim, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather2D_Simple_Pull(T*                   local,
                                                                   Array<T*, kMaxRanks> uc,
                                                                   SystemSemaphoreInfo* semaphores,
                                                                   int                  rank,
                                                                   int                  ranks,
                                                                   int64_t              pitch,
                                                                   int64_t              stride,
                                                                   int                  width,
                                                                   int                  height,
                                                                   int                  log2_groups,
                                                                   constant<log2_block_dim>,
                                                                   Relaxed relaxed)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);

    const int log2_threads = log2_block_dim - log2_groups;
    const int threads      = 1 << log2_threads;
    const int groups       = 1 << log2_groups;

    const int gi = threadIdx.x >> log2_threads;
    const int di = (threadIdx.x & (threads - 1));
    const int bi = blockIdx.x * groups + gi;
    const int bn = gridDim.x * groups;

    sem.Wait(relaxed);

    __syncthreads();

    for (int i = 1; i < ranks; ++i) {
        const int     p      = rank + i < ranks ? rank + i : rank + i - ranks;
        const T*      ch     = cvta_generic_to_global(uc[p]);
        const int64_t offset = stride * p;
        for (int x = di; x < width; x += threads) {
            for (int y = bi; y < height; y += bn) {
                local[offset + y * pitch + x] = ch[offset + y * pitch + x];
            }
        }
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, int log2_block_dim, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather2D_NVLS_V2(T*                   uc_buf,
                                                               T*                   mc_buf,
                                                               SystemSemaphoreInfo* semaphores,
                                                               int                  rank,
                                                               int                  ranks,
                                                               int64_t              pitch,
                                                               int64_t              stride,
                                                               int                  width,
                                                               int                  height,
                                                               int                  log2_groups,
                                                               constant<log2_block_dim>,
                                                               Relaxed relaxed)
{

#if TURBOMIND_ARCH_SM90

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    const int log2_threads = log2_block_dim - log2_groups;
    const int threads      = 1 << log2_threads;
    const int groups       = 1 << log2_groups;

    const int gi = threadIdx.x >> log2_threads;
    const int di = (threadIdx.x & (threads - 1));
    const int bi = blockIdx.x * groups + gi;
    const int bn = gridDim.x * groups;

    __syncthreads();

    const int64_t offset = stride * rank;
    for (int y = bi; y < height; y += bn) {
        for (int x = di; x < width; x += threads) {
            const int64_t idx = offset + y * pitch + x;
            multimem_st(&mc_buf[idx], uc_buf[idx]);
        }
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
#endif
}

void CudaIpcCommImpl::AllGather2D(const void*  sendbuff,
                                  void*        recvbuff,
                                  size_t       pitch,
                                  size_t       stride,
                                  int          width,
                                  int          height,
                                  DataType     type,
                                  int2         flags,
                                  int          group,
                                  cudaStream_t stream)
{
    const size_t byte_width  = byte_size(type, width);
    const size_t byte_pitch  = byte_size(type, pitch);
    const size_t byte_stride = byte_size(type, stride);

    const size_t nbytes = byte_width * height;

    const int ranks = this->n_ranks(group);
    const int rank  = this->rank(group);

    TM_CHECK_EQ((char*)sendbuff, (char*)recvbuff + rank * byte_stride);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T = decltype(t);

        const int threads     = 1024;
        int       log2_groups = 0;
        while ((threads * sizeof(T) >> log2_groups) > byte_width * 2) {
            ++log2_groups;
        }
        const int groups = 1 << log2_groups;

        auto symm_ptr = get_symmetric_v2((T*)recvbuff, group);

        if (symm_ptr.mc) {
            const int blocks = std::min<int>(4, (height + groups - 1) >> log2_groups);
            Allgather2D_NVLS_V2<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,
                                                                   symm_ptr.mc,
                                                                   semaphore,
                                                                   rank,
                                                                   this->n_ranks(group),
                                                                   byte_pitch / sizeof(T),
                                                                   byte_stride / sizeof(T),
                                                                   byte_width / sizeof(T),
                                                                   height,
                                                                   log2_groups,
                                                                   constant<10>{},
                                                                   std::true_type{});
        }
        else {
            const int blocks = std::min<int>(max_ctas_.apply(48), (height + groups - 1) >> log2_groups);
            Allgather2D_Simple_Pull<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
                                                                       symm_ptr.uc,
                                                                       semaphore,
                                                                       rank,
                                                                       ranks,
                                                                       byte_pitch / sizeof(T),
                                                                       byte_stride / sizeof(T),
                                                                       byte_width / sizeof(T),
                                                                       height,
                                                                       log2_groups,
                                                                       constant<10>{},
                                                                       std::true_type{});
        }
    };

    auto invoke_copy_engine = [&] {
        auto symm_ptr = get_symmetric_v2((char*)recvbuff, group);

        Barrier(group, stream);

        for (int i = 1; i < ranks; ++i) {
            const int p = (rank + i) % ranks;
            check_cuda_error(cudaMemcpy2DAsync(symm_ptr.uc[p] + rank * byte_stride,
                                               byte_pitch,
                                               (char*)recvbuff + rank * byte_stride,
                                               byte_pitch,
                                               byte_width,
                                               height,
                                               cudaMemcpyDefault,
                                               stream));
        }

        Barrier(group, stream);
    };

    if (nbytes < copy_threshold_) {
        if (byte_width % sizeof(uint4) == 0) {
            invoke(uint4{});
        }
        else if (byte_width % sizeof(uint2) == 0) {
            invoke(uint2{});
        }
        else if (byte_width % sizeof(uint) == 0) {
            invoke(uint{});
        }
        else {
            TM_CHECK(0) << "not implemented";
        }
    }
    else {
        invoke_copy_engine();
    }
}

}  // namespace turbomind::comm
