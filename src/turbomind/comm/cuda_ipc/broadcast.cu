// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/multimem.cuh"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/comm/cuda_ipc/semaphore.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Broadcast_NVLS_V2(const T*             uc,
                                                             T*                   mc,
                                                             SystemSemaphoreInfo* semaphores,
                                                             int                  rank,
                                                             int                  ranks,
                                                             int                  root,
                                                             int64_t              slice,
                                                             int64_t              count,
                                                             Relaxed              relaxed)
{

#if TURBOMIND_ARCH_SM90
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    int64_t first = rank * slice;
    int64_t last  = min(first + slice, count);

    for (int64_t idx = first + threadIdx.x + blockIdx.x * blockDim.x; idx < last; idx += blockDim.x * gridDim.x) {
        multimem_st(&mc[idx], uc[idx]);
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
#endif
}

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Broadcast_Simple_Pull(Array<T*, kMaxRanks> uc,
                                                                 SystemSemaphoreInfo* semaphores,
                                                                 int                  rank,
                                                                 int                  ranks,
                                                                 int                  root,
                                                                 int64_t              slice,
                                                                 Relaxed              relaxed)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    auto dst = uc[rank];
    auto src = uc[root];

    if (rank != root) {
        for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < slice; idx += blockDim.x * gridDim.x) {
            dst[idx] = src[idx];
        }
        __syncthreads();
    }

    sem.Signal(relaxed);
    sem.Wait(relaxed);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Broadcast_Simple_V2(Array<T*, kMaxRanks> uc,
                                                               SystemSemaphoreInfo* semaphores,
                                                               int                  index,
                                                               int                  rank,
                                                               int                  ranks,
                                                               int                  root,
                                                               int64_t              slice,
                                                               int64_t              count,
                                                               Relaxed              relaxed)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    auto dst = uc[rank];
    auto src = uc[root];

    int64_t first = index * slice;
    int64_t last  = min(first + slice, count);

    if (rank != root) {
        for (int64_t idx = first + threadIdx.x + blockIdx.x * blockDim.x; idx < last; idx += blockDim.x * gridDim.x) {
            dst[idx] = src[idx];
            for (int i = 0; i < ranks; ++i) {
                int p = rank + i < ranks ? rank + i : rank + i - ranks;
                if (p != root) {
                    uc[p][idx] = dst[idx];
                }
            }
        }
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::Broadcast(const void*  sendbuff,  //
                                void*        recvbuff,
                                size_t       count,
                                DataType     type,
                                int          root,
                                int          group,
                                cudaStream_t stream)
{

    const int rank  = this->rank(group);
    const int ranks = this->n_ranks(group);

    const size_t bytesize = turbomind::byte_size(type, count);

    auto semaphore = groups_.at(group).semaphore.handle();

    const int algo = 5;

    if (algo == 0) {
        Barrier(group, stream);
        if (rank != root) {
            SymmetricPtr_V2<char> symm_ptr = get_symmetric_v2((char*)recvbuff, group);
            check_cuda_error(cudaMemcpyAsync((char*)recvbuff, symm_ptr.uc[root], bytesize, cudaMemcpyDefault, stream));
        }
        Barrier(group, stream);
    }
    else if (algo == 1) {
        const int    slices = 16;
        const size_t slice  = bytesize / slices;
        TM_CHECK(bytesize % slices == 0);
        TM_CHECK_EQ(root, 0);
        SymmetricPtr_V2<char> symm_ptr = get_symmetric_v2((char*)recvbuff, group);
        for (int i = 1; i <= ranks + slices - 2; ++i) {
            Barrier(group, stream);
            int s = i - rank;
            if (0 <= s && s < slices && rank != root) {
                check_cuda_error(cudaMemcpyAsync(
                    (char*)recvbuff + s * slice, symm_ptr.uc[rank - 1] + s * slice, slice, cudaMemcpyDefault, stream));
            }
        }
        Barrier(group, stream);
    }
    else if (algo == 2) {
        SymmetricPtr_V2<char> symm_ptr = get_symmetric_v2((char*)recvbuff, group);
        TM_CHECK_EQ(ranks, 8);
        TM_CHECK_EQ(root, 0);
        Barrier(group, stream);
        if (rank == 4) {
            check_cuda_error(
                cudaMemcpyAsync((char*)recvbuff, symm_ptr.uc[rank - 4], bytesize, cudaMemcpyDefault, stream));
        }
        Barrier(group, stream);
        if (rank == 2 || rank == 6) {
            check_cuda_error(
                cudaMemcpyAsync((char*)recvbuff, symm_ptr.uc[rank - 2], bytesize, cudaMemcpyDefault, stream));
        }
        Barrier(group, stream);
        if (rank & 1) {
            check_cuda_error(
                cudaMemcpyAsync((char*)recvbuff, symm_ptr.uc[rank - 1], bytesize, cudaMemcpyDefault, stream));
        }
        Barrier(group, stream);
    }
    else if (algo == 3) {
        using T               = uint4;
        const auto   symm_ptr = get_symmetric_v2((T*)recvbuff, group);
        const size_t count    = bytesize / sizeof(T);
        const size_t slice    = cdiv<size_t>(count, ranks);
        const int    threads  = 1024;
        const int    blocks   = std::min<int>(2, (slice + threads - 1) / threads);
        Broadcast_NVLS_V2<T><<<blocks, threads, 0, stream>>>(
            symm_ptr.uc[root], symm_ptr.mc, semaphore, rank, ranks, root, slice, count, std::true_type{});
    }
    else if (algo == 4) {
        using T               = uint4;
        const auto   symm_ptr = get_symmetric_v2((T*)recvbuff, group);
        const size_t slice    = bytesize / sizeof(T);
        const int    threads  = 1024;
        const int    blocks   = std::min<int>(32, (slice + threads - 1) / threads);
        Broadcast_Simple_Pull<T>
            <<<blocks, threads, 0, stream>>>(symm_ptr.uc, semaphore, rank, ranks, root, slice, std::false_type{});
    }
    else if (algo == 5) {
        using T               = uint4;
        const auto   symm_ptr = get_symmetric_v2((T*)recvbuff, group);
        const size_t count    = bytesize / sizeof(T);
        const int    peers    = ranks - 1;
        const size_t slice    = (count + peers - 1) / peers;
        const int    threads  = 1024;
        const int    blocks   = std::min<int>(32, (slice + threads - 1) / threads);
        const int    index    = rank >= root ? rank - 1 : rank;
        Broadcast_Simple_V2<T><<<blocks, threads, 0, stream>>>(
            symm_ptr.uc, semaphore, index, rank, ranks, root, slice, count, std::false_type{});
    }
    else if (algo == 6) {
        TM_CHECK_EQ(ranks, 8);
        TM_CHECK_EQ(root, 0);
        const auto   symm_ptr = get_symmetric_v2((char*)recvbuff, group);
        const size_t count    = bytesize;
        const size_t slice    = cdiv<size_t>(count, ranks);

        // 0->4
        // 0->2,       4->6
        // 0->1, 2->3, 4->5, 6->7
        Barrier(group, stream);
        if (rank == 0) {
            check_cuda_error(cudaMemcpyAsync(symm_ptr.uc[rank + 4] + slice * (rank + 4),
                                             symm_ptr.uc[rank + 0] + slice * (rank + 4),
                                             slice * 4,
                                             cudaMemcpyDefault,
                                             stream));
        }
        Barrier(group, stream);
        if (rank == 0 || rank == 4) {
            check_cuda_error(cudaMemcpyAsync(symm_ptr.uc[rank + 2] + slice * (rank + 2),
                                             symm_ptr.uc[rank + 0] + slice * (rank + 2),
                                             slice * 2,
                                             cudaMemcpyDefault,
                                             stream));
        }
        Barrier(group, stream);
        if (rank % 2 == 0) {
            check_cuda_error(cudaMemcpyAsync(symm_ptr.uc[rank + 1] + slice * (rank + 1),
                                             symm_ptr.uc[rank + 0] + slice * (rank + 1),
                                             slice * 1,
                                             cudaMemcpyDefault,
                                             stream));
        }
        Barrier(group, stream);
        for (int i = 1; i < ranks; ++i) {
            const int p = (rank + i) % ranks;
            check_cuda_error(cudaMemcpyAsync(symm_ptr.uc[p] + rank * slice,  //
                                             (char*)recvbuff + rank * slice,
                                             slice,
                                             cudaMemcpyDefault,
                                             stream));
        }
        Barrier(group, stream);
    }
}

}  // namespace turbomind::comm
