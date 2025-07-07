// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/device_semaphore.h"
#include "src/turbomind/comm/cuda_ipc/multimem.cuh"

#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

// Modified from
// https://github.com/microsoft/mscclpp/blob/591276f9d07d2df8e2a45a16738e27867e468ca3/test/mscclpp-test/allgather_test.cu#L294
template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather_Simple_Pull(T*                           local,
                                                                 Array<T*, kMaxNearPeers>     near,
                                                                 mscclpp::D2DSemaphoreHandle* semaphores,
                                                                 int                          rank,
                                                                 int                          peers,
                                                                 int64_t                      slice,
                                                                 Relaxed                      relaxed)
{
    const int       sem_id = blockIdx.x * peers + threadIdx.x;
    DeviceSemaphore sem;
    if (threadIdx.x < peers) {
        sem.Load(&semaphores[sem_id]);
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    Rank r{rank, peers};

    for (int i = 0; i < peers; ++i) {
        const int p      = r.get_next_peer(i);
        const int p_rank = r.get_peer_rank(p);
        const T*  ch     = cvta_generic_to_global(near[p]);
        for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < slice; idx += blockDim.x * gridDim.x) {
            local[slice * p_rank + idx] = ch[slice * p_rank + idx];
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(true);
        sem.Save(&semaphores[sem_id]);
    }
}

void CudaIpcCommImpl::AllGather(
    const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, int group, cudaStream_t stream)
{
    const size_t bytesize = turbomind::byte_size(type) * sendcount;

    const int peers = this->n_ranks(group) - 1;
    const int rank  = this->rank(group);

    auto semaphores = groups_.at(group).d2d_semaphores;

    auto invoke = [&](auto t) {
        using T              = decltype(t);
        const auto   near    = get_symmetric((T*)recvbuff, group).uc;
        const size_t slice   = bytesize / sizeof(T);
        const int    threads = 1024;
        const int    blocks  = std::min<int>(32, (slice + threads - 1) / threads);
        Allgather_Simple_Pull<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
                                                                 near,
                                                                 semaphores,
                                                                 rank,
                                                                 peers,
                                                                 slice,
                                                                 std::false_type{});
    };

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
        throw std::runtime_error("not implemented");
    }
}

template<class T, int log2_block_dim, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather2D_Simple_Pull(T*                           local,
                                                                   Array<T*, kMaxNearPeers>     near,
                                                                   mscclpp::D2DSemaphoreHandle* semaphores,
                                                                   int                          rank,
                                                                   int                          peers,
                                                                   int64_t                      pitch,
                                                                   int64_t                      stride,
                                                                   int                          width,
                                                                   int                          height,
                                                                   int                          log2_groups,
                                                                   constant<log2_block_dim>,
                                                                   Relaxed relaxed)
{
    const int       sem_id = blockIdx.x * peers + threadIdx.x;
    DeviceSemaphore sem;
    if (threadIdx.x < peers) {
        sem.Load(&semaphores[sem_id]);
        sem.SignalAndWait(relaxed);
    }

    const int log2_threads = log2_block_dim - log2_groups;
    const int threads      = 1 << log2_threads;
    const int groups       = 1 << log2_groups;

    const int gi = threadIdx.x >> log2_threads;
    const int di = (threadIdx.x & (threads - 1));
    const int bi = blockIdx.x * groups + gi;
    const int bn = gridDim.x * groups;

    __syncthreads();

    Rank r{rank, peers};

    for (int i = 0; i < peers; ++i) {
        const int     p      = r.get_next_peer(i);
        const int     p_rank = r.get_peer_rank(p);
        const T*      ch     = cvta_generic_to_global(near[p]);
        const int64_t offset = stride * p_rank;
        for (int x = di; x < width; x += threads) {
            for (int y = bi; y < height; y += bn) {
                local[offset + y * pitch + x] = ch[offset + y * pitch + x];
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(true);
        sem.Save(&semaphores[sem_id]);
    }
}

template<class T, int log2_block_dim, class Relaxed>
__global__ void __launch_bounds__(1024, 1) Allgather2D_Simple_NVLS(
    // T*                           local,

    T*                           uc_buf,
    T*                           mc_buf,
    Array<T*, kMaxNearPeers>     near,
    mscclpp::D2DSemaphoreHandle* semaphores,
    int                          rank,
    int                          peers,
    int64_t                      pitch,
    int64_t                      stride,
    int                          width,
    int                          height,
    int                          log2_groups,
    constant<log2_block_dim>,
    Relaxed relaxed)
{
    const int       sem_id = blockIdx.x * peers + threadIdx.x;
    DeviceSemaphore sem;
    if (threadIdx.x < peers) {
        sem.Load(&semaphores[sem_id]);
        sem.SignalAndWait(relaxed);
    }

    const int log2_threads = log2_block_dim - log2_groups;
    const int threads      = 1 << log2_threads;
    const int groups       = 1 << log2_groups;

    const int gi = threadIdx.x >> log2_threads;
    const int di = (threadIdx.x & (threads - 1));
    const int bi = blockIdx.x * groups + gi;
    const int bn = gridDim.x * groups;

    __syncthreads();

    if (blockIdx.x % 2 == 0 || 1) {
        const int64_t offset = stride * rank;
        for (int y = bi; y < height; y += bn) {
            for (int x = di; x < width; x += threads) {
                const int64_t idx = offset + y * pitch + x;
                multimem_st(mc_buf + idx, uc_buf[idx]);
            }
        }
    }
    else {
        Rank          r{rank, peers};
        const int64_t offset = stride * rank;
        for (int i = 0; i < peers; ++i) {
            const int p = r.get_next_peer(i);
            // const int     p_rank = r.get_peer_rank(p);
            T* ch = cvta_generic_to_global(near[p]);
            for (int x = di; x < width; x += threads) {
                for (int y = bi; y < height; y += bn) {
                    ch[offset + y * pitch + x] = uc_buf[offset + y * pitch + x];
                }
            }
            // const int64_t offset = stride * p_rank;
            // for (int x = di; x < width; x += threads) {
            //     for (int y = bi; y < height; y += bn) {
            //         uc_buf[offset + y * pitch + x] = ch[offset + y * pitch + x];
            //     }
            // }
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(true);
        sem.Save(&semaphores[sem_id]);
    }
}

__global__ void Barrier(mscclpp::D2DSemaphoreHandle* semaphores, int peers)
{
    const int sem_id = blockIdx.x * peers + threadIdx.x;

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[sem_id]);
        sem.SignalAndWait(false);
        sem.Save(&semaphores[sem_id]);
    }
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

    const int peers = this->n_ranks(group) - 1;
    const int rank  = this->rank(group);

    auto semaphores = groups_.at(group).d2d_semaphores;

#if 1
    auto invoke = [&](auto t) {
        using T = decltype(t);

        const int threads     = 1024;
        int       log2_groups = 0;
        while ((threads * sizeof(T) >> log2_groups) > byte_width * 2) {
            ++log2_groups;
        }
        const int groups = 1 << log2_groups;
        // const int blocks = std::min<int>(48, (height + groups - 1) >> log2_groups);
        // Allgather2D_Simple_Pull<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
        //                                                            near,
        //                                                            semaphores,
        //                                                            rank,
        //                                                            peers,
        //                                                            byte_pitch / sizeof(T),
        //                                                            byte_stride / sizeof(T),
        //                                                            byte_width / sizeof(T),
        //                                                            height,
        //                                                            log2_groups,
        //                                                            constant<10>{},
        //                                                            std::false_type{});

        auto symm_ptr = get_symmetric((T*)recvbuff, group);

        const int blocks = std::min<int>(1, (height + groups - 1) >> log2_groups);

        Allgather2D_Simple_NVLS<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
                                                                   symm_ptr.mc,
                                                                   symm_ptr.uc,
                                                                   semaphores,
                                                                   rank,
                                                                   peers,
                                                                   byte_pitch / sizeof(T),
                                                                   byte_stride / sizeof(T),
                                                                   byte_width / sizeof(T),
                                                                   height,
                                                                   log2_groups,
                                                                   constant<10>{},
                                                                   std::false_type{});
    };

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
        throw std::runtime_error("not implemented");
    }
#else
    auto near = get_near((char*)base);
    for (auto& p : near) {
        if (p) {
            p += offset;
        }
    }
    const int   peers = world_size_ - 1;
    Rank        rank{rank_, peers};
    const char* buff = (const char*)recvbuff;
    if (flags.x) {  // make sure peer buffers are ready to receive
        Barrier<<<1, peers, 0, stream>>>(device_semaphores_, peers);
    }
    for (int i = 0; i < peers; ++i) {
        const int p = rank.get_next_peer(i);
        check_cuda_error(cudaMemcpy2DAsync(near[p] + rank_ * byte_stride,
                                           byte_pitch,
                                           buff + rank_ * byte_stride,
                                           byte_pitch,
                                           byte_width,
                                           height,
                                           cudaMemcpyDefault,
                                           stream));
    }
    if (flags.y) {  // make sure receiving is done
        Barrier<<<1, peers, 0, stream>>>(device_semaphores_, peers);
    }
#endif
}

}  // namespace turbomind::comm
