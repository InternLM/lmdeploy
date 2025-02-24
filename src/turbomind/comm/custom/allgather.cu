// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/custom/custom_comm.h"

#include "src/turbomind/comm/custom/device_semaphore.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

// Modified from
// https://github.com/microsoft/mscclpp/blob/591276f9d07d2df8e2a45a16738e27867e468ca3/test/mscclpp-test/allgather_test.cu#L294
template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1)
    Allgather_Simple_Pull(T*                                             local,
                          Array<T*, kMaxNearPeers>                       near,
                          mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                          int                                            rank,
                          int                                            peers,
                          int64_t                                        slice,
                          Relaxed                                        relaxed)
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

void CustomComm::AllGather(const void* sendbuff, void* recvbuff, size_t sendcount, DataType type, cudaStream_t stream)
{
    const size_t bytesize = get_elem_size(type) * sendcount;

    auto invoke = [&](auto t) {
        using T              = decltype(t);
        const auto   near    = get_near((T*)recvbuff);
        const size_t slice   = bytesize / sizeof(T);
        const int    threads = 1024;
        const int    blocks  = std::min<int>(32, (slice + threads - 1) / threads);
        Allgather_Simple_Pull<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
                                                                 near,
                                                                 device_semaphores_,
                                                                 rank_,
                                                                 world_size_ - 1,
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

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1)
    Allgather2D_Simple_Pull(T*                                             local,
                            Array<T*, kMaxNearPeers>                       near,
                            mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                            int                                            rank,
                            int                                            peers,
                            int64_t                                        pitch,
                            int64_t                                        stride,
                            int                                            width,
                            int                                            height,
                            Relaxed                                        relaxed)
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
        const int     p      = r.get_next_peer(i);
        const int     p_rank = r.get_peer_rank(p);
        const T*      ch     = cvta_generic_to_global(near[p]);
        const int64_t offset = stride * p_rank;
        for (int y = blockIdx.x; y < height; y += gridDim.x) {
            for (int x = threadIdx.x; x < width; x += blockDim.x) {
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

__global__ void Barrier(mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores, int peers)
{
    const int sem_id = blockIdx.x * peers + threadIdx.x;

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[sem_id]);
        sem.SignalAndWait(false);
        sem.Save(&semaphores[sem_id]);
    }
}

void CustomComm::AllGather2D(const void*  sendbuff,
                             void*        recvbuff,
                             size_t       pitch,
                             size_t       stride,
                             int          width,
                             int          height,
                             DataType     type,
                             int2         flags,
                             cudaStream_t stream)
{
    const size_t byte_width  = get_elem_size(type) * width;
    const size_t byte_pitch  = get_elem_size(type) * pitch;
    const size_t byte_stride = get_elem_size(type) * stride;

    void*  base{};
    size_t offset{};
    for (auto& [p, m] : registered_memories_) {
        if ((char*)p <= (char*)recvbuff && (char*)recvbuff < (char*)p + m.front().second) {
            base   = p;
            offset = (char*)recvbuff - (char*)p;
        }
    }
    FT_CHECK(base);

#if 1
    auto invoke = [&](auto t) {
        using T   = decltype(t);
        auto near = get_near((T*)base);
        for (auto& p : near) {
            if (p) {
                p += offset / sizeof(T);
            }
        }
        const int threads = 1024;
        const int blocks  = std::min<int>(32, height);
        Allgather2D_Simple_Pull<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
                                                                   near,
                                                                   device_semaphores_,
                                                                   rank_,
                                                                   world_size_ - 1,
                                                                   byte_pitch / sizeof(T),
                                                                   byte_stride / sizeof(T),
                                                                   byte_width / sizeof(T),
                                                                   height,
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
