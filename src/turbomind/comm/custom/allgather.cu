#include <stdexcept>

#include "src/turbomind/comm/custom/custom_comm.h"

#include "src/turbomind/comm/custom/device_semaphore.h"

#include "mscclpp/concurrency_device.hpp"

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/string_utils.h"

namespace turbomind {

// __device__ inline void local_allgather(SmChannels&            channels,  //
//                                        mscclpp::DeviceSyncer* device_syncer,
//                                        int                    rank,
//                                        int                    world_size,
//                                        size_t                 size)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     const int block_num  = gridDim.x;
//     const int thread_num = blockDim.x * block_num;

//     const int n_peer = world_size - 1;

//     if (tid < n_peer) {
//         channels[tid].signal();
//         channels[tid].wait();
//     }

//     device_syncer->sync(gridDim.x);

//     for (int i = 0; i < n_peer; ++i) {
//         const int    peer_idx        = i + rank < n_peer ? i + rank : i + rank - n_peer;
//         const int    remote_rank_idx = peer_idx < rank ? peer_idx : peer_idx + 1;
//         const size_t offset          = size * remote_rank_idx;
//         channels[peer_idx].get<16, false>(offset, size, tid, thread_num);
//     }
// }

// __global__ void __launch_bounds__(1024, 1) local_allgather_kernel(SmChannels             channels,  //
//                                                                   mscclpp::DeviceSyncer* device_syncer,
//                                                                   int                    rank,
//                                                                   int                    world_size,
//                                                                   size_t                 size)
// {
//     local_allgather(channels, device_syncer, rank, world_size, size);
// }

template<class T, class Relaxed>
__global__ void __launch_bounds__(1024, 1) AllgatherKernel(T*                                             local,
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

    for (int i = 0; i < peers; ++i) {
        const int p      = i + rank < peers ? i + rank : i + rank - peers;
        const int p_rank = p < rank ? p : p + 1;
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
        AllgatherKernel<T><<<blocks, threads, 0, stream>>>((T*)recvbuff,  //
                                                           near,
                                                           device_semaphores_,
                                                           rank_,
                                                           world_size_ - 1,
                                                           slice,
                                                           false);
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
        std::cout << "fuck\n";
        throw std::runtime_error("not implemented");
    }
}

}  // namespace turbomind