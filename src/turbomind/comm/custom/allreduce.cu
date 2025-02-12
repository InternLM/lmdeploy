

#include <atomic>
#include <stdexcept>

#include "src/turbomind/comm/custom/custom_comm.h"

#include "src/turbomind/comm/custom/device_semaphore.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/utils/Tensor.h"

#include "mscclpp/concurrency_device.hpp"
#include "mscclpp/packet_device.hpp"

namespace turbomind {

// template<int vec_size, class T>
// __global__ void __launch_bounds__(1024, 1) local_allreduce_kernel(T*                     data,
//                                                                   SmChannels             channels,  // registered
//                                                                   mscclpp::DeviceSyncer* device_syncer,
//                                                                   int                    rank,
//                                                                   int                    world_size,
//                                                                   size_t                 count)
// {
//     local_reduce_scatter<vec_size>(data, channels, device_syncer, rank, world_size, count / world_size);
//     device_syncer->sync(gridDim.x);
//     local_allgather(channels, device_syncer, rank, world_size, sizeof(T) * (count / world_size));
// }

// __launch_bounds__(1024, 1)

template<class T, int vec_size, class Relaxed>
__global__ void AllreduceKernel_Simple(T*                                             buf,
                                       Array<T*, kMaxNearPeers>                       chns,
                                       mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                                       int                                            rank,
                                       int                                            peers,
                                       int                                            slice,
                                       int                                            count,
                                       constant<vec_size>,
                                       Relaxed relaxed)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    for (int i = 0; i < peers; ++i) {
        const int p   = i + rank < peers ? i + rank : i + rank - peers;
        auto      chn = cvta_generic_to_global(chns[p]);
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Vec acc, tmp;
            Load(tmp, chn + idx * vec_size);
            Load(acc, buf + idx * vec_size);
            acc = acc + tmp;
            Store(buf + idx * vec_size, acc);
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    for (int i = 0; i < peers; ++i) {
        const int p      = i + rank < peers ? i + rank : i + rank - peers;
        const int p_rank = p < rank ? p : p + 1;
        const int first  = p_rank * slice;
        const int last   = min(count, first + slice);
        auto      chn    = cvta_generic_to_global(chns[p]);
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Vec vec;
            Load(vec, chn + idx * vec_size);
            Store(buf + idx * vec_size, vec);
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(true);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
}

// template<int ctas_per_peer, class T>
// __global__ void __launch_bounds__(1024, 1) local_allreduce_kernel_0(T*                     dst,
//                                                                     const T*               src,
//                                                                     void*                  packet_buff,
//                                                                     SmChannels             packet_chns,  //
//                                                                     mscclpp::DeviceSyncer* barrier,
//                                                                     int                    rank,
//                                                                     int                    world_size,
//                                                                     int                    count,
//                                                                     uint32_t               flag)
// {
//     constexpr int vec_size = sizeof(uint2) / sizeof(T);

//     const int n_peer = world_size - 1;
//     const int n_pkt  = count / vec_size;

//     using Vec = Array<T, vec_size>;

//     {
//         const int pi = blockIdx.x / ctas_per_peer;
//         const int bi = blockIdx.x % ctas_per_peer;

//         const int offset = (pi < rank ? rank - 1 : rank) * n_pkt;

//         // cta sends the assigned packets
//         for (int idx = threadIdx.x + bi * blockDim.x; idx < n_pkt; idx += ctas_per_peer * blockDim.x) {
//             mscclpp::LLPacket packet{*((const uint2*)src + idx), flag};
//             packet_chns[pi].write(offset + idx, packet);
//         }
//     }

//     if (src == dst) {
//         // device wide sync is required as recved from all peers does not imply sending is done
//         // there will be data race on `src` unless we do it out-of-place
//         barrier->sync(gridDim.x);
//     }

//     {
//         mscclpp::LLPacket* incoming = reinterpret_cast<mscclpp::LLPacket*>(packet_buff);

//         using namespace ops;

//         // all blocks recv packets and reduce
//         for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n_pkt; idx += blockDim.x * gridDim.x) {
//             Vec vec;
//             Load(vec, src + idx * vec_size);
//             for (int p = 0; p < n_peer; ++p) {
//                 uint2 data = incoming[p * n_pkt + idx].read(flag);
//                 vec        = vec + reinterpret_cast<Vec&>(data);
//             }
//             Store(dst + idx * vec_size, vec);
//         }
//     }
// }

using mscclpp::LLPacket;

// reduce-scatter + allgather using LL16Packet
template<class T, class CtasPerPeer>
__global__ void __launch_bounds__(1024, 1) AllreduceKernel_LL(T*                              dst,
                                                              const T*                        src,
                                                              LLPacket*                       incoming,
                                                              Array<LLPacket*, kMaxNearPeers> outgoing,
                                                              int                             rank,
                                                              int                             peers,
                                                              int                             slice,  // padded slice
                                                              int                             count,  // actual count
                                                              uint32_t                        flag,
                                                              CtasPerPeer                     ctas_per_peer)
{

    constexpr int vec_size = sizeof(uint2) / sizeof(T);

    using Vec = Array<T, vec_size>;

    const int p = [&] {
        const int i = blockIdx.x / ctas_per_peer;
        return i + rank < peers ? i + rank : i + rank - peers;
    }();

    const int bi = blockIdx.x % ctas_per_peer;

    const int p_rank = p < rank ? p : p + 1;

    {  // send slice of `src` to peers  (src -> packet0)
        const int p_offset = (p < rank ? rank - 1 : rank) * slice;
        auto      chn      = outgoing[p] + p_offset;
        for (int idx = threadIdx.x + bi * blockDim.x; idx < slice; idx += ctas_per_peer * blockDim.x) {
            chn[idx].write(*((const uint2*)src + p_rank * slice + idx), flag);
        }
    }

    // device-wide barrier not required as what we are sending is not what we are going to modify

    {  // recv data | reduce | send results (src -> packet0 -> packet1)
        __shared__ LLPacket* out[kMaxNearPeers];
        for (int i = 0; i < peers; ++i) {
            const int p = i + rank < peers ? i + rank : i + rank - peers;
            out[i]      = outgoing[p] + (peers + rank) * slice;
        }
        using namespace ops;
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < slice; idx += blockDim.x * gridDim.x) {
            Vec vec;
            Load(vec, src + (rank * slice + idx) * vec_size);
            for (int p = 0; p < peers; ++p) {
                uint2 data = incoming[p * slice + idx].read(flag);
                vec        = vec + (Vec&)data;
            }
            Store(dst + (rank * slice + idx) * vec_size, vec);
            for (int i = 0; i < peers; ++i) {
                out[i][idx].write((uint2&)vec, flag);
            }
        }
    }

    {  // recv results (packet1 -> dst)
        incoming += (peers + p_rank) * slice;
        dst += p_rank * slice * vec_size;
        // ! note that `dst` MUST have same partition as we are sending `src`
        for (int idx = threadIdx.x + bi * blockDim.x; idx < slice; idx += ctas_per_peer * blockDim.x) {
            uint2 data = incoming[idx].read(flag);
            Store(dst + idx * vec_size, (Vec&)data);
        }
    }
}

void CustomComm::AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream)
{
    FT_CHECK(sendbuff == recvbuff);

    void* data = recvbuff;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (sizeof(T) * count <= 1 << 20) {
            FT_CHECK((int)packet_chns_.size() == world_size_ - 1);
            constexpr int vec_size      = sizeof(uint2) / sizeof(T);
            const int     slice         = (count / vec_size + world_size_ - 1) / world_size_;
            constexpr int ctas_per_peer = 3;
            constexpr int threads       = 1024;
            const int     blocks        = (world_size_ - 1) * ctas_per_peer;
            auto          incoming      = (LLPacket*)packet_buff_;
            auto          outgoing      = get_near(incoming);
            AllreduceKernel_LL<<<blocks, threads, 0, stream>>>((T*)data,  //
                                                               (T*)data,
                                                               incoming,
                                                               outgoing,
                                                               rank_,
                                                               world_size_ - 1,
                                                               slice,
                                                               count / vec_size,
                                                               flag_++,
                                                               constant<ctas_per_peer>{});
        }
        else {
            constexpr int vec_size = sizeof(uint4) / sizeof(T);
            const int     slice    = (count / vec_size + world_size_ - 1) / world_size_;
            constexpr int threads  = 1024;
            const int     blocks   = std::min(48, (slice + threads - 1) / threads);
            auto          chns     = get_near((T*)data);
            AllreduceKernel_Simple<<<blocks, threads, 0, stream>>>((T*)data,
                                                                   chns,
                                                                   device_semaphores_,
                                                                   rank_,
                                                                   world_size_ - 1,
                                                                   slice,
                                                                   count / vec_size,
                                                                   constant<vec_size>{},
                                                                   std::false_type{});
        }
    };

    switch (type) {
        case DataType::TYPE_FP16:
            return invoke(half{});
        case DataType::TYPE_BF16:
            return invoke(nv_bfloat16{});
        default:
            throw std::runtime_error("not implemented");
    }
}

}  // namespace turbomind