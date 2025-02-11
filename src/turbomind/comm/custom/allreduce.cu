

#include <atomic>
#include <stdexcept>

#include "src/turbomind/comm/common.h"

#include "src/turbomind/comm/custom/allgather.h"
#include "src/turbomind/comm/custom/custom_comm.h"
#include "src/turbomind/comm/custom/device_semaphore.h"
#include "src/turbomind/comm/custom/reduce_scatter.h"

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/utils/Tensor.h"

#include "mscclpp/concurrency_device.hpp"
#include "mscclpp/packet_device.hpp"

namespace turbomind {

template<int vec_size, class T>
__global__ void __launch_bounds__(1024, 1) local_allreduce_kernel(T*                     data,
                                                                  SmChannels             channels,  // registered
                                                                  mscclpp::DeviceSyncer* device_syncer,
                                                                  int                    rank,
                                                                  int                    world_size,
                                                                  size_t                 count)
{
    local_reduce_scatter<vec_size>(data, channels, device_syncer, rank, world_size, count / world_size);
    device_syncer->sync(gridDim.x);
    local_allgather(channels, device_syncer, rank, world_size, sizeof(T) * (count / world_size));
}

// __launch_bounds__(1024, 1)

template<int vec_size, class T>
__global__ void local_allreduce_kernel_v2(T*                                             buf,
                                          Array<T*, 8>                                   chns,
                                          mscclpp::DeviceSyncer*                         barrier,
                                          mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                                          int                                            rank,
                                          int                                            world_size,
                                          size_t                                         count)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_peer = world_size - 1;

    DeviceSemaphore sem;

    const int lane_id = threadIdx.x % WARP_SIZE;
    if (threadIdx.x < n_peer) {
        sem.Load(&semaphores[blockIdx.x * n_peer + lane_id]);
    }
    // __syncwarp();

    // if (blockIdx.x == 0 && threadIdx.x < n_peer) {
    //     channels[threadIdx.x].signal();
    //     channels[threadIdx.x].wait();
    // }
    // barrier->sync(block_num);

    if (threadIdx.x < n_peer) {
        // It seems that fence is not needed on NVLink devices
        sem.Signal(cuda::memory_order_relaxed);
        sem.Wait(cuda::memory_order_relaxed);

        // sem.Signal(cuda::memory_order_release);
        // sem.Wait(cuda::memory_order_acquire);
    }

    __syncthreads();

    count /= vec_size * world_size;

    const int offset = rank * (int)count;

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int first = offset;
    const int last  = offset + (int)count;

    for (int p = 0; p < n_peer; ++p) {
        const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
        auto      chn  = cvta_generic_to_global(chns[peer]);
        Vec       acc, tmp;
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Load(tmp, chn + idx * vec_size);
            Load(acc, buf + idx * vec_size);
            acc = acc + tmp;
            Store(buf + idx * vec_size, acc);
        }
    }

    __syncthreads();

    if (threadIdx.x < n_peer) {
        // asm volatile("fence.acq_rel.sys;" ::: "memory");
        // sem.relaxedSignal();
        // sem.wait();

        // It seems that fence is not needed on NVLink devices
        sem.Signal(cuda::memory_order_relaxed);
        sem.Wait(cuda::memory_order_relaxed);

        // sem.Signal(cuda::memory_order_release);
        // sem.Wait(cuda::memory_order_acquire);
    }

    __syncthreads();

    // barrier->sync(block_num);
    // if (blockIdx.x == 0 && threadIdx.x < n_peer) {
    //     channels[threadIdx.x].signal();
    //     channels[threadIdx.x].wait();
    // }
    // barrier->sync(block_num);

    for (int p = 0; p < n_peer; ++p) {
        const int peer      = p + rank < n_peer ? p + rank : p + rank - n_peer;
        const int peer_rank = peer < rank ? peer : peer + 1;
        const int first     = (int)count * peer_rank;
        const int last      = first + (int)count;
        auto      chn       = cvta_generic_to_global(chns[peer]);
        for (size_t idx = first + thread_idx; idx < last; idx += thread_num) {
            Vec vec;
            Load(vec, chn + idx * vec_size);
            Store(buf + idx * vec_size, vec);
        }
    }

    if (threadIdx.x < n_peer) {
        sem.Save(&semaphores[blockIdx.x * n_peer + lane_id]);
    }
}

template<int ctas_per_peer, class T>
__global__ void __launch_bounds__(1024, 1) local_allreduce_kernel_0(T*                     dst,
                                                                    const T*               src,
                                                                    void*                  packet_buff,
                                                                    SmChannels             packet_chns,  //
                                                                    mscclpp::DeviceSyncer* barrier,
                                                                    int                    rank,
                                                                    int                    world_size,
                                                                    int                    count,
                                                                    uint32_t               flag)
{
    constexpr int vec_size = sizeof(uint2) / sizeof(T);

    const int n_peer = world_size - 1;
    const int n_pkt  = count / vec_size;

    using Vec = Array<T, vec_size>;

    {
        const int pi = blockIdx.x / ctas_per_peer;
        const int bi = blockIdx.x % ctas_per_peer;

        const int offset = (pi < rank ? rank - 1 : rank) * n_pkt;

        // cta sends the assigned packets
        for (int idx = threadIdx.x + bi * blockDim.x; idx < n_pkt; idx += ctas_per_peer * blockDim.x) {
            mscclpp::LLPacket packet{*((const uint2*)src + idx), flag};
            packet_chns[pi].write(offset + idx, packet);
        }
    }

    if (src == dst) {
        // device wide sync is required as recved from all peers does not imply sending is done
        // there will be data race on `src` unless we do it out-of-place
        barrier->sync(gridDim.x);
    }

    {
        mscclpp::LLPacket* incoming = reinterpret_cast<mscclpp::LLPacket*>(packet_buff);

        using namespace ops;

        // all blocks recv packets and reduce
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n_pkt; idx += blockDim.x * gridDim.x) {
            Vec vec;
            Load(vec, src + idx * vec_size);
            for (int p = 0; p < n_peer; ++p) {
                uint2 data = incoming[p * n_pkt + idx].read(flag);
                vec        = vec + reinterpret_cast<Vec&>(data);
            }
            Store(dst + idx * vec_size, vec);
        }
    }
}

// reduce-scatter + allgather using LL16Packet
template<int ctas_per_peer, class T>
__global__ void __launch_bounds__(1024, 1) local_allreduce_kernel_1(T*                     dst,
                                                                    const T*               src,
                                                                    void*                  packet_buff,
                                                                    SmChannels             packet_chns,  //
                                                                    mscclpp::DeviceSyncer* barrier,
                                                                    int                    rank,
                                                                    int                    world_size,
                                                                    int                    count,
                                                                    uint32_t               flag)
{

    constexpr int vec_size = sizeof(uint2) / sizeof(T);

    const int n_peer = world_size - 1;
    const int n_pkt  = count / vec_size / world_size;

    using Vec = Array<T, vec_size>;

    const int _p = blockIdx.x / ctas_per_peer;

    const int pi = _p + rank < n_peer ? _p + rank : _p + rank - n_peer;
    const int bi = blockIdx.x % ctas_per_peer;

    const int peer_rank = pi < rank ? pi : pi + 1;

    {  // send slice of `src` to peers  (src -> packet0)
        const int peer_offset = (pi < rank ? rank - 1 : rank) * n_pkt;
        auto      chn         = packet_chns[pi];
        for (int idx = threadIdx.x + bi * blockDim.x; idx < n_pkt; idx += ctas_per_peer * blockDim.x) {
            mscclpp::LLPacket packet{*((const uint2*)src + peer_rank * n_pkt + idx), flag};
            chn.write(peer_offset + idx, packet);
        }
    }

    // device-wide barrier not required as what we are sending is not what we are going to modify

    {  // recv data | reduce | send results (src -> packet0 -> packet1)
        mscclpp::LLPacket* incoming = reinterpret_cast<mscclpp::LLPacket*>(packet_buff);
        using namespace ops;
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n_pkt; idx += blockDim.x * gridDim.x) {
            Vec vec;
            Load(vec, src + (rank * n_pkt + idx) * vec_size);
            for (int p = 0; p < n_peer; ++p) {
                uint2 data = incoming[p * n_pkt + idx].read(flag);
                vec        = vec + reinterpret_cast<Vec&>(data);
            }
            Store(dst + (rank * n_pkt + idx) * vec_size, vec);
            mscclpp::LLPacket packet{(uint2&)vec, flag};
            for (int p = 0; p < n_peer; ++p) {
                const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
                packet_chns[peer].write(idx + (n_peer + rank) * n_pkt, packet);
            }
        }
    }

    {  // recv results (packet1 -> dst)
        mscclpp::LLPacket* incoming = reinterpret_cast<mscclpp::LLPacket*>(packet_buff);
        incoming += (n_peer + peer_rank) * n_pkt;
        // ! note that `dst` MUST have same partition as we are sending `src`
        for (int idx = threadIdx.x + bi * blockDim.x; idx < n_pkt; idx += ctas_per_peer * blockDim.x) {
            uint2 data = incoming[idx].read(flag);
            Store(dst + (peer_rank * n_pkt + idx) * vec_size, (Vec&)data);
        }
    }
}

void CustomComm::AllReduceSum(const void* sendbuff, void* recvbuff, size_t count, DataType type, cudaStream_t stream)
{
    FT_CHECK(sendbuff == recvbuff);
    auto& data_chns = registered_channels_.at(recvbuff);

    SmChannels d_data_chns{};
    for (size_t i = 0; i < data_chns.size(); ++i) {
        d_data_chns[i] = mscclpp::deviceHandle(data_chns[i]);
    }

    SmChannels d_packet_chns{};
    for (size_t i = 0; i < packet_chns_.size(); ++i) {
        d_packet_chns[i] = mscclpp::deviceHandle(packet_chns_[i]);
    }

    auto&           peer_mems = registered_memories_.at(recvbuff);
    Array<void*, 8> peer_data{};
    for (size_t i = 0; i < peer_mems.size(); ++i) {
        peer_data[i] = peer_mems[i].data();
    }

    void* data = recvbuff;

    auto invoke = [&](auto t, auto vec_size) {
        using T = decltype(t);
        if (sizeof(T) * count <= 1 << 20) {
            FT_CHECK((int)packet_chns_.size() == world_size_ - 1);
            constexpr int ctas_per_peer = 4;
            constexpr int threads       = 1024;
            const int     blocks        = (world_size_ - 1) * ctas_per_peer;
            local_allreduce_kernel_1<ctas_per_peer><<<blocks, threads, 0, stream>>>((T*)data,  //
                                                                                    (T*)data,
                                                                                    packet_buff_,
                                                                                    d_packet_chns,
                                                                                    device_syncer_,
                                                                                    rank_,
                                                                                    world_size_,
                                                                                    count,
                                                                                    flag_++);
        }
        else {
            constexpr int threads = 1024;
            constexpr int blocks  = 32;
            local_allreduce_kernel_v2<vec_size.value><<<blocks, threads, 0, stream>>>((T*)data,  //
                                                                                      (Array<T*, 8>&)peer_data,
                                                                                      device_syncer_,
                                                                                      device_semaphores_,
                                                                                      rank_,
                                                                                      world_size_,
                                                                                      count);
        }
    };

    switch (type) {
        case DataType::TYPE_FP16:
            return invoke(half{}, std::integral_constant<int, 8>{});
        default:
            throw std::runtime_error("not implemented");
    }
}

}  // namespace turbomind