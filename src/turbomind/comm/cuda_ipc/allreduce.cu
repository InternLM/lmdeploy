// Copyright (c) OpenMMLab. All rights reserved.

#include <atomic>
#include <stdexcept>

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/device_semaphore.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

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

    Rank r{rank, peers};

    const int bi     = blockIdx.x % ctas_per_peer;
    const int p      = r.get_next_peer(blockIdx.x / ctas_per_peer);
    const int p_rank = p < rank ? p : p + 1;

    {  // send slice of `src` to peers  (src -> packet0)
        auto chn = outgoing[p] + r.inverse_peer(p) * slice;
        for (int idx = threadIdx.x + bi * blockDim.x; idx < slice; idx += ctas_per_peer * blockDim.x) {
            chn[idx].write(*((const uint2*)src + p_rank * slice + idx), flag);
        }
    }

    // device-wide barrier not required as what we are sending is not what we are going to modify

    {  // recv data | reduce | send results (src -> packet0 -> packet1)
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
                const int p = r.get_next_peer(i);
                outgoing[p][(peers + rank) * slice + idx].write((uint2&)vec, flag);
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

// No obvious gain vs non-WS variant
template<class T, class CtasPerPeer>
__global__ void __launch_bounds__(1024, 1) AllreduceKernel_LL_WS(T*                              dst,
                                                                 const T*                        src,
                                                                 LLPacket*                       incoming,
                                                                 Array<LLPacket*, kMaxNearPeers> outgoing,
                                                                 int                             rank,
                                                                 int                             peers,
                                                                 int                             slice,
                                                                 int                             count,
                                                                 uint32_t                        flag,
                                                                 CtasPerPeer                     ctas_per_peer)
{

    constexpr int vec_size = sizeof(uint2) / sizeof(T);

    using Vec = Array<T, vec_size>;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int block_dim  = blockDim.x / 2;
    const int thread_idx = warp_id / 2 * WARP_SIZE + lane_id;

    Rank r{rank, peers};

    if (warp_id % 2 == 0) {
        // send slice of `src` to peers  (src -> packet0)
        const int bi     = blockIdx.x % ctas_per_peer;
        const int p      = r.get_next_peer(blockIdx.x / ctas_per_peer);
        const int p_rank = r.get_peer_rank(p);
        auto      chn    = outgoing[p] + r.inverse_peer(p) * slice;
        for (int idx = thread_idx + bi * block_dim; idx < slice; idx += ctas_per_peer * block_dim) {
            chn[idx].write(*((const uint2*)src + p_rank * slice + idx), flag);
        }
        // recv results (packet1 -> dst)
        incoming += (peers + p_rank) * slice;
        dst += p_rank * slice * vec_size;
        // ! note that `dst` MUST have same partition as we are sending `src`
        for (int idx = thread_idx + bi * block_dim; idx < slice; idx += ctas_per_peer * block_dim) {
            uint2 data = incoming[idx].read(flag);
            Store(dst + idx * vec_size, (Vec&)data);
        }
    }
    else {
        // recv data | reduce | send results (src -> packet0 -> packet1)
        using namespace ops;
        for (int idx = thread_idx + blockIdx.x * block_dim; idx < slice; idx += block_dim * gridDim.x) {
            Vec vec;
            Load(vec, src + (rank * slice + idx) * vec_size);
            for (int p = 0; p < peers; ++p) {
                uint2 data = incoming[p * slice + idx].read(flag);
                vec        = vec + (Vec&)data;
            }
            Store(dst + (rank * slice + idx) * vec_size, vec);
            for (int i = 0; i < peers; ++i) {
                const int p = r.get_next_peer(i);
                outgoing[p][(peers + rank) * slice + idx].write((uint2&)vec, flag);
            }
        }
    }
}

// Modified from
// https://github.com/microsoft/mscclpp/blob/591276f9d07d2df8e2a45a16738e27867e468ca3/test/mscclpp-test/allreduce_test.cu#L963
template<class T, int vec_size, class Relaxed>
__global__ void Allreduce_Simple_Pull(T*                           buf,
                                      Array<T*, kMaxNearPeers>     chns,
                                      mscclpp::D2DSemaphoreHandle* semaphores,
                                      int                          rank,
                                      int                          peers,
                                      int                          slice,
                                      int                          count,
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

// ! slice <= grid size
// Slightly lower latency
template<class T, int vec_size, class Relaxed>
__global__ void Allreduce_Simple_Push(T*                           buf,
                                      T*                           scratch,
                                      Array<T*, kMaxNearPeers>     near,
                                      mscclpp::D2DSemaphoreHandle* semaphores,
                                      int                          rank,
                                      int                          peers,
                                      int                          slice,  // in vec
                                      int                          count,  // in vec
                                      constant<vec_size>,
                                      Relaxed relaxed)
{
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    using Vec = Array<T, vec_size>;

    Rank r{rank, peers};

    for (int i = 0; i < peers; ++i) {
        const int p      = r.get_next_peer(i);
        const int p_rank = r.get_peer_rank(p);
        if (int idx = thread_idx; idx < slice) {
            Vec vec;
            Load(vec, buf + (p_rank * slice + idx) * vec_size);
            Store(near[p] + (r.inverse_peer(p) * slice + idx) * vec_size, vec);
        }
    }

    __syncthreads();

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    using namespace ops;
    if (int idx = thread_idx; idx < slice) {
        Vec acc;
        Load(acc, buf + (rank * slice + idx) * vec_size);
        for (int i = 0; i < peers; ++i) {
            Vec tmp;
            Load(tmp, scratch + (i * slice + idx) * vec_size);
            acc = acc + tmp;
        }
        Store(buf + (rank * slice + idx) * vec_size, acc);
        for (int i = 0; i < peers; ++i) {
            const int p = r.get_next_peer(i);
            Store(near[p] + ((peers + rank) * slice + idx) * vec_size, acc);
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }

    __syncthreads();

    for (int p = 0; p < peers; ++p) {
        const int p_rank = r.get_peer_rank(p);
        if (int idx = thread_idx; idx < slice) {
            Vec vec;
            Load(vec, scratch + ((peers + p_rank) * slice + idx) * vec_size);
            Store(buf + (p_rank * slice + idx) * vec_size, vec);
        }
    }
}

template<class T, int vec_size, class Relaxed>
__global__ void Allreduce_Simple_Push_v2(T*                           buf,
                                         T*                           scratch,
                                         Array<T*, kMaxNearPeers>     near_buf,
                                         Array<T*, kMaxNearPeers>     near_scratch,
                                         mscclpp::D2DSemaphoreHandle* semaphores,
                                         int                          rank,
                                         int                          peers,
                                         int                          slice,  // in vec
                                         int                          count,  // in vec
                                         constant<vec_size>,
                                         Relaxed relaxed)
{
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_num = blockDim.x * gridDim.x;

    using Vec = Array<T, vec_size>;

    Rank r{rank, peers};

    for (int i = 0; i < peers; ++i) {
        const int p      = r.get_next_peer(i);
        const int p_rank = r.get_peer_rank(p);
        for (int idx = thread_idx; idx < slice; idx += thread_num) {
            Vec vec;
            Load(vec, buf + (p_rank * slice + idx) * vec_size);
            Store(near_scratch[p] + (r.inverse_peer(p) * slice + idx) * vec_size, vec);
        }
    }

    __syncthreads();

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    using namespace ops;
    for (int idx = thread_idx; idx < slice; idx += thread_num) {
        Vec acc;
        Load(acc, buf + (rank * slice + idx) * vec_size);
        for (int i = 0; i < peers; ++i) {
            Vec tmp;
            Load(tmp, scratch + (i * slice + idx) * vec_size);
            acc = acc + tmp;
        }
        Store(buf + (rank * slice + idx) * vec_size, acc);
        for (int i = 0; i < peers; ++i) {
            const int p = r.get_next_peer(i);
            Store(near_buf[p] + (rank * slice + idx) * vec_size, acc);
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
}

void CudaIpcCommImpl::AllReduceSum(
    const void* sendbuff, void* recvbuff, size_t count, DataType type, int group, cudaStream_t stream)
{
    FT_CHECK(sendbuff == recvbuff);

    void* data = recvbuff;

    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphores = groups_.at(group).d2d_semaphores;

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        const size_t bytesize = sizeof(T) * count;
        if (bytesize <= 1 << 20) {
            constexpr int vec_size      = sizeof(uint2) / sizeof(T);
            const int     slice         = (count / vec_size + n_ranks - 1) / n_ranks;
            constexpr int ctas_per_peer = 4;
            constexpr int threads       = 1024;
            const int     blocks        = (n_ranks - 1) * ctas_per_peer;
            auto          incoming      = (LLPacket*)packet_buff_;
            auto          outgoing      = get_symmetric(incoming, group).uc;
            AllreduceKernel_LL<<<blocks, threads, 0, stream>>>((T*)data,  //
                                                               (T*)data,
                                                               incoming,
                                                               outgoing,
                                                               rank,
                                                               n_ranks - 1,
                                                               slice,
                                                               count / vec_size,
                                                               flag_++,
                                                               constant<ctas_per_peer>{});
        }
        else {
            constexpr int vec_size = sizeof(uint4) / sizeof(T);
            const int     slice    = (count / vec_size + n_ranks - 1) / n_ranks;
            if (bytesize <= kScratchBuffSize && bytesize <= 6 << 20) {
                constexpr int threads = 1024;
                const int     blocks  = std::min(48, (slice + threads - 1) / threads);
                Allreduce_Simple_Push_v2<<<blocks, threads, 0, stream>>>((T*)data,
                                                                         (T*)scratch_buff_,
                                                                         get_symmetric((T*)data, group).uc,
                                                                         get_symmetric((T*)scratch_buff_, group).uc,
                                                                         semaphores,
                                                                         rank,
                                                                         n_ranks - 1,
                                                                         slice,
                                                                         count / vec_size,
                                                                         constant<vec_size>{},
                                                                         std::false_type{});
            }
            else {
                constexpr int threads = 1024;
                const int     blocks  = std::min(48, (slice + threads - 1) / threads);
                Allreduce_Simple_Pull<<<blocks, threads, 0, stream>>>((T*)data,
                                                                      get_symmetric((T*)data, group).uc,
                                                                      semaphores,
                                                                      rank,
                                                                      n_ranks - 1,
                                                                      slice,
                                                                      count / vec_size,
                                                                      constant<vec_size>{},
                                                                      std::false_type{});
            }
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, invoke);
}

}  // namespace turbomind::comm
