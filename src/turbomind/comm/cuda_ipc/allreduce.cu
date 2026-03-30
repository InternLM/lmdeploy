// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/mscclpp.h"
#include "src/turbomind/comm/cuda_ipc/multimem.cuh"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

using mscclpp::LLPacket;

// reduce-scatter + allgather using LL16Packet
template<class T, class CtasPerPeer>
__global__ void __launch_bounds__(1024, 1) Allreduce_LL16_V2(T*                          dst,
                                                             const T*                    src,
                                                             LLPacket*                   incoming,
                                                             Array<LLPacket*, kMaxRanks> outgoing,
                                                             int                         rank,
                                                             int                         ranks,
                                                             int                         slice,  // padded slice
                                                             int                         count,  // actual count
                                                             uint32_t                    flag,
                                                             CtasPerPeer                 ctas_per_peer)
{

    constexpr int vec_size = sizeof(uint2) / sizeof(T);

    using Vec = Array<T, vec_size>;

    const int bi = blockIdx.x % ctas_per_peer;
    const int p  = [&, i = blockIdx.x / ctas_per_peer + 1] { return rank + i < ranks ? rank + i : rank + i - ranks; }();
    const int n  = min(count, p * slice + slice) - p * slice;

    {  // send slice of `src` to peers  (src -> packet0)
        auto chn = outgoing[p] + rank * slice;
        for (int idx = threadIdx.x + bi * blockDim.x; idx < n; idx += ctas_per_peer * blockDim.x) {
            chn[idx].write(*((const uint2*)src + p * slice + idx), flag);
        }
    }

    // device-wide barrier not required as what we are sending is not what we are going to modify

    {  // recv data | reduce | send results (src -> packet0 -> packet1)
        using namespace ops;
        const int n = min(count, rank * slice + slice) - rank * slice;
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
            Vec vec;
            Load(vec, src + (rank * slice + idx) * vec_size);
            for (int i = 1; i < ranks; ++i) {
                const int p    = rank + i < ranks ? rank + i : rank + i - ranks;
                uint2     data = incoming[p * slice + idx].read(flag);
                vec            = vec + (Vec&)data;
            }
            Store(dst + (rank * slice + idx) * vec_size, vec);
            for (int i = 1; i < ranks; ++i) {
                const int p = rank + i < ranks ? rank + i : rank + i - ranks;
                outgoing[p][(ranks + rank) * slice + idx].write((uint2&)vec, flag);
            }
        }
    }

    {  // recv results (packet1 -> dst)
        incoming += (ranks + p) * slice;
        dst += p * slice * vec_size;
        // ! note that `dst` MUST have same partition as we are sending `src`
        for (int idx = threadIdx.x + bi * blockDim.x; idx < n; idx += ctas_per_peer * blockDim.x) {
            uint2 data = incoming[idx].read(flag);
            Store(dst + idx * vec_size, (Vec&)data);
        }
    }
}

// Modified from
// https://github.com/microsoft/mscclpp/blob/591276f9d07d2df8e2a45a16738e27867e468ca3/test/mscclpp-test/allreduce_test.cu#L963
template<class T, int vec_size, class Relaxed>
__global__ void Allreduce_Simple_Pull(T*                   buf,
                                      Array<T*, kMaxRanks> chns,
                                      SystemSemaphoreInfo* semaphores,
                                      int                  rank,
                                      int                  ranks,
                                      int                  slice,
                                      int                  count,
                                      constant<vec_size>,
                                      Relaxed relaxed)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    for (int i = 1; i < ranks; ++i) {
        const int p   = rank + i < ranks ? rank + i : rank + i - ranks;
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

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    for (int i = 1; i < ranks; ++i) {
        const int p     = rank + i < ranks ? rank + i : rank + i - ranks;
        const int first = p * slice;
        const int last  = min(count, first + slice);
        auto      chn   = cvta_generic_to_global(chns[p]);
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Vec vec;
            Load(vec, chn + idx * vec_size);
            Store(buf + idx * vec_size, vec);
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, int vec_size, class Relaxed>
__global__ void Allreduce_Simple_Push_v3(T*                   buf,
                                         T*                   scratch,
                                         Array<T*, kMaxRanks> symm_buf,
                                         Array<T*, kMaxRanks> symm_scratch,
                                         SystemSemaphoreInfo* semaphores,
                                         int                  rank,
                                         int                  ranks,
                                         int                  slice,  // in vec
                                         int                  count,  // in vec
                                         constant<vec_size>,
                                         Relaxed relaxed)
{
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_num = blockDim.x * gridDim.x;

    using Vec = Array<T, vec_size>;

    for (int i = 1; i < ranks; ++i) {
        const int p = rank + i < ranks ? rank + i : rank + i - ranks;
        const int n = min(count, p * slice + slice) - p * slice;
        for (int idx = thread_idx; idx < n; idx += thread_num) {
            Vec vec;
            Load(vec, buf + (p * slice + idx) * vec_size);
            Store(symm_scratch[p] + (rank * slice + idx) * vec_size, vec);
        }
    }

    __syncthreads();

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    using namespace ops;
    const int n = min(count, rank * slice + slice) - rank * slice;
    for (int idx = thread_idx; idx < n; idx += thread_num) {
        Vec acc;
        Load(acc, buf + (rank * slice + idx) * vec_size);
        for (int i = 1; i < ranks; ++i) {
            const int p = rank + i < ranks ? rank + i : rank + i - ranks;
            Vec       tmp;
            Load(tmp, scratch + (p * slice + idx) * vec_size);
            acc = acc + tmp;
        }
        Store(buf + (rank * slice + idx) * vec_size, acc);
        for (int i = 1; i < ranks; ++i) {
            const int p = rank + i < ranks ? rank + i : rank + i - ranks;
            Store(symm_buf[p] + (rank * slice + idx) * vec_size, acc);
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, int vec_size, class Relaxed>
__global__ void Allreduce_NVLS_V2(
    T* mc_buf, SystemSemaphoreInfo* semaphores, int ranks, int first, int last, constant<vec_size>, Relaxed relaxed)
{
#if TURBOMIND_ARCH_SM90
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    for (int idx = first + thread_idx; idx < last; idx += thread_num) {
        Vec vsum = multimem_ld_reduce_sum((const Vec*)(mc_buf + idx * vec_size));
        multimem_st(mc_buf + idx * vec_size, vsum);
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
#endif
}

void CudaIpcCommImpl::AllReduceSum(
    const void* sendbuff, void* recvbuff, size_t count, DataType type, int group, cudaStream_t stream)
{
    FT_CHECK(sendbuff == recvbuff);

    void* data = recvbuff;

    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        const size_t bytesize = sizeof(T) * count;

        auto symm_ptr = get_symmetric_v2((T*)data, group);

        if (symm_ptr.mc) {
            constexpr int vec_size = sizeof(uint4) / sizeof(T);
            constexpr int threads  = 1024;
            const int     slice    = (count / vec_size + n_ranks - 1) / n_ranks;
            const int     first    = rank * slice;
            const int     last     = std::min<int>(count / vec_size, first + slice);
            const int     max_ctas = max_ctas_.apply(8);
            const int     blocks   = std::min(max_ctas, (slice + threads - 1) / threads);
            Allreduce_NVLS_V2<<<blocks, threads, 0, stream>>>(symm_ptr.mc,  //
                                                              semaphore,
                                                              n_ranks,
                                                              first,
                                                              last,
                                                              constant<vec_size>{},
                                                              std::false_type{});
        }
#if 1
        else if (round_up(bytesize, 2 * n_ranks * sizeof(LLPacket)) <= std::min<size_t>(1 << 20, kPacketBuffSize)) {
            constexpr int vec_size      = sizeof(uint2) / sizeof(T);
            const int     slice         = (count / vec_size + n_ranks - 1) / n_ranks;
            constexpr int ctas_per_peer = 4;
            constexpr int threads       = 1024;
            const int     blocks        = (n_ranks - 1) * ctas_per_peer;
            auto          incoming      = (LLPacket*)packet_buff_;
            auto          outgoing      = get_symmetric_v2(incoming, group).uc;
            Allreduce_LL16_V2<<<blocks, threads, 0, stream>>>((T*)data,  //
                                                              (T*)data,
                                                              incoming,
                                                              outgoing,
                                                              rank,
                                                              n_ranks,
                                                              slice,
                                                              count / vec_size,
                                                              flag_++,
                                                              constant<ctas_per_peer>{});
        }
#endif
        else if (round_up(bytesize, n_ranks * sizeof(uint4)) <= std::min<size_t>(6 << 20, kScratchBuffSize)) {
            constexpr int vec_size = sizeof(uint4) / sizeof(T);
            constexpr int threads  = 1024;
            const int     slice    = (count / vec_size + n_ranks - 1) / n_ranks;
            const int     max_ctas = max_ctas_.apply(48);
            const int     blocks   = std::min(max_ctas, (slice + threads - 1) / threads);
            Allreduce_Simple_Push_v3<<<blocks, threads, 0, stream>>>((T*)data,
                                                                     (T*)scratch_buff_,
                                                                     symm_ptr.uc,
                                                                     get_symmetric_v2((T*)scratch_buff_, group).uc,
                                                                     semaphore,
                                                                     rank,
                                                                     n_ranks,
                                                                     slice,
                                                                     count / vec_size,
                                                                     constant<vec_size>{},
                                                                     std::false_type{});
        }
        else {
            constexpr int vec_size = sizeof(uint4) / sizeof(T);
            constexpr int threads  = 1024;
            const int     slice    = (count / vec_size + n_ranks - 1) / n_ranks;
            const int     max_ctas = max_ctas_.apply(48);
            const int     blocks   = std::min(max_ctas, (slice + threads - 1) / threads);
            Allreduce_Simple_Pull<<<blocks, threads, 0, stream>>>((T*)data,
                                                                  symm_ptr.uc,
                                                                  semaphore,
                                                                  rank,
                                                                  n_ranks,
                                                                  slice,
                                                                  count / vec_size,
                                                                  constant<vec_size>{},
                                                                  std::false_type{});
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, invoke);
}

}  // namespace turbomind::comm
