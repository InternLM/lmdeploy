// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/meta.h"
#include <numeric>

namespace turbomind::comm {

template<class T, int vec_size>
__global__ void ReduceScatterV_Simple_Pull(T*                   buf,  //
                                           Array<T*, kMaxRanks> chns,
                                           SystemSemaphoreInfo* semaphores,
                                           int                  rank,
                                           int                  ranks,
                                           int                  offset,
                                           int                  count,
                                           constant<vec_size>)
{
    const int thread_num = blockDim.x * gridDim.x;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(true);
    sem.Wait(true);

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    for (int i = 1; i < ranks; ++i) {
        const int p   = rank + i < ranks ? rank + i : rank + i - ranks;
        auto      chn = cvta_generic_to_global(chns[p]);
        for (int idx = thread_idx; idx < count; idx += thread_num) {
            Vec acc, tmp;
            Load(tmp, chn + (offset + idx) * vec_size);
            Load(acc, buf + idx * vec_size);
            acc = acc + tmp;
            Store(buf + idx * vec_size, acc);
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::ReduceScatterV(const void*   sendbuff,  //
                                     void*         recvbuff,
                                     const size_t* counts,
                                     DataType      type,
                                     int           group,
                                     cudaStream_t  stream)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    void* data = const_cast<void*>(sendbuff);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T = decltype(t);

        auto symm_ptr = get_symmetric_v2((T*)data, group);

        constexpr int vec_size = sizeof(uint4) / sizeof(T);

        TM_CHECK_EQ(counts[rank] % vec_size, (size_t)0);
        const int count = counts[rank] / vec_size;

        auto offset = std::accumulate(counts, counts + rank, 0LL) / vec_size;

        constexpr int threads  = 1024;
        const int     max_ctas = max_ctas_.apply(48);
        // blocks on all ranks must be same
        std::vector<int> blocks_on_all_rank(n_ranks);
        for (int i = 0; i < n_ranks; ++i) {
            blocks_on_all_rank[i] = std::max((size_t)1, (counts[i] / vec_size + threads - 1) / threads);
        }
        const int blocks = std::min(max_ctas, *std::max_element(blocks_on_all_rank.begin(), blocks_on_all_rank.end()));
        ReduceScatterV_Simple_Pull<<<blocks, threads, 0, stream>>>(
            (T*)recvbuff, symm_ptr.uc, semaphore, rank, n_ranks, offset, count, constant<vec_size>{});
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, invoke);
}

}  // namespace turbomind::comm
