
#pragma once


#include "src/turbomind/comm/common.h"

#include "mscclpp/concurrency_device.hpp"

namespace turbomind {

__device__ inline void local_allgather(SmChannels&            channels,  //
                                       mscclpp::DeviceSyncer* device_syncer,
                                       int                    rank,
                                       int                    world_size,
                                       size_t                 size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;

    const int n_peer = world_size - 1;

    if (tid < n_peer) {
        channels[tid].signal();
        channels[tid].wait();
    }

    device_syncer->sync(gridDim.x);

    for (int i = 0; i < n_peer; ++i) {
        const int    peer_idx        = i + rank < n_peer ? i + rank : i + rank - n_peer;
        const int    remote_rank_idx = peer_idx < rank ? peer_idx : peer_idx + 1;
        const size_t offset          = size * remote_rank_idx;
        channels[peer_idx].get<16, false>(offset, size, tid, thread_num);
    }
}

}  // namespace turbomind