#pragma once

#include "src/turbomind/comm/common.h"
#include "src/turbomind/kernels/core/array_ops.h"

#include "mscclpp/concurrency_device.hpp"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

template<int vec_size, class T>
__device__ void local_reduce_scatter(T*                     data,
                                     SmChannels             channels,  //
                                     mscclpp::DeviceSyncer* device_syncer,
                                     int                    rank,
                                     int                    world_size,
                                     size_t                 count)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int n_peer     = world_size - 1;

    if (thread_idx < n_peer) {
        channels[thread_idx].signal();
        channels[thread_idx].wait();
    }
    device_syncer->sync(block_num);

    count /= vec_size;
    const size_t offset = rank * count;

    using Vec   = Array<T, vec_size>;
    using Proxy = proxy_type<Vec>;

    for (int p = 0; p < n_peer; ++p) {
        const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
        for (size_t idx = thread_idx; idx < count; idx += thread_num) {
            auto proxy = channels[peer].read<Proxy>(offset + idx);
            Vec& vec1  = reinterpret_cast<Vec&>(proxy);
            Vec  vec0;
            Load(vec0, data + (idx + offset) * vec_size);
            for (int c = 0; c < vec_size; ++c) {
                vec0[c] += vec1[c];
            }
            Store(data + (idx + offset) * vec_size, vec0);
        }
    }
}

}  // namespace turbomind