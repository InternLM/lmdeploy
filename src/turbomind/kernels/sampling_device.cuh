#pragma once

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11000)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include <curand_kernel.h>

#include "src/turbomind/kernels/sampling_topp_kernels.h"

namespace turbomind {

template<int BlockSize>
struct ProcessedDistributionSampleStorage {
    typename cub::BlockScan<float, BlockSize>::TempStorage scan;
    float threshold;
    int   selected_index;
};

template<typename T, int BlockSize>
__device__ int SampleProcessedDistribution(const T*                                         row,
                                           int                                              kept,
                                           curandState_t*                                   random_state,
                                           ProcessedDistributionSampleStorage<BlockSize>& storage)
{
    const int tid = threadIdx.x;
    if (tid == 0) {
        storage.threshold = curand_uniform(random_state);
    }
    __syncthreads();

    BlockPrefixCallbackOp prefix_op{0.f};
    const int             end = (kept + BlockSize - 1) / BlockSize * BlockSize;
    for (int i = tid; i < end; i += BlockSize) {
        const float probability = i < kept ? static_cast<float>(row[i]) : 0.f;
        float       inclusive_mass{};
        cub::BlockScan<float, BlockSize>(storage.scan).InclusiveSum(probability, inclusive_mass, prefix_op);

        const int count = __syncthreads_count(inclusive_mass > storage.threshold);
        if (count != 0 || i + BlockSize >= end) {
            if (tid == min(BlockSize - count, BlockSize - 1)) {
                storage.selected_index = min(i, kept - 1);
            }
            break;
        }
    }
    __syncthreads();
    return storage.selected_index;
}

}  // namespace turbomind
