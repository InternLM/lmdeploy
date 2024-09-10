#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11000)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/utils/constant.h"

namespace turbomind {

template<typename T, int BLOCK_SIZE>
__global__ void sampling(const T*       logits,
                         const int      stride,
                         const int*     indices,
                         const int*     kept,
                         curandState_t* curandstate,
                         int*           output_ids,
                         int*           sequence_length,
                         float*         sampled_logprobs,
                         uint32_t*      sampled_indexes,
                         uint32_t*      sampled_nums)
{
    int tid      = threadIdx.x;
    int batch_id = blockIdx.x;
    int n        = kept[batch_id];

    logits += stride * batch_id;
    indices += stride * batch_id;

    __shared__ float rand_num_s;
    __shared__ int   selected;
    if (tid == 0) {
        rand_num_s = curand_uniform(curandstate + batch_id);
    }
    __syncthreads();

    typedef cub::BlockScan<float, BLOCK_SIZE>  BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    float                 local_rand = rand_num_s;
    float                 prefix_sum = 0.f;
    BlockPrefixCallbackOp prefix_op{0};
    int                   end = (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    for (int i = tid; i < end; i += BLOCK_SIZE) {
        float thread_logit = (i < n) ? static_cast<float>(logits[i]) : 0.f;
        BlockScan(temp_storage).InclusiveSum(thread_logit, prefix_sum, prefix_op);
        auto count = __syncthreads_count(prefix_sum > local_rand);
        if (count != 0 || (i + BLOCK_SIZE) >= end) {
            if (tid == min(BLOCK_SIZE - count, BLOCK_SIZE - 1)) {
                selected             = min(i, n - 1);
                output_ids[batch_id] = indices[selected];

                if (sequence_length != nullptr) {
                    sequence_length[batch_id] += 1;
                }
            }
            break;
        }
    }

    if (sampled_logprobs != nullptr && sampled_indexes != nullptr && sampled_nums != nullptr) {
        __syncthreads();
        sampled_logprobs += batch_id * kMaxLogProb;
        sampled_indexes += batch_id * kMaxLogProb;
        int end = min(n, kMaxLogProb);
        for (int i = tid; i < end; i += BLOCK_SIZE) {
            sampled_logprobs[i] = logf(logits[i]);
            sampled_indexes[i]  = indices[i];
        }
        if (n > kMaxLogProb && selected >= kMaxLogProb) {
            if ((kMaxLogProb - 1 + BLOCK_SIZE - tid) % BLOCK_SIZE == 0) {
                sampled_logprobs[kMaxLogProb - 1] = logf(logits[selected]);
                sampled_indexes[kMaxLogProb - 1]  = indices[selected];
            }
        }
        sampled_nums[batch_id] = min(n, kMaxLogProb);
    }
}

template<typename T>
void invokeSampling(SamplingParams& params, cudaStream_t stream)
{
    const int grid  = params.batch_size;
    const int block = 256;
    sampling<T, block><<<grid, block, 0, stream>>>((T*)params.logits,
                                                   params.stride,
                                                   params.indices,
                                                   params.kept,
                                                   params.curandstate,
                                                   params.output_ids,
                                                   params.sequence_length,
                                                   params.sampled_logprobs,
                                                   params.sampled_indexes,
                                                   params.sampled_nums);
}

template void invokeSampling<float>(SamplingParams& params, cudaStream_t stream);

}  // namespace turbomind
