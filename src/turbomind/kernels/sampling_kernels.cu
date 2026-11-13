#include "src/turbomind/kernels/sampling_device.cuh"
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T, int BLOCK_SIZE>
__global__ void sampling(const T*       probabilities,
                         const int      stride,
                         const int*     indices,
                         const int*     kept,
                         curandState_t* curandstate,
                         const int*     curandstate_indices,
                         const bool*    sample_mask,
                         int*           selected_tokens,
                         T*             sampled_logprobs,
                         int*           sampled_indexes,
                         int*           sampled_nums)
{
    const int batch_id = blockIdx.x;

    if (sample_mask != nullptr && !sample_mask[batch_id]) {
        return;
    }

    const int tid = threadIdx.x;
    const int n   = kept[batch_id];

    probabilities += stride * batch_id;
    indices += stride * batch_id;

    __shared__ ProcessedDistributionSampleStorage<BLOCK_SIZE> storage;
    const int selected = SampleProcessedDistribution<T, BLOCK_SIZE>(
        probabilities, n, curandstate + curandstate_indices[batch_id], storage);
    if (tid == 0) {
        selected_tokens[batch_id] = indices[selected];
    }

    if (sampled_logprobs != nullptr && sampled_indexes != nullptr && sampled_nums != nullptr) {
        __syncthreads();
        sampled_logprobs += batch_id * kMaxLogProb;
        sampled_indexes += batch_id * kMaxLogProb;
        int end = min(n, kMaxLogProb);
        for (int i = tid; i < end; i += BLOCK_SIZE) {
            sampled_logprobs[i] = logf(probabilities[i]);
            sampled_indexes[i]  = indices[i];
        }
        if (n > kMaxLogProb && selected >= kMaxLogProb) {
            if ((kMaxLogProb - 1 + BLOCK_SIZE - tid) % BLOCK_SIZE == 0) {
                sampled_logprobs[kMaxLogProb - 1] = logf(probabilities[selected]);
                sampled_indexes[kMaxLogProb - 1]  = indices[selected];
            }
        }
        sampled_nums[batch_id] = min(n, kMaxLogProb);
    }
}

template<typename T>
void invokeSampling(SamplingParams& params, cudaStream_t stream)
{
    if (params.batch_size == 0) {
        return;
    }

    const int grid  = params.batch_size;
    const int block = 256;
    sampling<T, block><<<grid, block, 0, stream>>>((const T*)params.probabilities,
                                                   params.stride,
                                                   params.indices,
                                                   params.kept,
                                                   params.curandstate,
                                                   params.curandstate_indices,
                                                   params.sample_mask,
                                                   params.selected_tokens,
                                                   (T*)params.sampled_logprobs,
                                                   params.sampled_indexes,
                                                   params.sampled_nums);
    TM_CUDA_CHECK(cudaGetLastError());
}

template void invokeSampling<float>(SamplingParams& params, cudaStream_t stream);

}  // namespace turbomind
