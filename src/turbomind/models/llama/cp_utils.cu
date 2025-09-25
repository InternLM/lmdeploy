// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/cp_utils.h"

namespace turbomind {

template<typename T>
__global__ void
CpReduce(T* out, float* ML, int token_num, int head_num, int size_per_head, int cp_size, int cp_rank, float exp_scale)
{
    __shared__ float scale[WARP_SIZE];
    float            frag_M = -std::numeric_limits<float>::infinity();
    float            frag_L = 0.0f;

    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id == 0 && lane_id < cp_size) {
        const int index = (lane_id * token_num * head_num + token_idx * head_num + head_idx) * 2;
        frag_M          = ML[index];
        frag_L          = ML[index + 1];
    }

    float block_M = frag_M;
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_M = fmaxf(block_M, __shfl_xor_sync(uint32_t(-1), block_M, mask));
    }

    float expdiff_M = exp2f((frag_M - block_M) * exp_scale);

    float block_L = frag_L * expdiff_M;
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        block_L += __shfl_xor_sync(uint32_t(-1), block_L, mask);
    }

    if (warp_id == 0 && lane_id < cp_size) {
        scale[lane_id] = expdiff_M / block_L;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < size_per_head; i += blockDim.x) {
        int index  = token_idx * head_num * size_per_head + head_idx * size_per_head + i;
        out[index] = (T)((float)out[index] * scale[cp_rank]);
    }
}

template<typename T>
void invokeCpReduce(T*           out,
                    float*       ML,
                    int          token_num,
                    int          head_num,
                    int          size_per_head,
                    int          cp_size,
                    int          cp_rank,
                    float        exp_scale,
                    cudaStream_t stream)
{
    TM_CHECK(cp_size <= WARP_SIZE);
    const dim3 block = 4 * WARP_SIZE;
    const dim3 grid(token_num, head_num);
    size_t     smem_size = sizeof(float) * WARP_SIZE;
    CpReduce<<<grid, block, smem_size, stream>>>(
        out, ML, token_num, head_num, size_per_head, cp_size, cp_rank, exp_scale);
    sync_check_cuda_error();
}

template void invokeCpReduce(half*        out,
                             float*       ML,
                             int          token_num,
                             int          head_num,
                             int          size_per_head,
                             int          cp_size,
                             int          cp_rank,
                             float        exp_scale,
                             cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeCpReduce(__nv_bfloat16* out,
                             float*         ML,
                             int            token_num,
                             int            head_num,
                             int            size_per_head,
                             int            cp_size,
                             int            cp_rank,
                             float          exp_scale,
                             cudaStream_t   stream);
#endif

}  // namespace turbomind
