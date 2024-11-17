// Copyright (c) OpenMMLab. All rights reserved.

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

template<class T, class Accum, int block_dim, int vec_size>
__global__ void RMSNormKernel(
    T* dst, int dst_ld, const T* src, int src_ld, const T* weights, int dims, int num, float eps, float inv_dims)
{
    const int ti = blockIdx.x;
    const int di = threadIdx.x * vec_size;

    if (ti >= num) {
        return;
    }

    src += src_ld * ti;

    Array<Accum, vec_size> accum{};
    Array<T, vec_size>     vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Array<Accum, vec_size> tmp = cast<Accum>(vec);
        using namespace ops;
        accum = accum + tmp * tmp;
    }

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += accum[i];
    }

    using BlockReduce = cub::BlockReduce<Accum, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
        shared_sum = rsqrtf(sum * inv_dims + eps);
    }

    __syncthreads();

    sum = shared_sum;

    dst += dst_ld * ti;

    Array<T, vec_size> sv;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Array<Accum, vec_size> tmp = cast<Accum>(vec);
        Load(sv, &weights[i]);
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            tmp[c] *= (float)sv[c] * sum;
        }
        Store(&dst[i], cast<T>(tmp));
    }
}

template<class T>
void invokeRMSNorm(
    T* dst, int dst_ld, const T* src, int src_ld, const T* weights, int dims, int num, float eps, cudaStream_t st)
{
    constexpr int threads = 256;
    const int     blocks  = num;

    RMSNormKernel<T, float, threads, 8><<<blocks, threads, 0, st>>>(dst,  //
                                                                    dst_ld,
                                                                    src,
                                                                    src_ld,
                                                                    weights,
                                                                    dims,
                                                                    num,
                                                                    eps,
                                                                    1.f / dims);
}

template void invokeRMSNorm(half*        dst,
                            int          dst_ld,
                            const half*  src,
                            int          src_ld,
                            const half*  weights,
                            int          dims,
                            int          num,
                            float        eps,
                            cudaStream_t st);
#if ENABLE_BF16
template void invokeRMSNorm(nv_bfloat16*       dst,
                            int                dst_ld,
                            const nv_bfloat16* src,
                            int                src_ld,
                            const nv_bfloat16* weights,
                            int                dims,
                            int                num,
                            float              eps,
                            cudaStream_t       st);
#endif

}  // namespace turbomind