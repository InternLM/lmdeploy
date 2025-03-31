// Copyright (c) OpenMMLab. All rights reserved.

#include <stdexcept>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

template<class T, class Accum, int block_dim, int vec_size>
__global__ void RMSNormKernel(T*       dst,
                              int      dst_ld,
                              const T* src,
                              int      src_ld,
                              const T* __restrict__ weights,
                              int   dims,
                              int   num,
                              float eps,
                              float inv_dims)
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
        Ldg(sv, &weights[i]);
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            vec[c] = (T)((float)vec[c] * sum) * sv[c];
            // vec[c] = (T)((float)vec[c] * sum * (float)sv[c]);
        }
        Store(&dst[i], vec);
    }
}

template<class T>
void invokeRMSNorm(
    T* dst, int dst_ld, const T* src, int src_ld, const T* weights, int dims, int num, float eps, cudaStream_t st)
{
    if (num == 0) {
        return;
    }

    constexpr int vec_size = 16 / sizeof(T);

    constexpr int threads = 512;
    const int     blocks  = num;

    RMSNormKernel<T, float, threads, vec_size><<<blocks, threads, 0, st>>>(dst,  //
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

template<class T, class A, int vec_size, int max_dim>
__global__ void QkRMSNormKernel(T*       data,  //
                                int      ld,
                                const T* weight,
                                int      dim,
                                int      n,
                                int      token_num,
                                float    eps,
                                float    inv_dim)
{
    static_assert((max_dim & (max_dim - 1)) == 0);

    constexpr int thr_per_qk = max_dim / vec_size;

    const int bi = (threadIdx.x + blockIdx.x * blockDim.x) / thr_per_qk;
    const int di = threadIdx.x % thr_per_qk * vec_size;
    const int ti = bi / n;
    const int hi = bi % n;

    if (bi >= token_num * n) {
        return;
    }

    data += ti * ld + hi * dim;

    Array<T, vec_size> vec{};
    if (di < dim) {
        Load(vec, &data[di]);
    }

    using namespace ops;
    auto acc = cast<A>(vec);
    acc      = acc * acc;

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += acc[i];
    }

    PRAGMA_UNROLL
    for (int mask = thr_per_qk / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync((uint32_t)-1, sum, mask);
    }

    sum = rsqrtf(sum * inv_dim + eps);

    Array<T, vec_size> w;
    if (di < dim) {
        Ldg(w, &weight[di]);
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            vec[i] = (T)((float)vec[i] * sum) * w[i];
        }
        Store(&data[di], vec);
    }
}

void invokeQkRMSNorm(void*        data,
                     int          ld,
                     const void*  weight,
                     DataType     dtype,
                     int          head_dim,
                     int          n,
                     int          token_num,
                     float        eps,
                     cudaStream_t stream)
{
    auto invoke = [&](auto t, auto max_dim_t) {
        using T = decltype(t);

        constexpr int vec_size   = sizeof(uint4) / sizeof(T);
        constexpr int max_dim    = max_dim_t.value;
        constexpr int thr_per_qk = max_dim / vec_size;

        FT_CHECK(head_dim % vec_size == 0);

        const int threads   = thr_per_qk * n * (int64_t)token_num;
        const int block_dim = 512;
        const int grid_dim  = cdiv(threads, block_dim);

        QkRMSNormKernel<T, float, vec_size, max_dim><<<grid_dim, block_dim, 0, stream>>>(
            (T*)data, ld, (const T*)weight, head_dim, n, token_num, eps, 1.f / head_dim);
    };

    constexpr constant<128> max_dim{};
    FT_CHECK(head_dim <= max_dim);

    switch (dtype) {
        case TYPE_FP16:
            return invoke(half{}, max_dim);
        case TYPE_BF16:
            return invoke(nv_bfloat16{}, max_dim);
        default:
            throw std::runtime_error("not implemented");
    }
}

// r' <- r + (h + b)
// h' <- norm(r') * w
template<class T, class Tacc, int block_dim, int vec_size>
__global__ void BiasResidualRMSNormKernel(T* __restrict__ residual,
                                          T* __restrict__ hidden_states,
                                          const T* __restrict__ weights,
                                          const T* __restrict__ bias,
                                          int   dims,
                                          int   num,
                                          float eps,
                                          float inv_dims)
{
    const int ti = blockIdx.x;
    const int di = threadIdx.x * vec_size;

    if (ti >= num) {
        return;
    }

    residual += dims * ti;
    hidden_states += dims * ti;

    Array<Tacc, vec_size> accum{};

    Array<T, vec_size> r_vec;
    Array<T, vec_size> h_vec;
    Array<T, vec_size> b_vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(r_vec, &residual[i]);
        Load(h_vec, &hidden_states[i]);

        using namespace ops;
        r_vec = r_vec + h_vec;

        if (bias) {
            Ldg(b_vec, &bias[i]);
            r_vec = r_vec + b_vec;
        }

        Store(&residual[i], r_vec);

        Array<Tacc, vec_size> tmp = cast<Tacc>(r_vec);

        accum = accum + tmp * tmp;
    }

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += accum[i];
    }

    using BlockReduce = cub::BlockReduce<Tacc, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
        shared_sum = rsqrtf(sum * inv_dims + eps);
    }

    __syncthreads();

    sum = shared_sum;

    Array<T, vec_size> w_vec;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(r_vec, &residual[i]);
        Ldg(w_vec, &weights[i]);
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            r_vec[c] = (T)((float)r_vec[c] * sum) * w_vec[c];
        }
        Store(&hidden_states[i], r_vec);
    }
}

template<class T>
void invokeBiasResidualRMSNorm(
    T* residual, T* hidden_states, const T* weights, const T* bias, int dims, int num, float eps, cudaStream_t st)
{
    constexpr int vec_size = 16 / sizeof(T);
    constexpr int threads  = 512;
    const int     blocks   = num;

    BiasResidualRMSNormKernel<T, float, threads, vec_size><<<blocks, threads, 0, st>>>(residual,  //
                                                                                       hidden_states,
                                                                                       weights,
                                                                                       bias,
                                                                                       dims,
                                                                                       num,
                                                                                       eps,
                                                                                       1.f / dims);
}

template void invokeBiasResidualRMSNorm(half*        residual,
                                        half*        hidden_states,
                                        const half*  weights,
                                        const half*  bias,
                                        int          dims,
                                        int          num,
                                        float        eps,
                                        cudaStream_t st);

#if ENABLE_BF16
template void invokeBiasResidualRMSNorm(nv_bfloat16*       residual,
                                        nv_bfloat16*       hidden_states,
                                        const nv_bfloat16* weights,
                                        const nv_bfloat16* bias,
                                        int                dims,
                                        int                num,
                                        float              eps,
                                        cudaStream_t       st);
#endif

void invokeResidualBiasRMSNorm(void*        hidden_states,
                               void*        residual,
                               const void*  weights,
                               const void*  bias,
                               DataType     dtype,
                               int          dims,
                               int          num,
                               float        eps,
                               cudaStream_t st)
{
    if (num == 0) {
        return;
    }
    auto invoke = [&](auto t) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        constexpr int threads  = 512;
        const int     blocks   = num;
        BiasResidualRMSNormKernel<T, float, threads, vec_size><<<blocks, threads, 0, st>>>((T*)residual,  //
                                                                                           (T*)hidden_states,
                                                                                           (const T*)weights,
                                                                                           (const T*)bias,
                                                                                           dims,
                                                                                           num,
                                                                                           eps,
                                                                                           1.f / dims);
    };
    switch (dtype) {
        case DataType::TYPE_FP16:
            return invoke(half{});
        case DataType::TYPE_BF16:
            return invoke(nv_bfloat16{});
        default:
            FT_CHECK(0);
    }
}

}  // namespace turbomind
