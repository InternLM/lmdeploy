// Copyright (c) OpenMMLab. All rights reserved.

#include <stdexcept>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/kernels/norm/rms_norm_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

namespace kernel {

template<bool ZeroCentered, class T, class Accum, int block_dim, int vec_size>
__global__ void RMSNorm(T*       dst,
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
            vec[c] = ApplyRMSnorm<ZeroCentered>(vec[c], sum, sv[c]);
        }
        Store(&dst[i], vec);
    }
}

}  // namespace kernel

void invokeRMSNorm(Tensor& out, const Tensor& x, const Tensor& w, float eps, bool zero_centered, cudaStream_t st)
{
    if (x.size() == 0) {
        return;
    }

    TM_CHECK(x.ndim() == 2);
    TM_CHECK(out.shape() == x.shape());
    TM_CHECK(out.dtype() == x.dtype());
    TM_CHECK(w.dtype() == x.dtype() && w.shape(-1) == x.shape(-1));

    auto invoke = [&](auto t) {
        using T = decltype(t);

        const auto [num, dim] = x.shapes(0, 1);

        constexpr int vec_size = 16 / sizeof(T);

        constexpr int threads = 512;
        const int     blocks  = num;

        auto launch = [&](auto zero_centered_c) {
            constexpr bool ZeroCentered = decltype(zero_centered_c)::value;
            kernel::RMSNorm<ZeroCentered, T, float, threads, vec_size>
                <<<blocks, threads, 0, st>>>((T*)out.raw_data(),  //
                                             out.stride(0),
                                             (const T*)x.raw_data(),
                                             x.stride(0),
                                             (const T*)w.raw_data(),
                                             dim,
                                             num,
                                             eps,
                                             1.f / dim);
        };

        if (zero_centered) {
            launch(constant<true>{});
        }
        else {
            launch(constant<false>{});
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(x.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

namespace kernel {

template<class T, class A, int vec_size, int max_dim, bool ZeroCentered>
__global__ void QkRMSNorm(T*       qkv,
                          int      ld,
                          const T* q_weight,
                          const T* k_weight,
                          int      dim,
                          int      q_head_num,
                          int      k_head_num,
                          int      token_num,
                          int      q_block_num,
                          float    eps,
                          float    inv_dim,
                          constant<ZeroCentered>)
{
    static_assert((max_dim & (max_dim - 1)) == 0);

    constexpr int threads_per_head = max_dim / vec_size;

    const bool is_k        = blockIdx.x >= q_block_num;
    const int  block_idx   = is_k ? blockIdx.x - q_block_num : blockIdx.x;
    const int  head_num    = is_k ? k_head_num : q_head_num;
    const int  head_offset = is_k ? q_head_num : 0;
    const T*   weight      = is_k ? k_weight : q_weight;

    const int bi = (threadIdx.x + block_idx * blockDim.x) / threads_per_head;
    const int di = threadIdx.x % threads_per_head * vec_size;

    if (bi >= token_num * head_num) {
        return;
    }

    const int ti = bi / head_num;
    const int hi = bi % head_num;

    qkv += ti * ld + (head_offset + hi) * dim;

    Array<T, vec_size> vec{};
    if (di < dim) {
        Load(vec, &qkv[di]);
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
    for (int mask = threads_per_head / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync((uint32_t)-1, sum, mask);
    }

    sum = rsqrtf(sum * inv_dim + eps);

    Array<T, vec_size> w;
    if (di < dim) {
        Ldg(w, &weight[di]);
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            vec[i] = ApplyRMSnorm<ZeroCentered>(vec[i], sum, w[i]);
        }
        Store(&qkv[di], vec);
    }
}

}  // namespace kernel

void invokeQkRMSNorm(Tensor&       qkv,
                     const Tensor& q_weight,
                     const Tensor& k_weight,
                     int           q_head_num,
                     int           k_head_num,
                     float         eps,
                     bool          zero_centered,
                     cudaStream_t  stream)
{
    TM_CHECK(qkv.ndim() == 3);

    const int token_num = qkv.shape(0);
    const int head_dim  = qkv.shape(2);

    TM_CHECK(qkv.stride(1) == head_dim);

    auto data   = qkv.raw_data();
    auto stride = qkv.stride(0);

    auto invoke = [&](auto t) {
        using T = decltype(t);

        auto launch = [&](auto max_dim_c, auto zero_centered_c) {
            constexpr int kMaxDim = std::decay_t<decltype(max_dim_c)>::value;
            TM_CHECK_LE(head_dim, kMaxDim);

            constexpr int vec_size         = sizeof(uint4) / sizeof(T);
            constexpr int threads_per_head = kMaxDim / vec_size;
            constexpr int block_dim        = 512;

            TM_CHECK(head_dim % vec_size == 0);

            const int q_block_num = cdiv(token_num * q_head_num * threads_per_head, block_dim);
            const int k_block_num = cdiv(token_num * k_head_num * threads_per_head, block_dim);
            const int grid_dim    = q_block_num + k_block_num;

            kernel::QkRMSNorm<T, float, vec_size, kMaxDim>
                <<<grid_dim, block_dim, 0, stream>>>((T*)data,
                                                     stride,
                                                     (const T*)q_weight.raw_data(),
                                                     (const T*)k_weight.raw_data(),
                                                     head_dim,
                                                     q_head_num,
                                                     k_head_num,
                                                     token_num,
                                                     q_block_num,
                                                     eps,
                                                     1.f / head_dim,
                                                     zero_centered_c);
        };

        if (head_dim <= 128) {
            if (zero_centered) {
                launch(constant<128>{}, constant<true>{});
            }
            else {
                launch(constant<128>{}, constant<false>{});
            }
        }
        else {
            if (zero_centered) {
                launch(constant<256>{}, constant<true>{});
            }
            else {
                launch(constant<256>{}, constant<false>{});
            }
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(qkv.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

// r' <- r + (h + b)
// h' <- norm(r') * w
template<bool ZeroCentered, class T, class Tacc, int block_dim, int vec_size>
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
            r_vec[c] = kernel::ApplyRMSnorm<ZeroCentered>(r_vec[c], sum, w_vec[c]);
        }
        Store(&hidden_states[i], r_vec);
    }
}

template<class T>
void invokeBiasResidualRMSNorm(T*           residual,
                               T*           hidden_states,
                               const T*     weights,
                               const T*     bias,
                               int          dims,
                               int          num,
                               float        eps,
                               bool         zero_centered,
                               cudaStream_t st)
{
    constexpr int vec_size = 16 / sizeof(T);
    constexpr int threads  = 512;
    const int     blocks   = num;

    auto launch = [&](auto zero_centered_c) {
        constexpr bool ZeroCentered = decltype(zero_centered_c)::value;
        BiasResidualRMSNormKernel<ZeroCentered, T, float, threads, vec_size><<<blocks, threads, 0, st>>>(residual,  //
                                                                                                         hidden_states,
                                                                                                         weights,
                                                                                                         bias,
                                                                                                         dims,
                                                                                                         num,
                                                                                                         eps,
                                                                                                         1.f / dims);
    };

    if (zero_centered) {
        launch(constant<true>{});
    }
    else {
        launch(constant<false>{});
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

template void invokeBiasResidualRMSNorm(half*        residual,
                                        half*        hidden_states,
                                        const half*  weights,
                                        const half*  bias,
                                        int          dims,
                                        int          num,
                                        float        eps,
                                        bool         zero_centered,
                                        cudaStream_t st);

#if ENABLE_BF16
template void invokeBiasResidualRMSNorm(nv_bfloat16*       residual,
                                        nv_bfloat16*       hidden_states,
                                        const nv_bfloat16* weights,
                                        const nv_bfloat16* bias,
                                        int                dims,
                                        int                num,
                                        float              eps,
                                        bool               zero_centered,
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
                               bool         zero_centered,
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
        auto          launch   = [&](auto zero_centered_c) {
            constexpr bool ZeroCentered = decltype(zero_centered_c)::value;
            BiasResidualRMSNormKernel<ZeroCentered, T, float, threads, vec_size>
                <<<blocks, threads, 0, st>>>((T*)residual,  //
                                             (T*)hidden_states,
                                             (const T*)weights,
                                             (const T*)bias,
                                             dims,
                                             num,
                                             eps,
                                             1.f / dims);
        };

        if (zero_centered) {
            launch(constant<true>{});
        }
        else {
            launch(constant<false>{});
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T, class B, int vec_size>
__global__ void biasKernel(T* data, const B* bias, int num, int dim)
{
    int ti = blockIdx.x;
    int di = threadIdx.x * vec_size;

    Array<B, vec_size> b;
    Ldg(b, bias + di);

    Array<T, vec_size> x;
    Load(x, data + ti * dim + di);
    using namespace ops;
    x = x + cast<T>(b);
    Store(data + ti * dim + di, x);
}

void ApplyBias(Tensor& data, const Tensor& bias, cudaStream_t st)
{
    if (!bias) {
        return;
    }

    const int num = data.shape(0);
    const int dim = data.shape(1);

    TM_CHECK_EQ(dim, bias.shape(-1));

    auto invoke0 = [&](auto t) {
        using T      = decltype(t);
        auto invoke1 = [&](auto b) {
            using B                = decltype(b);
            constexpr int vec_size = sizeof(uint4) / std::max(sizeof(T), sizeof(B));
            TM_CHECK(dim % vec_size == 0);
            const int blocks  = num;
            const int threads = dim / vec_size;
            TM_CHECK_LE(threads, 1024);
            biasKernel<T, B, vec_size><<<blocks, threads, 0, st>>>(data.data<T>(),  //
                                                                   bias.data<B>(),
                                                                   num,
                                                                   dim);
        };
        if constexpr (data_type_v<T> == kFloat) {
            TM_DISPATCH_PRIMARY_DTYPES(bias.dtype(), invoke1);
        }
        else {  // skip mixing half and bf16
            invoke1(t);
        }
    };
    TM_DISPATCH_DTYPES(data.dtype(), invoke0, float, half, nv_bfloat16);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T, int vec_size>
__global__ void biasKernel(T* data, const T* bias, const int* offsets, int num, int dim, int groups, float scale)
{
    int ti = blockIdx.x;
    int di = threadIdx.x * vec_size;

    __shared__ int s_idx;

    if (int tid = threadIdx.x; tid < groups) {
        int b = __ldg(&offsets[tid]);
        int e = __ldg(&offsets[tid + 1]);
        if (b <= ti && ti < e) {
            s_idx = tid;
        }
    }

    data += ti * dim;

    __syncthreads();

    bias += s_idx * dim;

    if (di >= dim) {
        return;
    }

    Array<T, vec_size> b;
    Ldg(b, bias + di);

    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        b[i] = (T)((float)b[i] * scale);
    }

    Array<T, vec_size> x;
    Load(x, data + di);

    using namespace ops;
    x = x + b;

    Store(data + di, x);
}

void ApplyBias(Tensor& data, const Tensor& bias, const Buffer_<int>& offsets, float scale, cudaStream_t st)
{
    if (!bias) {
        return;
    }

    const int num    = data.shape(0);
    const int dim    = data.shape(1);
    const int groups = offsets.size() - 1;

    TM_CHECK_EQ(dim, bias.shape(-1));

    // std::cout << data << " " << bias << " " << offsets << "\n";

    auto invoke = [&](auto t) {
        using T = decltype(t);

        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        TM_CHECK(dim % vec_size == 0);

        const int blocks  = num;
        const int threads = std::max(dim / vec_size, groups);

        TM_CHECK_LE(threads, 1024);

        biasKernel<T, vec_size><<<blocks, threads, 0, st>>>(data.data<T>(),  //
                                                            bias.data<T>(),
                                                            offsets.data(),
                                                            num,
                                                            dim,
                                                            offsets.size() - 1,
                                                            scale);
    };

    TM_DISPATCH_PRIMARY_DTYPES(data.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
