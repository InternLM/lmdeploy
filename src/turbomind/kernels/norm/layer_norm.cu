// Copyright (c) OpenMMLab. All rights reserved.

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/layer_norm.h"

namespace turbomind {

namespace kernel {

struct SumPair {
    float              s;
    float              sq;
    __device__ SumPair operator+(const SumPair& other) const
    {
        return {s + other.s, sq + other.sq};
    }
};

template<class T, class Tacc, int block_dim, int vec_size, bool kHasBias>
__global__ void LayerNorm(T*       dst,
                          int      dst_ld,
                          const T* src,
                          int      src_ld,
                          const T* __restrict__ weight,
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

    src += src_ld * ti;
    dst += dst_ld * ti;

    Array<Tacc, vec_size> sum_v{};
    Array<Tacc, vec_size> sq_v{};
    Array<T, vec_size>    vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Array<Tacc, vec_size> tmp = cast<Tacc>(vec);
        using namespace ops;
        sum_v = sum_v + tmp;
        sq_v  = sq_v + tmp * tmp;
    }

    SumPair pair{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        pair.s += sum_v[i];
        pair.sq += sq_v[i];
    }

    using BlockReduce = cub::BlockReduce<SumPair, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    pair = BlockReduce{temp_storage}.Sum(pair);

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    if (threadIdx.x == 0) {
        const float mean = pair.s * inv_dims;
        const float var  = fmaxf(pair.sq * inv_dims - mean * mean, 0.f);
        shared_mean      = mean;
        shared_inv_std   = rsqrtf(var + eps);
    }

    __syncthreads();

    const float mean    = shared_mean;
    const float inv_std = shared_inv_std;

    Array<T, vec_size> w_vec;
    Array<T, vec_size> b_vec;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Ldg(w_vec, &weight[i]);
        if constexpr (kHasBias) {
            Ldg(b_vec, &bias[i]);
        }
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            float v = ((float)vec[c] - mean) * inv_std * (float)w_vec[c];
            if constexpr (kHasBias) {
                v += (float)b_vec[c];
            }
            vec[c] = (T)v;
        }
        Store(&dst[i], vec);
    }
}

// r' <- r + h + residual_bias
// h' <- LayerNorm(r') * norm_weight + norm_bias
template<class T, class Tacc, int block_dim, int vec_size, bool kHasNormBias, bool kHasResidualBias>
__global__ void ResidualBiasLayerNorm(T* __restrict__ hidden_states,
                                      T* __restrict__ residual,
                                      const T* __restrict__ norm_weight,
                                      const T* __restrict__ norm_bias,
                                      const T* __restrict__ residual_bias,
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

    Array<Tacc, vec_size> sum_v{};
    Array<Tacc, vec_size> sq_v{};
    Array<T, vec_size>    r_vec;
    Array<T, vec_size>    h_vec;
    Array<T, vec_size>    rb_vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(r_vec, &residual[i]);
        Load(h_vec, &hidden_states[i]);

        using namespace ops;
        r_vec = r_vec + h_vec;
        if constexpr (kHasResidualBias) {
            Ldg(rb_vec, &residual_bias[i]);
            r_vec = r_vec + rb_vec;
        }

        Store(&residual[i], r_vec);

        Array<Tacc, vec_size> tmp = cast<Tacc>(r_vec);
        sum_v                     = sum_v + tmp;
        sq_v                      = sq_v + tmp * tmp;
    }

    SumPair pair{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        pair.s += sum_v[i];
        pair.sq += sq_v[i];
    }

    using BlockReduce = cub::BlockReduce<SumPair, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    pair = BlockReduce{temp_storage}.Sum(pair);

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    if (threadIdx.x == 0) {
        const float mean = pair.s * inv_dims;
        const float var  = fmaxf(pair.sq * inv_dims - mean * mean, 0.f);
        shared_mean      = mean;
        shared_inv_std   = rsqrtf(var + eps);
    }

    __syncthreads();

    const float mean    = shared_mean;
    const float inv_std = shared_inv_std;

    Array<T, vec_size> w_vec;
    Array<T, vec_size> nb_vec;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(r_vec, &residual[i]);
        Ldg(w_vec, &norm_weight[i]);
        if constexpr (kHasNormBias) {
            Ldg(nb_vec, &norm_bias[i]);
        }
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            float v = ((float)r_vec[c] - mean) * inv_std * (float)w_vec[c];
            if constexpr (kHasNormBias) {
                v += (float)nb_vec[c];
            }
            r_vec[c] = (T)v;
        }
        Store(&hidden_states[i], r_vec);
    }
}

}  // namespace kernel

void invokeLayerNorm(
    Tensor& out, const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, cudaStream_t stream)
{
    if (x.size() == 0) {
        return;
    }

    TM_CHECK(x.ndim() == 2);
    TM_CHECK(out.shape() == x.shape());
    TM_CHECK(out.dtype() == x.dtype());
    TM_CHECK(weight.dtype() == x.dtype() && weight.shape(-1) == x.shape(-1));
    if (bias) {
        TM_CHECK(bias.dtype() == x.dtype() && bias.shape(-1) == x.shape(-1));
    }

    auto invoke = [&](auto t) {
        using T = decltype(t);

        const int num = x.shape(0);
        const int dim = x.shape(1);

        constexpr int vec_size = 16 / sizeof(T);
        const int     blocks   = num;

        TM_CHECK(dim % vec_size == 0) << "dim=" << dim << " must be divisible by vec_size=" << vec_size;

        auto launch = [&](auto has_bias_c) {
            // Redeclare these as constexpr inside this nested lambda. MSVC odr-uses
            // (captures) constexpr locals read from an enclosing lambda and then
            // refuses them as non-type template arguments; see rms_norm.cu (RMSNormQK).
            constexpr int  kThreads = 512;
            constexpr int  kVecSize = 16 / sizeof(T);
            constexpr bool kHasBias = decltype(has_bias_c)::value;
            kernel::LayerNorm<T, float, kThreads, kVecSize, kHasBias>
                <<<blocks, kThreads, 0, stream>>>((T*)out.raw_data(),
                                                  out.stride(0),
                                                  (const T*)x.raw_data(),
                                                  x.stride(0),
                                                  (const T*)weight.raw_data(),
                                                  kHasBias ? (const T*)bias.raw_data() : nullptr,
                                                  dim,
                                                  num,
                                                  eps,
                                                  1.f / dim);
        };

        if (bias) {
            launch(std::true_type{});
        }
        else {
            launch(std::false_type{});
        }
    };

    TM_DISPATCH_DTYPES(x.dtype(), invoke, half_t, bfloat16_t);
}

void invokeResidualBiasLayerNorm(void*        hidden_states,
                                 void*        residual,
                                 const void*  norm_weight,
                                 const void*  norm_bias,
                                 const void*  residual_bias,
                                 DataType     dtype,
                                 int          dims,
                                 int          num,
                                 float        eps,
                                 cudaStream_t stream)
{
    if (num == 0) {
        return;
    }

    TM_CHECK(hidden_states);
    TM_CHECK(residual);
    TM_CHECK(norm_weight);

    auto invoke = [&](auto t) {
        using T = decltype(t);

        constexpr int vec_size = 16 / sizeof(T);
        const int     blocks   = num;

        TM_CHECK(dims % vec_size == 0) << "dims=" << dims << " must be divisible by vec_size=" << vec_size;

        auto launch = [&](auto has_norm_bias_c, auto has_residual_bias_c) {
            // Redeclare these as constexpr inside this nested lambda. MSVC odr-uses
            // (captures) constexpr locals read from an enclosing lambda and then
            // refuses them as non-type template arguments; see rms_norm.cu (RMSNormQK).
            constexpr int  kThreads         = 512;
            constexpr int  kVecSize         = 16 / sizeof(T);
            constexpr bool kHasNormBias     = decltype(has_norm_bias_c)::value;
            constexpr bool kHasResidualBias = decltype(has_residual_bias_c)::value;

            kernel::ResidualBiasLayerNorm<T, float, kThreads, kVecSize, kHasNormBias, kHasResidualBias>
                <<<blocks, kThreads, 0, stream>>>((T*)hidden_states,
                                                  (T*)residual,
                                                  (const T*)norm_weight,
                                                  kHasNormBias ? (const T*)norm_bias : nullptr,
                                                  kHasResidualBias ? (const T*)residual_bias : nullptr,
                                                  dims,
                                                  num,
                                                  eps,
                                                  1.f / dims);
        };

        if (norm_bias && residual_bias) {
            launch(std::true_type{}, std::true_type{});
        }
        else if (norm_bias) {
            launch(std::true_type{}, std::false_type{});
        }
        else if (residual_bias) {
            launch(std::false_type{}, std::true_type{});
        }
        else {
            launch(std::false_type{}, std::false_type{});
        }
    };

    TM_DISPATCH_DTYPES(dtype, invoke, half_t, bfloat16_t);
}

}  // namespace turbomind
