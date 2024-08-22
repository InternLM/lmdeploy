// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/kernels/gemm/types.h"

#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>

namespace turbomind::gemm {

// quantize using `scale` and `zeros`,
template<class T>
__global__ void find_stats(Array<T, 2>* minmax, const T* src, int N, int K, int G)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y;

    if (n_idx >= N || k_idx * G >= K) {
        return;
    }

    float minval = std::numeric_limits<float>::infinity();
    float maxval = -minval;

    const int L = min(K, G);

    for (int k = 0; k < L; k += 8) {
        Array<T, 8> vec;
        Load(vec, &src[n_idx * K + k_idx * G + k]);
        PRAGMA_UNROLL
        for (int i = 0; i < vec.size(); ++i) {
            minval = __hmin(minval, vec[i]);
            maxval = __hmax(maxval, vec[i]);
        }
    }

    // store in n-major
    Store(minmax[k_idx * N + n_idx].data(), Array<T, 2>{minval, maxval});
}

template<class Q, bool asym, class T>
__global__ void find_params(T* param, const Array<T, 2>* minmax, int count)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= count) {
        return;
    }
    auto        stats     = minmax[global_idx];
    const float inv_q_max = fdividef(1.f, (1 << bitsof<Q>)-1);

    static_assert(asym);

    float scale = (T)(((float)stats[1] - (float)stats[0]) * inv_q_max);

    // force trivial scale / zero for debugging
    if constexpr (0) {
        stats[0] = 0;
        scale    = 1.f;
    }

    Store(param + global_idx * 2, Array<T, 2>{scale, stats[0]});
}

template<class Q, class T>
__global__ void quantize(uint16_t* dst, T* pseudo, const T* src, const T* stats, int N, int K, int G)
{
    static_assert(bitsof<Q> <= 16);
    static_assert(bitsof<T> == 16);  // fp16 & bf16

    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y;

    if (n_idx >= N || k_idx * G >= K) {
        return;
    }

    Array<T, 2> param;
    Load(param, stats + (k_idx * N + n_idx) * 2);

    float inv_scale = fdividef(1.f, param[0]);

    const int L = min(K, G);

    for (int k = 0; k < L; k += 8) {
        Array<T, 8>        vi;
        Array<uint16_t, 8> vo;
        Load(vi, &src[n_idx * K + k_idx * G + k]);

        PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
            float u = (static_cast<float>(vi[i] - param[1])) * inv_scale;
            vo[i]   = quant<uint16_t>(u, bitsof<Q>);
        }
        Store(&dst[n_idx * K + k_idx * G + k], vo);

        if (pseudo) {
            Array<T, 8> vf;
            PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) {
                vf[i] = __hfma(static_cast<T>(vo[i]), param[0], param[1]);
            }
            Store(&pseudo[n_idx * K + k_idx * G + k], vf);
        }
    }
}

template<class T>
__global__ void transpose(const T* src, T* dst, int s, int c)
{
    const int cid = threadIdx.x + blockIdx.x * blockDim.x;
    const int sid = threadIdx.y + blockIdx.y * blockDim.y;
    if (sid < s && cid < c) {
        dst[cid * s + sid] = src[sid * c + cid];
    }
}

template<class T>
void invokeTranspose(const T* src, T* dst, int s, int c, cudaStream_t stream)
{
    const dim3 block{32, 16};
    const dim3 grid(ceil_div<int>(c, block.x), ceil_div<int>(s, block.y));

    transpose<<<grid, block, 0, stream>>>(src, dst, s, c);
}

template<class D, class S>
void Quantize(const thrust::universal_vector<S>&  x,
              int                                 m,
              int                                 k,
              Order                               order,
              int                                 group_size,
              thrust::universal_vector<S>&        x_p,  // pseudo-quantized
              thrust::universal_vector<uint16_t>& x_q,  // quantized ushort
              thrust::universal_vector<S>&        x_u,  // scales & zeros (always m-major)
              cudaStream_t                        stream)

{
    auto policy = thrust::device.on(stream);

    thrust::universal_vector<S>           _x(x.size());
    thrust::universal_vector<S>           _x_p(x.size());
    thrust::universal_vector<uint16_t>    _x_q(x.size());
    thrust::universal_vector<Array<S, 2>> stats(ceil_div(k, group_size) * m);

    x_p.resize(x.size());
    x_q.resize(x.size());
    /// FIXME: correct the size
    x_u.resize(stats.size() * 2);

    if (order == Order::kRowMajor) {
        thrust::copy(policy, x.begin(), x.end(), _x.begin());
    }
    else {
        invokeTranspose(x.data().get(), _x.data().get(), k, m, stream);
    }

    const int  block = std::min(256, m);
    const dim3 grid(ceil_div(m, block), ceil_div(k, group_size));

    find_stats<<<grid, block, 0, stream>>>(stats.data().get(),  //
                                           _x.data().get(),
                                           m,
                                           k,
                                           group_size);

    find_params<D, true><<<ceil_div<int>(stats.size(), 256), 256, 0, stream>>>(  //
        x_u.data().get(),
        stats.data().get(),
        stats.size());

    quantize<D><<<grid, block, 0, stream>>>(_x_q.data().get(),  //
                                            _x_p.data().get(),
                                            _x.data().get(),
                                            x_u.data().get(),
                                            m,
                                            k,
                                            group_size);

    if (order == Order::kRowMajor) {
        thrust::copy(policy, _x_p.begin(), _x_p.end(), x_p.begin());
        thrust::copy(policy, _x_q.begin(), _x_q.end(), x_q.begin());
    }
    else {
        invokeTranspose(_x_p.data().get(), x_p.data().get(), m, k, stream);
        invokeTranspose(_x_q.data().get(), x_q.data().get(), m, k, stream);
    }

    cudaStreamSynchronize(stream);

    // Compare(_x_p.data().get(), _x.data().get(), k, k, m);

    const int kg = ceil_div(k, group_size);
    for (int i = 0; i < m * kg; ++i) {
        // int mi = i % m;
        // int ki = i / m;

        // x_u[i * 2]     = i;
        // x_u[i * 2 + 1] = i;

        // x_u[i * 2]     = i * 2;
        // x_u[i * 2 + 1] = i * 2 + 1;
    }
}

}  // namespace turbomind::gemm
