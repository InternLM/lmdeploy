

#pragma once

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

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

    float min = std::numeric_limits<float>::infinity();
    float max = -min;

    for (int k = 0; k < G; k += 8) {
        Array<T, 8> vec;
        Load(vec, &src[n_idx * K + k_idx * G + k]);
        PRAGMA_UNROLL
        for (int i = 0; i < vec.size(); ++i) {
            min = __hmin(min, vec[i]);
            max = __hmax(max, vec[i]);
        }
    }

    // store in n-major
    Store(minmax[k_idx * N + n_idx].data(), Array<T, 2>{min, max});
}

template<class Q, bool asym, class T>
__global__ void find_params(T* param, const Array<T, 2>* minmax, int count)
{
    // int global_idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx >= count) {
        return;
    }
    const auto  stats     = minmax[global_idx];
    const float inv_q_max = fdividef(1.f, (1 << bitsof<Q>)-1);

    static_assert(asym);

    float scale = (T)(((float)stats[1] - (float)stats[0]) * inv_q_max);
    Store(param + global_idx * 2, Array<T, 2>{scale, stats[0]});
}

template<class Q, class T>
__global__ void quantize(uint16_t* dst, T* fake, const T* src, const T* stats, int N, int K, int G)
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

    for (int k = 0; k < G; k += 8) {
        Array<T, 8>        vi;
        Array<uint16_t, 8> vo;
        Load(vi, &src[n_idx * K + k_idx * G + k]);

        PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
            float u = (static_cast<float>(vi[i] - param[1])) * inv_scale;
            vo[i]   = quant<uint16_t>(u, bitsof<Q>);
        }
        Store(&dst[n_idx * K + k_idx * G + k], vo);

        if (fake) {
            Array<T, 8> vf;
            PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) {
                vf[i] = __hfma(static_cast<T>(vo[i]), param[0], param[1]);
            }
            Store(&fake[n_idx * K + k_idx * G + k], vf);
        }
    }
}

}  // namespace turbomind::gemm