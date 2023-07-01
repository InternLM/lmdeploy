// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/prefix_cache.h"

// <L,H,D/X,s,X> -> <L,H,D/X,S[:s],X>
template<typename T>
__global__ void insertKeyCache(T* key_cache, const T* src, int L, int H, int Dx, int s, int X, size_t S)
{
    for (int i = threadIdx.x; i < L * H * Dx * s * X; i += blockDim.x) {
        int i0 = i / X;
        int x  = i % X;

        int i1 = i0 / s;
        int t  = i0 % s;

        size_t j     = (i1 * S + t) * X + x;
        key_cache[j] = src[i];
    }
}

template<typename T>
void invokeInsertKeyCache(T* key_cache, const T* src, int L, int H, int Dx, int s, int X, int S, cudaStream_t st)
{
    insertKeyCache<<<1, 512, 0, st>>>(key_cache, src, L, H, Dx, s, X, S);
}
template void
invokeInsertKeyCache(float* key_cache, const float* src, int L, int H, int Dx, int s, int X, int S, cudaStream_t st);
template void
invokeInsertKeyCache(half* key_cache, const half* src, int L, int H, int Dx, int s, int X, int S, cudaStream_t st);

// <L,H,s,D> -> <L,H,S[:s],D>
template<typename T>
__global__ void insertValueCache(T* value_cache, const T* src, int L, int H, int s, int D, size_t S)
{
    for (int i = threadIdx.x; i < L * H * s * D; i += blockDim.x) {
        int i0 = i / D;
        int d  = i % D;

        int i1 = i0 / s;
        int t  = i0 % s;

        size_t j       = (i1 * S + t) * D + d;
        value_cache[j] = src[i];
    }
}

template<typename T>
void invokeInsertValueCache(T* value_cache, const T* src, int L, int H, int s, int D, int S, cudaStream_t st)
{
    insertValueCache<<<1, 512, 0, st>>>(value_cache, src, L, H, s, D, S);
}
template void
invokeInsertValueCache(float* value_cache, const float* src, int L, int H, int s, int D, int S, cudaStream_t st);
template void
invokeInsertValueCache(half* value_cache, const half* src, int L, int H, int s, int D, int S, cudaStream_t st);
