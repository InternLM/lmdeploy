// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>

template<typename T>
void invokeInsertKeyCache(T* key_cache, const T* src, int L, int H, int Dx, int s, int X, int S, cudaStream_t st);

template<typename T>
void invokeInsertValueCache(T* value_cache, const T* src, int L, int H, int s, int D, int S, cudaStream_t st);