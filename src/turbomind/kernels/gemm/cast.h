// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/data_type.h"
#include <cuda_runtime.h>

namespace turbomind {

void extend_to_u8(uint8_t* dst, const uint4_t* src, size_t n, cudaStream_t st = {});

void extend_to_u16(uint16_t* dst, const uint4_t* src, size_t n, cudaStream_t st = {});

void compact_to_u4(uint4_t* dst, const uint8_t* src, size_t n, cudaStream_t st = {});

void transpose_u4(uint4_t* dst, const uint4_t* src, int s, int c, cudaStream_t st = {});

void fuse_scales_and_zeros(half* fused, const half* scales, half* zeros, size_t n, cudaStream_t st = {});

template<class T>
void interleave_output_dims_impl(T* fused, const T* a, const T* b, int m, int k, cudaStream_t st);

template<class T>
inline void interleave_output_dims(T* fused, const T* a, const T* b, int m, int k, cudaStream_t st)
{
    auto dispatch = [&](auto u) {
        using U = decltype(u);
        return interleave_output_dims_impl((U*)fused, (const U*)a, (const U*)b, m, k, st);
    };
    if constexpr (bitsof<T> == 8) {
        return dispatch(uint8_t{});
    }
    else if constexpr (bitsof<T> == 16) {
        return dispatch(uint16_t{});
    }
    else if constexpr (bitsof<T> == 32) {
        return dispatch(uint32_t{});
    }
}

}  // namespace turbomind
