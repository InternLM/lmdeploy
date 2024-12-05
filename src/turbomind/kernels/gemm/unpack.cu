// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include <iostream>

namespace turbomind {

namespace {

__device__ void atomic_assign_u4(uint32_t* address, uint32_t index, uint32_t value)
{
    uint32_t old = *address;
    uint32_t assumed;
    do {
        assumed      = old;
        uint32_t tmp = (assumed & ~(0xfu << (index * 4u))) | (value << (index * 4u));
        old          = atomicCAS(address, assumed, tmp);
    } while (assumed != old);
}

__device__ uint32_t read_u4(const uint32_t* address, uint32_t index)
{
    return (*address >> (index * 4u)) & 0xfu;
}

template<int... Ds>
__global__ void permute_u4(uint* dst, const uint* src, Array<int, sizeof...(Ds)> dims)
{
    constexpr int N = sizeof...(Ds);

    size_t count = 1;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        count *= dims[i];
    }

    constexpr int order[] = {Ds...};

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {

        int indices[N]{};

        PRAGMA_UNROLL
        for (int j = N - 1, ii = i; j >= 0; --j) {
            indices[j] = ii % dims[j];
            ii /= dims[j];
        }

        auto data = read_u4(src + i / 8, i % 8);

        int index = 0;

        PRAGMA_UNROLL
        for (int j = N - 1, stride = 1; j >= 0; --j) {
            index += indices[order[j]] * stride;
            stride *= dims[order[j]];
        }

        atomic_assign_u4(dst + index / 8, index % 8, data);
    }
}

}  // namespace

// col-major interleaved
void unpack_awq_gemm(uint4_t* dst, const uint4_t* src, int rows, int cols, cudaStream_t st)
{
    Array<int, 4> shape{cols, rows / 8, 2, 4};
    permute_u4<0, 1, 3, 2><<<512, 512, 0, st>>>((uint*)dst, (const uint*)src, shape);
}

__global__ void transpose_u4_kernel(uint4_t* dst, const uint4_t* src, int s, int c)
{
    const int idx_c = 8 * (threadIdx.x + blockIdx.x * blockDim.x);
    const int idx_s = 8 * (threadIdx.y + blockIdx.y * blockDim.y);
    if (idx_c >= c || idx_s >= s) {
        return;
    }
    uint32_t ivec[8];
    PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
        ivec[i] = ((const uint32_t*)src)[((idx_s + i) * c + idx_c) / 8];
    }
    uint32_t ovec[8]{};
    PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
        PRAGMA_UNROLL
        for (int j = 0; j < 8; ++j) {
            ovec[i] |= (((ivec[j] >> (i * 4)) & 0xfu) << (j * 4));
        }
    }
    PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
        ((uint32_t*)dst)[((idx_c + i) * s + idx_s) / 8] = ovec[i];
    }
}

void transpose_u4(uint4_t* dst, const uint4_t* src, int s, int c, cudaStream_t st)
{
    if (s % 8 || c % 8) {
        std::cerr << "transpose_u4: invalid shape (" << s << "," << c << "), must be multiple of 8" << std::endl;
        return;
    }
    // Array<int, 2> shape{s, c};
    // permute_u4<1, 0><<<512, 512, 0, st>>>((uint*)dst, (const uint*)src, shape);

    const dim3 block(16, 16);
    const dim3 grid((c + 15) / 16, (s + 15) / 16);
    transpose_u4_kernel<<<grid, block, 0, st>>>(dst, src, s, c);
}

// load -> unpack -> extend_to_u8 -> manipulation -> compat_to_u4 -> store
// load -> extend_to_u16 -> convert -> run

}  // namespace turbomind
