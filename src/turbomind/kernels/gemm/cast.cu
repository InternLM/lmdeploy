// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

template<class Ti, class To>
struct Cast {
    template<int N>
    __device__ static Array<To, N> apply(const Array<Ti, N>& vi)
    {
        Array<To, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            vo[i] = static_cast<To>(vi[i]);
        }
        return vo;
    }
};

template<class Ti>
struct Cast<Ti, uint4_t> {
    template<int N>
    __device__ static Array<uint4_t, N> apply(const Array<Ti, N>& vi)
    {
        static_assert(N % 8 == 0);
        Array<uint4_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            uint32_t& v = (uint32_t&)vo[i];
            v           = 0;
            PRAGMA_UNROLL
            for (int j = 7; j >= 0; --j) {
                v = (v << 4) | vi[i + j];
            }
        }
        return vo;
    }
};

template<class To>
struct Cast<uint4_t, To> {
    template<int N>
    __device__ static Array<To, N> apply(const Array<uint4_t, N>& vi)
    {
        static_assert(N % 8 == 0);
        Array<To, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            uint32_t v = (const uint32_t&)vi[i];
            PRAGMA_UNROLL
            for (int j = 0; j < 8; ++j) {
                vo[i + j] = (v & 0xf);
                v >>= 4;
            }
        }
        return vo;
    }
};

template<>
struct Cast<uint4_t, uint4_t> {
    template<int N>
    __device__ static Array<uint4_t, N> apply(const Array<uint4_t, N>& vi)
    {
        return vi;
    }
};

template<int VecSize, class Ti, class To>
__global__ void cast_kernel(To* dst, const Ti* src, size_t n)
{
    n /= VecSize;

    auto p_src = (const Array<Ti, VecSize>*)src;
    auto p_dst = (Array<To, VecSize>*)dst;

    for (size_t p = threadIdx.x + blockDim.x * blockIdx.x; p < n; p += blockDim.x * gridDim.x) {
        Array<Ti, VecSize> vi;
        Ldg(vi, (const Ti*)&p_src[p]);

        Array<To, VecSize> vo = Cast<Ti, To>::apply(vi);

        Store((To*)&p_dst[p], vo);
    }
}

template<int VecSize, class Ti, class To>
void invokeCast(To* dst, const Ti* src, size_t n, cudaStream_t st)
{
    cast_kernel<VecSize><<<256, 256, 0, st>>>(dst, src, n);
}

void extend_to_u8(uint8_t* dst, const uint4_t* src, size_t n, cudaStream_t st)
{
    invokeCast<8>(dst, src, n, st);
}

void compact_to_u4(uint4_t* dst, const uint8_t* src, size_t n, cudaStream_t st)
{
    invokeCast<8>(dst, src, n, st);
}

void extend_to_u16(uint16_t* dst, const uint4_t* src, size_t n, cudaStream_t st)
{
    invokeCast<8>(dst, src, n, st);
}

template<int VecSize, class T>
__global__ void fuse_scales_and_zeros_kernel(T* fused, const T* scales, T* zeros, size_t n)
{
    n /= VecSize;

    auto p_scales = (const Array<T, VecSize>*)scales;
    auto p_zeros  = (const Array<T, VecSize>*)zeros;

    auto p_fused = (Array<T, VecSize * 2>*)fused;

    for (size_t p = threadIdx.x + blockDim.x * blockIdx.x; p < n; p += blockDim.x * gridDim.x) {
        Array<T, VecSize> vs;
        Ldg(vs, (const T*)&p_scales[p]);
        Array<T, VecSize> vz{};
        if (zeros) {
            Ldg(vz, (const T*)&p_zeros[p]);
        }
        Array<T, VecSize * 2> vf;
        PRAGMA_UNROLL
        for (int i = 0; i < VecSize; ++i) {
            vf[i * 2]     = vs[i];
            vf[i * 2 + 1] = -vz[i] * vs[i];
        }
        Store((T*)&p_fused[p], vf);
    }
}

void fuse_scales_and_zeros(half* fused, const half* scales, half* zeros, size_t n, cudaStream_t st)
{
    fuse_scales_and_zeros_kernel<4><<<256, 256, 0, st>>>(fused, scales, zeros, n);
}

template<int VecSize, class T>
__global__ void
interleave_output_dims_kernel(T* __restrict__ fused, const T* __restrict__ a, const T* __restrict__ b, int m, int k)
{
    using Vec1 = Array<T, VecSize>;

    const int ki = blockIdx.y;

    auto p_a = reinterpret_cast<const Vec1*>(a + ki * m);
    auto p_b = reinterpret_cast<const Vec1*>(b + ki * m);

    using Vec2 = Array<T, VecSize * 2>;

    auto p_f = reinterpret_cast<Vec2*>(fused + ki * m * 2);

    m /= VecSize;

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int64_t mi = tidx; mi < m; mi += blockDim.x * gridDim.x) {
        Vec1 va;
        Vec1 vb;
        Ldg(va, (const T*)&p_a[mi]);
        Ldg(vb, (const T*)&p_b[mi]);
        Vec2 vc;
        PRAGMA_UNROLL
        for (int i = 0; i < VecSize; ++i) {
            vc[i * 2]     = va[i];
            vc[i * 2 + 1] = vb[i];
        }
        Store((T*)&p_f[mi], vc);
    }
}

template<class T>
void interleave_output_dims_impl(T* fused, const T* a, const T* b, int m, int k, cudaStream_t st)
{
    constexpr int kVecSize = std::min(8, 128 / (bitsof<T> * 2));

    constexpr int block = 256;
    const dim3    grid(1, k);  // x is a grid stride loop

    interleave_output_dims_kernel<kVecSize><<<grid, block, 0, st>>>(fused, a, b, m, k);
}

template void
interleave_output_dims_impl(uint8_t* fused, const uint8_t* a, const uint8_t* b, int m, int k, cudaStream_t st);
template void
interleave_output_dims_impl(uint16_t* fused, const uint16_t* a, const uint16_t* b, int m, int k, cudaStream_t st);
template void
interleave_output_dims_impl(uint32_t* fused, const uint32_t* a, const uint32_t* b, int m, int k, cudaStream_t st);

}  // namespace turbomind
