#pragma once

#include "src/turbomind/kernels/core/array.h"
#include <cuda_bf16.h>

namespace turbomind {

template<class T, int N>
inline __device__ Array<T, N> multimem_ld_reduce_sum(const Array<T, N>* mc_ptr)
{
    return {};
}

inline __device__ Array<half, 8> multimem_ld_reduce_sum(const Array<half, 8>* mc_ptr)
{
    union {
        Array<half, 8>     x;
        Array<uint32_t, 4> u;
    };
    // LDGMC.E.ADD.F16x8.RN.STRONG.SYS
    asm volatile("multimem.ld_reduce.weak.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3])
                 : "l"(mc_ptr)
                 : "memory");
    return x;
}

inline __device__ Array<nv_bfloat16, 8> multimem_ld_reduce_sum(const Array<nv_bfloat16, 8>* mc_ptr)
{
    union {
        Array<nv_bfloat16, 8> x;
        Array<uint32_t, 4>    u;
    };
    asm volatile("multimem.ld_reduce.weak.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3])
                 : "l"(mc_ptr)
                 : "memory");
    return x;
}

template<class T, int N>
inline __device__ void multimem_st(T* mc_ptr, const Array<T, N>& vec)
{
}

inline __device__ void multimem_st(half* mc_ptr, const Array<half, 8>& vec)
{
    union {
        Array<half, 8>     x;
        Array<uint32_t, 4> u;
    };
    x = vec;
    // STG.E.128
    asm volatile("multimem.st.weak.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc_ptr),
                 "r"(u[0]),
                 "r"(u[1]),
                 "r"(u[2]),
                 "r"(u[3]));
}

inline __device__ void multimem_st(nv_bfloat16* mc_ptr, const Array<nv_bfloat16, 8>& vec)
{
    union {
        Array<nv_bfloat16, 8> x;
        Array<uint32_t, 4>    u;
    };
    x = vec;
    asm volatile("multimem.st.weak.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc_ptr),
                 "r"(u[0]),
                 "r"(u[1]),
                 "r"(u[2]),
                 "r"(u[3]));
}

inline __device__ void multimem_st(uint4* mc_ptr, const uint4& u)
{
    asm volatile(
        "multimem.st.weak.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc_ptr), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w));
}

inline __device__ void multimem_st(uint2* mc_ptr, const uint2& u) {}

inline __device__ void multimem_st(uint* mc_ptr, const uint& u) {}

}  // namespace turbomind
