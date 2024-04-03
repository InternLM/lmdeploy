// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include <cassert>

namespace turbomind {

__inline__ __device__ void
mma_m16n8k8_row_col(Array<float, 4>& d, const Array<half, 4>& a, const Array<half, 2>& b, Array<float, 4>& c)
{
#if TURBOMIND_ARCH_SM75
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    float const*    C = reinterpret_cast<float const*>(&c);
    float*          D = reinterpret_cast<float*>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void
mma_m16n8k8_row_col(Array<half, 4>& d, const Array<half, 4>& a, const Array<half, 2>& b, Array<half, 4>& c)
{
#if TURBOMIND_ARCH_SM75
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    uint32_t const* C = reinterpret_cast<uint32_t const*>(&c);
    uint32_t*       D = reinterpret_cast<uint32_t*>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16  {%0,%1}, "
                 "{%2,%3}, {%4}, {%5,%6};\n"
                 : "=r"(D[0]), "=r"(D[1])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void mma_m16n8k8_row_col(Array<float, 4>&             d,
                                               const Array<nv_bfloat16, 4>& a,
                                               const Array<nv_bfloat16, 2>& b,
                                               Array<float, 4>&             c)
{
#if TURBOMIND_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    float const*    C = reinterpret_cast<float const*>(&c);
    float*          D = reinterpret_cast<float*>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "
                 "{%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else
    assert(TURBOMIND_ARCH_SM80);
#endif
}

__inline__ __device__ void mma_m16n8k8_row_col(Array<nv_bfloat16, 4>&       d,
                                               const Array<nv_bfloat16, 4>& a,
                                               const Array<nv_bfloat16, 2>& b,
                                               Array<nv_bfloat16, 4>&       c)
{
#if TURBOMIND_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    uint32_t const* C = reinterpret_cast<uint32_t const*>(&c);
    uint32_t*       D = reinterpret_cast<uint32_t*>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.bf16.bf16.bf16.bf16  {%0,%1}, "
                 "{%2,%3}, {%4}, {%5,%6};\n"
                 : "=r"(D[0]), "=r"(D[1])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
#else
    assert(TURBOMIND_ARCH_SM80);
#endif
}

__inline__ __device__ void
mma_m16n8k16_row_col(Array<float, 4>& d, const Array<half, 8>& a, const Array<half, 4>& b, Array<float, 4>& c)
{
#if TURBOMIND_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    float const*    C = reinterpret_cast<float const*>(&c);
    float*          D = reinterpret_cast<float*>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else
    const Array<half, 4>* _a = (const Array<half, 4>*)&a;
    const Array<half, 2>* _b = (const Array<half, 2>*)&b;
    mma_m16n8k8_row_col(d, _a[0], _b[0], c);
    mma_m16n8k8_row_col(d, _a[1], _b[1], d);
#endif
}

__inline__ __device__ void
mma_m16n8k16_row_col(Array<half, 4>& d, const Array<half, 8>& a, const Array<half, 4>& b, Array<half, 4>& c)
{
#if TURBOMIND_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    uint32_t const* C = reinterpret_cast<uint32_t const*>(&c);
    uint32_t*       D = reinterpret_cast<uint32_t*>(&d);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16  {%0,%1}, "
                 "{%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                 : "=r"(D[0]), "=r"(D[1])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
#else
    const Array<half, 4>* _a = (const Array<half, 4>*)&a;
    const Array<half, 2>* _b = (const Array<half, 2>*)&b;
    mma_m16n8k8_row_col(d, _a[0], _b[0], c);
    mma_m16n8k8_row_col(d, _a[1], _b[1], d);
#endif
}

__inline__ __device__ void mma_m16n8k16_row_col(Array<float, 4>&             d,
                                                const Array<nv_bfloat16, 8>& a,
                                                const Array<nv_bfloat16, 4>& b,
                                                Array<float, 4>&             c)
{
#if TURBOMIND_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    float const*    C = reinterpret_cast<float const*>(&c);
    float*          D = reinterpret_cast<float*>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#else
    const Array<nv_bfloat16, 4>* _a = (const Array<nv_bfloat16, 4>*)&a;
    const Array<nv_bfloat16, 2>* _b = (const Array<nv_bfloat16, 2>*)&b;
    mma_m16n8k8_row_col(d, _a[0], _b[0], c);
    mma_m16n8k8_row_col(d, _a[1], _b[1], d);
#endif
}

__inline__ __device__ void mma_m16n8k16_row_col(Array<nv_bfloat16, 4>&       d,
                                                const Array<nv_bfloat16, 8>& a,
                                                const Array<nv_bfloat16, 4>& b,
                                                Array<nv_bfloat16, 4>&       c)
{
#if TURBOMIND_ARCH_SM80
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    uint32_t const* C = reinterpret_cast<uint32_t const*>(&c);
    uint32_t*       D = reinterpret_cast<uint32_t*>(&d);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.bf16.bf16.bf16.bf16  {%0,%1}, "
                 "{%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                 : "=r"(D[0]), "=r"(D[1])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
#else
    const Array<nv_bfloat16, 4>* _a = (const Array<nv_bfloat16, 4>*)&a;
    const Array<nv_bfloat16, 2>* _b = (const Array<nv_bfloat16, 2>*)&b;
    mma_m16n8k8_row_col(d, _a[0], _b[0], c);
    mma_m16n8k8_row_col(d, _a[1], _b[1], d);
#endif
}

}  // namespace turbomind