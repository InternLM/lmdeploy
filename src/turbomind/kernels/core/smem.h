// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include <cassert>

namespace turbomind {

__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void const* const ptr)
{
    return (uint32_t)__cvta_generic_to_shared(ptr);
}

__inline__ __device__ void ldmatrix_m8n8_x4_b16(uint& d0, uint& d1, uint& d2, uint& d3, uint32_t smem_int_ptr)
{
#if TURBOMIND_ARCH_SM75
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(smem_int_ptr));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void ldsm_x4_trans(uint& d0, uint& d1, uint& d2, uint& d3, uint32_t smem_int_ptr)
{
#if TURBOMIND_ARCH_SM75
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(smem_int_ptr));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void ldmatrix_m8n8_x2_b16(uint& d0, uint& d1, uint32_t smem_int_ptr)
{
#if TURBOMIND_ARCH_SM75
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(d0), "=r"(d1) : "r"(smem_int_ptr));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void ldsm_x2_trans(uint& d0, uint& d1, uint32_t smem_int_ptr)
{
#if TURBOMIND_ARCH_SM75
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(smem_int_ptr));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void ldmatrix_m8n8_x1_b16(uint& d0, uint32_t smem_int_ptr)
{
#if TURBOMIND_ARCH_SM75
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 %0, [%1];\n" : "=r"(d0) : "r"(smem_int_ptr));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void ldsm_x1_trans(uint& d0, uint32_t smem_int_ptr)
{
#if TURBOMIND_ARCH_SM75
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 %0, [%1];\n" : "=r"(d0) : "r"(smem_int_ptr));
#else
    assert(TURBOMIND_ARCH_SM75);
#endif
}

__inline__ __device__ void ldsm_x4(Array<uint32_t, 4>& d, uint32_t smem_int_ptr)
{
    ldmatrix_m8n8_x4_b16(d[0], d[1], d[2], d[3], smem_int_ptr);
}

__inline__ __device__ void ldsm_x2(Array<uint32_t, 2>& d, uint32_t smem_int_ptr)
{
    ldmatrix_m8n8_x2_b16(d[0], d[1], smem_int_ptr);
}

__inline__ __device__ void ldsm_x1(Array<uint32_t, 1>& d, uint32_t smem_int_ptr)
{
    ldmatrix_m8n8_x1_b16(d[0], smem_int_ptr);
}

__inline__ __device__ void ldsm_x4_trans(Array<uint32_t, 4>& d, uint32_t smem_int_ptr)
{
    ldsm_x4_trans(d[0], d[1], d[2], d[3], smem_int_ptr);
}

__inline__ __device__ void ldsm_x2_trans(Array<uint32_t, 2>& d, uint32_t smem_int_ptr)
{
    ldsm_x2_trans(d[0], d[1], smem_int_ptr);
}

__inline__ __device__ void ldsm_x1_trans(Array<uint32_t, 1>& d, uint32_t smem_int_ptr)
{
    ldsm_x1_trans(d[0], smem_int_ptr);
}

}  // namespace turbomind
