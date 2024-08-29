/*
 * Modified by HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

namespace turbomind {
namespace marlin_qqq {

constexpr int ceildiv(int a, int b)
{
    return (a + b - 1) / b;
}

template<typename T>
inline std::string str(T x)
{
    return std::to_string(x);
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template<typename T, int n>
struct Vec {
    T          elems[n];
    __device__ T& operator[](int i)
    {
        return elems[i];
    }
};

using I4 = Vec<int, 4>;
// Matrix fragments for tensor core instructions; their precise layout is
// documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-integer-type
using FragA         = Vec<uint32_t, 2>;
using FragB         = Vec<uint32_t, 1>;
using FragC         = Vec<int, 4>;
using FragS_GROUP   = Vec<half2, 1>;  // weight per-group quantization scales
using FragS_CHANNEL = Vec<float, 2>;  // weight per-channel quantization scales or activaton
                                      // per-token quantization scales

// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    const int BYTES = 16;
    uint32_t  smem  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 "   .reg .pred p;\n"
                 "   setp.ne.b32 p, %0, 0;\n"
                 "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred),
                 "r"(smem),
                 "l"(glob_ptr),
                 "n"(BYTES));
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr)
{
    const int BYTES = 16;
    uint32_t  smem  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 "   cp.async.cg.shared.global [%0], [%1], %2;\n"
                 "}\n" ::"r"(smem),
                 "l"(glob_ptr),
                 "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template<int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count)
{
    if (threadIdx.x == 0) {
        int state = -1;
        do
            // Guarantee that subsequent writes by this threadblock will be visible
            // globally.
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        while (state != count);
    }
    __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) {
            lock[0] = 0;
            return;
        }
        int val = 1;
        // Make sure that all writes since acquiring this barrier are visible
        // globally, while releasing the barrier.
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}

// NOTE(HandH1998): cp.async.cg only support BYTES = 16, however,
// cp.async.ca can support BYTES = 4, 8, 16;
// as s1's shape is equal to prob_m, we need set s1 to float type,
// and cp_size = 1 float, i.e., 4 BYTES
// Asynchronous global->shared copy for activation quantizaton scales s1
__device__ inline void cp_async1(void* smem_ptr, const void* glob_ptr)
{
    const int BYTES = 4;
    uint32_t  smem  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 "   cp.async.ca.shared.global [%0], [%1], %2;\n"
                 "}\n" ::"r"(smem),
                 "l"(glob_ptr),
                 "n"(BYTES));
}

// m16n8k16 tensor core mma instruction with int8 inputs and int32
// output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    int*            c = reinterpret_cast<int*>(&frag_c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in int8 tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a    = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t  smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(a[0]), "=r"(a[1]) : "r"(smem));
}

inline __device__ half2 float2_to_half2(float2 f)
{
    uint32_t res;
    // NOTE(HandH1998): h0,h1 should be uint16_t, not half
    uint16_t h0, h1;
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h0) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h1) : "f"(f.y));
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(res) : "h"(h0), "h"(h1));
    return reinterpret_cast<half2&>(res);
}

inline __device__ float int32_to_float(int h)
{
    float res;
    asm volatile("cvt.rn.f32.s32 %0, %1;\n" : "=f"(res) : "r"(h));
    return res;
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template<int lut>
__device__ inline int lop3(int a, int b, int c)
{
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 int8 values
// for weight per channel dequant.
__device__ inline FragB dequant_per_channel(int q)
{
    static constexpr int MASK = 0xf0f0f0f0;
    FragB                frag_b;
    frag_b[0] = (q & MASK);
    return frag_b;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 int8 values
// for weight per group dequant.
__device__ inline FragB dequant_per_group(int q, FragS_GROUP& frag_s, int i)
{
    static constexpr uint32_t LO = 0x000f000f;
    static constexpr uint32_t HI = 0x00f000f0;
    static constexpr uint32_t EX = 0x64006400;
    // Guarantee that the `(a & b) | c` operations are LOP3s.
    uint32_t t0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    uint32_t t1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
    // directly into `SUB` and `ADD`.
    static constexpr uint32_t SUB  = 0x64086408;
    static constexpr uint32_t MUL  = 0x2c002c00;
    static constexpr uint32_t ADD  = 0xd480d480;
    *reinterpret_cast<half2*>(&t0) = __hsub2(*reinterpret_cast<half2*>(&t0), *reinterpret_cast<const half2*>(&SUB));
    *reinterpret_cast<half2*>(&t1) = __hfma2(
        *reinterpret_cast<half2*>(&t1), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));

    uint16_t s = reinterpret_cast<uint16_t*>(&frag_s)[i];
    uint32_t double_s;
    // pack 2xfp16 to half2
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(double_s) : "h"(s), "h"(s));
    // dequant and convert 4 half to 4 uint8 (be placed at the low 8 bits of 4
    // half, respectively)
    static constexpr uint32_t MAGIC_NUM = 0x64806480;
    *reinterpret_cast<half2*>(&t0)      = __hfma2(*reinterpret_cast<half2*>(&t0),
                                             *reinterpret_cast<half2*>(&double_s),
                                             *reinterpret_cast<const half2*>(&MAGIC_NUM));
    *reinterpret_cast<half2*>(&t1)      = __hfma2(*reinterpret_cast<half2*>(&t1),
                                             *reinterpret_cast<half2*>(&double_s),
                                             *reinterpret_cast<const half2*>(&MAGIC_NUM));
    // take out the 4 uint8 from 4 half, then convert them to 4 int8 and pack 4
    // int8 into 1 uint32
    FragB                     frag_b;
    uint32_t                  uint8s;
    static constexpr uint32_t MASK_0246            = 0x6420;
    static constexpr uint32_t UINT8s_TO_INT8s_MASK = 0x80808080;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(uint8s) : "r"(t0), "r"(t1), "n"(MASK_0246));
    frag_b[0] = (uint8s ^ UINT8s_TO_INT8s_MASK);
    return frag_b;
}

}  // namespace marlin_qqq
}  // namespace turbomind
