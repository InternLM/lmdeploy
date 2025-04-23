// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_bf16.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T, int vec_size>
__global__ void mla_copy_qkv_kernel(T*       qkv,
                                    const T* q,     // [h, head_dim]
                                    const T* kv_a,  // [kv_lora_rank, rope_dim]
                                    const T* kv_b,  // [h, nope_dim + v_head_dim]
                                    int      head_num,
                                    int      head_dim,
                                    int      nope_dim,
                                    int      rope_dim,
                                    int      kv_lora_rank,
                                    int      v_head_dim)
{
    const int type = blockIdx.y;

    const int64_t ti = blockIdx.x;
    const int     di = threadIdx.x;

    const int kv_b_dim = nope_dim + v_head_dim;

    // for (int hi = threadIdx.y; hi < head_num; hi += blockDim.y) {
    const int          hi = threadIdx.y;
    Array<T, vec_size> data{};
    if (type == 0) {  // Q
        if (di * vec_size < rope_dim) {
            Ldg(data, &q[ti * head_num * head_dim + hi * head_dim + nope_dim + di * vec_size]);
        }
        else {
            Ldg(data, &q[ti * head_num * head_dim + hi * head_dim + di * vec_size - rope_dim]);
        }
    }
    else if (type == 1) {  // K
        if (di * vec_size < rope_dim) {
            Ldg(data, &kv_a[ti * (kv_lora_rank + rope_dim) + kv_lora_rank + di * vec_size]);
        }
        else {
            Ldg(data, &kv_b[ti * head_num * kv_b_dim + hi * kv_b_dim + di * vec_size - rope_dim]);
        }
    }
    else {  // V
        if (di * vec_size < v_head_dim) {
            Ldg(data, &kv_b[ti * head_num * kv_b_dim + hi * kv_b_dim + nope_dim + di * vec_size]);
        }
    }
    const int stride = 3 * head_num * head_dim;
    Store(&qkv[ti * stride + type * head_num * head_dim + hi * head_dim + di * vec_size], data);
    // }
}

template<class T>
void invokeMLACopyQKV(T*           qkv,
                      const T*     q,
                      const T*     kv_a,
                      const T*     kv_b,
                      int          token_num,
                      int          head_num,
                      int          nope_dim,
                      int          rope_dim,
                      int          kv_lora_rank,
                      int          v_head_dim,
                      cudaStream_t stream)
{
    constexpr int vec_size = 16 / sizeof(T);
    const int     head_dim = nope_dim + rope_dim;

    dim3 block(head_dim / vec_size, head_num);
    // make sure block size <= 1024
    while (block.x * block.y > 1024) {
        block.y /= 2;
    }
    const dim3 grid(token_num, 3);

    mla_copy_qkv_kernel<T, vec_size><<<grid, block, 0, stream>>>(
        qkv, q, kv_a, kv_b, head_num, head_dim, nope_dim, rope_dim, kv_lora_rank, v_head_dim);
}

void MLACopyQKV(DataType     dtype,
                void*        qkv,
                const void*  q,
                const void*  kv_a,
                const void*  kv_b,
                int          token_num,
                int          head_num,
                int          nope_dim,
                int          rope_dim,
                int          kv_lora_rank,
                int          v_head_dim,
                cudaStream_t stream)
{
    auto invoke = [&](auto t) {
        using T = decltype(t);
        invokeMLACopyQKV((T*)qkv,
                         (const T*)q,
                         (const T*)kv_a,
                         (const T*)kv_b,
                         token_num,
                         head_num,
                         nope_dim,
                         rope_dim,
                         kv_lora_rank,
                         v_head_dim,
                         stream);
    };

    TM_CHECK_EQ(byte_size(dtype, 1), 2) << "unsupported data type: " << dtype;

    return invoke(uint16_t{});
}

}  // namespace turbomind
