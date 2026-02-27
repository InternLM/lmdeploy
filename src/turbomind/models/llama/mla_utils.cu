// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_bf16.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

template<class T, int vec_size>
__global__ void mla_copy_qkv_kernel(T*       qkv,        // [s, head_num + 2, kv_lora_rank + rope_dim]
                                    const T* q,          // [s, head_num,     kv_lora_rank + rope_dim]
                                    const T* kv_a_k_pe,  // [s, kv_lora_rank + rope_dim]
                                    int      head_num,   // q head num
                                    int      head_dim,   // kv_lora_rank + rope_dim
                                    int      kv_lora_rank,
                                    int      rope_dim)
{
    const int type = blockIdx.y;

    const int64_t ti = blockIdx.x;
    const int     di = threadIdx.x;

    const int offset = di * vec_size < rope_dim ? kv_lora_rank : -rope_dim;

    Array<T, vec_size> data;

    if (type == 0) {  // Q
        for (int hi = threadIdx.y; hi < head_num; hi += blockDim.y) {
            if (di * vec_size < head_dim) {
                Load(data, &q[ti * head_num * head_dim + hi * head_dim + di * vec_size + offset]);
                Store(&qkv[ti * (head_num + 2) * head_dim + hi * head_dim + di * vec_size], data);
            }
        }
    }
    else if (type == 1) {  // K
        if (di * vec_size < head_dim) {
            Ldg(data, &kv_a_k_pe[ti * head_dim + di * vec_size + offset]);
            Store(&qkv[ti * (head_num + 2) * head_dim + (head_num + 0) * head_dim + di * vec_size], data);
        }
    }
    else if (type == 2) {  // V
        if (di * vec_size < head_dim) {
            Ldg(data, &kv_a_k_pe[ti * head_dim + di * vec_size + offset]);
            Store(&qkv[ti * (head_num + 2) * head_dim + (head_num + 1) * head_dim + di * vec_size], data);
        }
    }
}

template<class T>
void invokeMLACopyQKV(T*           qkv,
                      const T*     q,
                      const T*     kv_a_k_pe,
                      int          token_num,
                      int          head_num,
                      int          kv_lora_rank,
                      int          rope_dim,
                      cudaStream_t stream)
{
    constexpr int vec_size = 16 / sizeof(T);

    const int head_dim = kv_lora_rank + rope_dim;  // 512 + 64 = 576

    dim3 block(round_up(head_dim / vec_size, WARP_SIZE), head_num);

    // make sure block size <= 1024
    while (block.x * block.y > 1024) {
        block.y /= 2;
    }

    const dim3 grid(token_num, 3);

    mla_copy_qkv_kernel<T, vec_size>
        <<<grid, block, 0, stream>>>(qkv, q, kv_a_k_pe, head_num, head_dim, kv_lora_rank, rope_dim);
}

void MLACopyQKV(DataType     dtype,
                void*        qkv,
                const void*  q,
                const void*  kv_a_k_pe,
                int          token_num,
                int          head_num,
                int          kv_lora_rank,
                int          rope_dim,
                cudaStream_t stream)
{
    auto invoke = [&](auto t) {
        using T = decltype(t);
        invokeMLACopyQKV(
            (T*)qkv, (const T*)q, (const T*)kv_a_k_pe, token_num, head_num, kv_lora_rank, rope_dim, stream);
    };

    TM_CHECK_EQ(byte_size(dtype, 1), 2) << "unsupported data type: " << dtype;

    return invoke(uint16_t{});
}

}  // namespace turbomind
