// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/qkv_preprocess.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

#include <type_traits>

namespace turbomind {

namespace {

constexpr int kWarpsPerBlock = 4;

// Per-head_dim launch traits. Adding a new head_dim is a single specialization here.
template<int HD>
struct HeadConfig;

template<>
struct HeadConfig<64> {
    static constexpr int kVecSize      = 8;
    static constexpr int kHeadsPerWarp = 4;
};

template<>
struct HeadConfig<72> {
    static constexpr int kVecSize      = 8;
    static constexpr int kHeadsPerWarp = 3;
};

template<>
struct HeadConfig<128> {
    static constexpr int kVecSize      = 8;  // 128 / 8 = 16 vec/head
    static constexpr int kHeadsPerWarp = 2;  // 16 * 2 = 32 == WARP_SIZE
};

template<int VecSize, typename T>
__device__ __forceinline__ void add_bias(Array<float, VecSize>& x, const T* bias)
{
    Array<T, VecSize> b;
    Ldg(b, bias);
    using namespace ops;
    x = x + cast<float>(b);
}

// Rotate adjacent (x[2k], x[2k+1]) pairs using cos/sin packed as
// rope[2k]=cos, rope[2k+1]=sin (see fast_rotary_pos_emb.cu:56-58).
template<int VecSize, typename T>
__device__ __forceinline__ void apply_rope_pair(Array<float, VecSize>& x, const Array<T, VecSize>& rope)
{
    auto cs = cast<float>(rope);
    PRAGMA_UNROLL
    for (int i = 0; i < VecSize; i += 2) {
        const float x0 = x[i];
        const float x1 = x[i + 1];
        const float c  = cs[i];
        const float s  = cs[i + 1];
        x[i]           = c * x0 - s * x1;
        x[i + 1]       = c * x1 + s * x0;
    }
}

// Load `src`, add `bias`, apply RoPE, store to `dst` (may equal `src` for in-place).
template<int VecSize, typename T>
__device__ __forceinline__ void fuse_bias_rope_store(T* dst, const T* src, const T* bias, const Array<T, VecSize>& rope)
{
    Array<T, VecSize> x_vec;
    Load(x_vec, src);
    auto x = cast<float>(x_vec);
    add_bias(x, bias);
    apply_rope_pair(x, rope);
    Store(dst, cast<T>(x));
}

// V-path: bias only, no RoPE.
template<int VecSize, typename T>
__device__ __forceinline__ void fuse_bias_store(T* dst, const T* src, const T* bias)
{
    Array<T, VecSize> x_vec;
    Load(x_vec, src);
    auto x = cast<float>(x_vec);
    add_bias(x, bias);
    Store(dst, cast<T>(x));
}

template<typename T, int HD>
__global__ __launch_bounds__(kWarpsPerBlock* WARP_SIZE) void prepareQKVKernel(T* __restrict__ qkv,
                                                                              T* __restrict__ kv,
                                                                              const T* __restrict__ bias,
                                                                              const T* __restrict__ rotary_pos_emb,
                                                                              const int* __restrict__ mapped_idx,
                                                                              int token_num,
                                                                              int local_head_num,
                                                                              int head_group_num,
                                                                              int rope_head_dim)
{
    using Cfg                   = HeadConfig<HD>;
    constexpr int kVecSize      = Cfg::kVecSize;
    constexpr int kHeadsPerWarp = Cfg::kHeadsPerWarp;
    constexpr int kVecPerHead   = HD / kVecSize;
    static_assert(HD % kVecSize == 0);
    static_assert(kVecPerHead * kHeadsPerWarp <= WARP_SIZE);

    const int warp_id   = threadIdx.x / WARP_SIZE;
    const int lane      = threadIdx.x - warp_id * WARP_SIZE;
    const int head_slot = lane / kVecPerHead;
    if (head_slot >= kHeadsPerWarp) {
        return;
    }

    const int global_warp = blockIdx.x * kWarpsPerBlock + warp_id;
    const int total_warps = token_num * head_group_num;
    if (global_warp >= total_warps) {
        return;
    }

    const int token_idx  = global_warp / head_group_num;
    const int head_group = global_warp - token_idx * head_group_num;
    const int head_idx   = head_group * kHeadsPerWarp + head_slot;
    if (head_idx >= local_head_num) {
        return;
    }

    const int vec_idx = lane - head_slot * kVecPerHead;
    const int di      = vec_idx * kVecSize;

    // QKV per-token layout: [Q_heads | K_heads | V_heads], head_num == kv_head_num for ViT.
    const int64_t qkv_stride = (int64_t)local_head_num * 3 * HD;
    T* const      q_ptr      = qkv + (int64_t)token_idx * qkv_stride + head_idx * HD + di;
    const T*      k_ptr      = q_ptr + (int64_t)local_head_num * HD;
    const T*      v_ptr      = k_ptr + (int64_t)local_head_num * HD;

    const T* q_bias = bias + head_idx * HD + di;
    const T* k_bias = q_bias + local_head_num * HD;
    const T* v_bias = k_bias + local_head_num * HD;

    // K/V destination in transposed [kv_head, 2, token, head_dim] layout.
    T* const k_dst = kv + ((int64_t)head_idx * 2 * token_num + token_idx) * HD + di;
    T* const v_dst = k_dst + (int64_t)token_num * HD;

    // rope[token, di] is shared between Q and K — load once, reuse twice.
    // When HD > rope_head_dim, padded di-slices have zero Q/K, so loading a
    // zero rope_vec there is correct (and avoids OOB on the [N, rope_head_dim]
    // buffer). kVecSize is aligned to rope_head_dim so each vec is fully in or
    // fully out of the rope range.
    Array<T, kVecSize> rope_vec{};
    if (di < rope_head_dim) {
        Ldg(rope_vec, rotary_pos_emb + (int64_t)mapped_idx[token_idx] * rope_head_dim + di);
    }

    fuse_bias_rope_store<kVecSize>(q_ptr, q_ptr, q_bias, rope_vec);  // Q: in-place
    fuse_bias_rope_store<kVecSize>(k_dst, k_ptr, k_bias, rope_vec);  // K: transposed
    fuse_bias_store<kVecSize>(v_dst, v_ptr, v_bias);                 // V: transposed, no RoPE
}

template<typename T>
void dispatchPrepareQKV(T*           qkv,
                        T*           kv,
                        const T*     qkv_bias,
                        const T*     rotary_pos_emb,
                        const int*   mapped_idx,
                        int          token_num,
                        int          local_head_num,
                        int          head_dim,
                        int          rope_head_dim,
                        cudaStream_t stream)
{
    auto invoke = [&](auto hd_c) {
        constexpr int HD = decltype(hd_c)::value;
        using Cfg        = HeadConfig<HD>;

        // Each vec_size-wide load must lie entirely in or out of the rope range.
        TM_CHECK(rope_head_dim % Cfg::kVecSize == 0)
            << "rope_head_dim (" << rope_head_dim << ") must be a multiple of kVecSize (" << Cfg::kVecSize << ")";
        TM_CHECK(rope_head_dim <= HD) << "rope_head_dim (" << rope_head_dim << ") cannot exceed head_dim (" << HD
                                      << ")";

        const int head_group_num = (local_head_num + Cfg::kHeadsPerWarp - 1) / Cfg::kHeadsPerWarp;
        const int total_warps    = token_num * head_group_num;
        dim3      grid((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
        prepareQKVKernel<T, HD><<<grid, kWarpsPerBlock * WARP_SIZE, 0, stream>>>(
            qkv, kv, qkv_bias, rotary_pos_emb, mapped_idx, token_num, local_head_num, head_group_num, rope_head_dim);
    };

    switch (head_dim) {
        case 64:
            return invoke(std::integral_constant<int, 64>{});
        case 72:
            return invoke(std::integral_constant<int, 72>{});
        case 128:
            return invoke(std::integral_constant<int, 128>{});
        default:
            TM_LOG_FATAL("unsupported Qwen3.5 ViT head_dim for qkv preprocess: {}", head_dim);
    }
}

}  // namespace

void invokeQwen3_5VitPrepareQKV(void*        qkv,
                                void*        kv,
                                const void*  qkv_bias,
                                const void*  rotary_pos_emb,
                                const int*   mapped_idx,
                                DataType     dtype,
                                int          token_num,
                                int          local_head_num,
                                int          head_dim,
                                int          rope_head_dim,
                                cudaStream_t stream)
{
    if (token_num == 0) {
        return;
    }

    auto invoke = [&](auto t) {
        using T = decltype(t);
        dispatchPrepareQKV((T*)qkv,
                           (T*)kv,
                           (const T*)qkv_bias,
                           (const T*)rotary_pos_emb,
                           mapped_idx,
                           token_num,
                           local_head_num,
                           head_dim,
                           rope_head_dim,
                           stream);
    };

    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
}

}  // namespace turbomind
