
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

namespace turbomind {

// =============================================================================
// Shared utility: block-level L2 norm inverse using warp + shared reduction
// Returns rsqrt(sum_sq + eps) visible to all threads in the block.
// smem must be at least ceil(blockDim.x/32) floats.
// SM70 specific optimization: bypass smem entirely if blockDim.x <= 32
// =============================================================================
__device__ __forceinline__ float block_l2_inv_norm(float partial_sq, float* smem, float eps = 1e-6f)
{
    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1)
        partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, mask);

    // Fast path: if the block is a single warp, we don't need shared memory
    if (blockDim.x <= 32) {
        return rsqrtf(partial_sq + eps);
    }

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane_id == 0)
        smem[warp_id] = partial_sq;
    __syncthreads();

    // First warp reduces across warps
    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, mask);
        if (lane_id == 0)
            smem[0] = rsqrtf(val + eps);
    }
    __syncthreads();
    return smem[0];
}

// Helper to accumulate squares of a 16-bit type scalar or vector2 using float32 arithmetic
template<typename T>
__device__ __forceinline__ float sq_acc(T val)
{
    return (float)val * (float)val;
}

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float sq_acc(half2 val)
{
    float2 fval = __half22float2(val);
    return fval.x * fval.x + fval.y * fval.y;
}
__device__ __forceinline__ float sq_acc(nv_bfloat162 val)
{
    float2 fval = __bfloat1622float2(val);
    return fval.x * fval.x + fval.y * fval.y;
}
#endif

// =============================================================================
// Gated Delta Rule — Unified batched kernel (decode + prefill)
//
// Grid = batch_size * num_v_heads blocks, one block per (b, h) pair.
// Reads state via (T*)state_ptrs[b]; token range from q_offsets[b/b+1].
// For decode (seq_len == 1) the time loop runs once; for prefill it runs
// seq_len times — identical body either way.
// =============================================================================
template<typename T>
__global__ void gated_delta_rule_batched_kernel(T*           v_out,
                                                const T*     qkv_in,
                                                const T*     beta_in,
                                                const T*     g_in,
                                                void* const* state_ptrs,
                                                const int*   q_offsets,
                                                int          num_v_heads,
                                                int          num_k_heads,
                                                int          key_head_dim,
                                                int          value_head_dim,
                                                int          k_dim_total,
                                                int          state_layer_offset)
{
    const int bh    = blockIdx.x;
    const int b     = bh / num_v_heads;
    const int h     = bh % num_v_heads;
    const int ratio = num_v_heads / num_k_heads;
    const int kh    = h / ratio;

    const int tok_off    = q_offsets[b];
    const int seq_len    = q_offsets[b + 1] - tok_off;
    const int state_size = key_head_dim * value_head_dim;
    const int conv_dim   = 2 * k_dim_total + num_v_heads * value_head_dim;
    const int v_dim      = num_v_heads * value_head_dim;

    T*          s_ptr = (T*)state_ptrs[b] + state_layer_offset + h * state_size;
    const float scale = rsqrtf((float)key_head_dim);

    __shared__ float smem[32];

    for (int t = 0; t < seq_len; ++t) {
        const int global_t = tok_off + t;

        const T* q_ptr = qkv_in + global_t * conv_dim + kh * key_head_dim;
        const T* k_ptr = qkv_in + global_t * conv_dim + k_dim_total + kh * key_head_dim;
        const T* v_ptr = qkv_in + global_t * conv_dim + 2 * k_dim_total + h * value_head_dim;
        T*       o_ptr = v_out + global_t * v_dim + h * value_head_dim;

        const float beta_val = (float)beta_in[global_t * num_v_heads + h];
        const float decay    = expf((float)g_in[global_t * num_v_heads + h]);

        // --- In-kernel L2-normalize Q (Vectorized) ---
        float q_sq = 0.f;
        if (key_head_dim % 2 == 0) {
            using T2           = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
            const T2* q_ptr_v2 = reinterpret_cast<const T2*>(q_ptr);
            for (int kd = threadIdx.x; kd < key_head_dim / 2; kd += blockDim.x)
                q_sq += sq_acc(q_ptr_v2[kd]);
        }
        else {
            for (int kd = threadIdx.x; kd < key_head_dim; kd += blockDim.x)
                q_sq += sq_acc(q_ptr[kd]);
        }
        const float q_inv_norm = block_l2_inv_norm(q_sq, smem);

        // --- In-kernel L2-normalize K (Vectorized) ---
        float k_sq = 0.f;
        if (key_head_dim % 2 == 0) {
            using T2           = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
            const T2* k_ptr_v2 = reinterpret_cast<const T2*>(k_ptr);
            for (int kd = threadIdx.x; kd < key_head_dim / 2; kd += blockDim.x)
                k_sq += sq_acc(k_ptr_v2[kd]);
        }
        else {
            for (int kd = threadIdx.x; kd < key_head_dim; kd += blockDim.x)
                k_sq += sq_acc(k_ptr[kd]);
        }
        const float k_inv_norm = block_l2_inv_norm(k_sq, smem);

        // --- Step 1: S *= decay ---
        for (int idx = threadIdx.x; idx < state_size; idx += blockDim.x)
            s_ptr[idx] = static_cast<T>((float)s_ptr[idx] * decay);
        __syncthreads();

        // --- Step 2: delta rule update ---
        for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
            float kv_mem = 0.f;
            for (int kd = 0; kd < key_head_dim; ++kd)
                kv_mem += (float)s_ptr[kd * value_head_dim + vd] * ((float)k_ptr[kd] * k_inv_norm);

            const float delta = ((float)v_ptr[vd] - kv_mem) * beta_val;

            for (int kd = 0; kd < key_head_dim; ++kd)
                s_ptr[kd * value_head_dim + vd] =
                    static_cast<T>((float)s_ptr[kd * value_head_dim + vd] + (float)k_ptr[kd] * k_inv_norm * delta);
        }
        __syncthreads();

        // --- Step 3: output = (S^T @ q) * scale ---
        for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
            float o = 0.f;
            for (int kd = 0; kd < key_head_dim; ++kd)
                o += (float)s_ptr[kd * value_head_dim + vd] * ((float)q_ptr[kd] * q_inv_norm);
            o_ptr[vd] = static_cast<T>(o * scale);
        }
        __syncthreads();
    }
}

void invokeGatedDeltaRuleBatched(Ref<Tensor>           v_out_,
                                 const Tensor&         qkv_in,
                                 const Tensor&         beta,
                                 const Tensor&         g,
                                 const Buffer_<void*>& state_ptrs,
                                 const Buffer_<int>&   q_offsets,
                                 int                   batch_size,
                                 int                   num_k_heads,
                                 int                   key_head_dim,
                                 int                   state_layer_offset,
                                 cudaStream_t          stream)
{
    auto& v_out = v_out_.get();

    const int num_v_heads    = beta.shape(1);
    const int v_dim          = v_out.shape(1);
    const int value_head_dim = v_dim / num_v_heads;
    const int k_dim_total    = (qkv_in.shape(1) - v_dim) / 2;

    if (batch_size == 0 || num_v_heads == 0)
        return;

    const int    num_blocks = batch_size * num_v_heads;
    const int    threads    = std::min(256, value_head_dim);
    const size_t smem_sz    = ((threads + 31) / 32) * sizeof(float);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        gated_delta_rule_batched_kernel<<<num_blocks, threads, smem_sz, stream>>>(v_out.data<T>(),
                                                                                  qkv_in.data<T>(),
                                                                                  beta.data<T>(),
                                                                                  g.data<T>(),
                                                                                  state_ptrs.data(),
                                                                                  q_offsets.data(),
                                                                                  num_v_heads,
                                                                                  num_k_heads,
                                                                                  key_head_dim,
                                                                                  value_head_dim,
                                                                                  k_dim_total,
                                                                                  state_layer_offset);
    };
    TM_DISPATCH_PRIMARY_DTYPES(v_out.dtype(), invoke);
}

using namespace gemm;

template<int k_head_dim, int v_head_dim, int block_dim, class T, class S>
__global__ void recurrent_gated_delta_rule_kernel_v2(T*         v_out,
                                                     const T*   qkv_in,
                                                     const T*   beta_in,
                                                     const T*   g_in,
                                                     S* const*  state_ptrs,
                                                     const int* q_offsets,
                                                     int        num_v_heads,
                                                     int        num_k_heads,
                                                     int        k_dim_total,
                                                     int        state_layer_offset)
{
    const int bh    = blockIdx.x;
    const int b     = bh / num_v_heads;
    const int h     = bh % num_v_heads;
    const int ratio = num_v_heads / num_k_heads;
    const int kh    = h / ratio;

    const int tok_off    = q_offsets[b];
    const int seq_len    = q_offsets[b + 1] - tok_off;
    const int state_size = k_head_dim * v_head_dim;
    const int conv_dim   = 2 * k_dim_total + num_v_heads * v_head_dim;
    const int v_dim      = num_v_heads * v_head_dim;

    S* s_ptr = state_ptrs[b] + state_layer_offset + h * state_size;

    const float scale = rsqrtf((float)k_head_dim);

    // DimC = v_head_dim (memory-contiguous), DimS = k_head_dim (strided)
    using Map_S = ThreadMap_V2<v_head_dim, k_head_dim, sizeof(uint4) / sizeof(S), Raked, block_dim / WARP_SIZE>;

    extern __shared__ __align__(16) char smem_buf[];

    // XOR swizzle: bits [10,13] (offset_k) XOR into column access-group index
    constexpr int kBase  = (sizeof(S) == 4) ? 2 : 3;  // log2(kAccessC)
    constexpr int kShift = 10 - kBase;
    using Layout         = SmemLayoutV2<k_head_dim, v_head_dim, -1, -1, Swizzle<4, kBase, kShift>>;
    SmemAccessor<S, Layout> smem_S{(S*)smem_buf};

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    constexpr int tile_k = 16;
    constexpr int tile_v = 4;

    constexpr int k_tiles = k_head_dim / tile_k;  // 8
    constexpr int v_tiles = v_head_dim / tile_v;  // 32

    constexpr int k_threads = k_tiles;
    constexpr int v_threads = block_dim / k_threads;

    constexpr int v_iters = cdiv(v_tiles, v_threads);

    Array<float, tile_v> vec_S[v_iters][tile_k];

    const int offset_k = threadIdx.x % k_tiles;
    const int offset_v = threadIdx.x / k_tiles;

    constexpr int kAccessC = Map_S::kAccessC;

    PRAGMA_UNROLL
    for (int s = 0; s < Map_S::kIterS; ++s) {
        Array<S, kAccessC> vec;
        PRAGMA_UNROLL
        for (int c = 0; c < Map_S::kIterC; ++c) {
            const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
            const int final_vd  = vd + c * Map_S::kDeltaC;
            const int final_kd  = kd + s * Map_S::kDeltaS;
            Load(vec, s_ptr + final_kd * v_head_dim + final_vd);
            Store(&smem_S(final_kd, final_vd), vec);
        }
    }

    __syncthreads();

    PRAGMA_UNROLL
    for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
        PRAGMA_UNROLL
        for (int k = 0; k < tile_k; ++k) {
            constexpr int kTileAccessC = (tile_v >= kAccessC) ? kAccessC : tile_v;
            static_assert(tile_v % kTileAccessC == 0);
            PRAGMA_UNROLL
            for (int c = 0; c < tile_v / kTileAccessC; ++c) {
                Array<S, kTileAccessC> tmp;
                Load(tmp, &smem_S(offset_k * tile_k + k, (offset_v + v_iter * v_threads) * tile_v + c * kTileAccessC));
                (Array<float, kTileAccessC>&)vec_S[v_iter][k][c * kTileAccessC] = cast<float>(tmp);
            }
        }
    }

    for (int t = 0; t < seq_len; ++t) {
        const int global_t = tok_off + t;

        const T* q_ptr = qkv_in + global_t * conv_dim + kh * k_head_dim;
        const T* k_ptr = qkv_in + global_t * conv_dim + k_dim_total + kh * k_head_dim;
        const T* v_ptr = qkv_in + global_t * conv_dim + 2 * k_dim_total + h * v_head_dim;
        T*       o_ptr = v_out + global_t * v_dim + h * v_head_dim;

        const float beta_val = (float)beta_in[global_t * num_v_heads + h];
        const float decay    = expf((float)g_in[global_t * num_v_heads + h]);

        Array<float, tile_k> vec_K;
        Array<float, tile_k> vec_Q;

        // --- In-kernel L2-normalize K/Q (Vectorized) ---
        {
            {
                Array<T, tile_k> tmp_K;
                Array<T, tile_k> tmp_Q;
                Load(tmp_K, &k_ptr[offset_k * tile_k]);
                Load(tmp_Q, &q_ptr[offset_k * tile_k]);
                vec_K = cast<float>(tmp_K);
                vec_Q = cast<float>(tmp_Q);
            }

            float k_sum = 0.f;
            float q_sum = 0.f;

            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                k_sum += vec_K[k] * vec_K[k];
                q_sum += vec_Q[k] * vec_Q[k];
            }

            PRAGMA_UNROLL
            for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                k_sum += __shfl_xor_sync(0xffffffff, k_sum, mask);
                q_sum += __shfl_xor_sync(0xffffffff, q_sum, mask);
            }

            const float k_inv_norm = rsqrtf(k_sum + 1e-6f);
            const float q_inv_norm = rsqrtf(q_sum + 1e-6f);

            PRAGMA_UNROLL
            for (int i = 0; i < tile_k; ++i) {
                vec_K[i] = vec_K[i] * k_inv_norm;
                vec_Q[i] = vec_Q[i] * q_inv_norm;
            }
        }

        // Precompute KQ = dot(K, Q) — invariant across all v elements
        float KQ = 0.f;
        PRAGMA_UNROLL
        for (int k = 0; k < tile_k; ++k)
            KQ += vec_K[k] * vec_Q[k];
        PRAGMA_UNROLL
        for (int mask = k_threads / 2; mask > 0; mask /= 2)
            KQ += __shfl_xor_sync(0xffffffff, KQ, mask);

        Array<T, tile_v> vec_V[v_iters];

        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            Load(vec_V[v_iter], &v_ptr[(offset_v + v_iter * v_threads) * tile_v]);
        }

        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            Array<T, tile_v> vec_O;
            PRAGMA_UNROLL
            for (int v = 0; v < tile_v; ++v) {
                // Fused: decay + dual dot product (kv_mem and SQ simultaneously)
                float kv_mem = 0.f, SQ = 0.f;
                PRAGMA_UNROLL
                for (int k = 0; k < tile_k; ++k) {
                    float s_decayed     = vec_S[v_iter][k][v] * decay;
                    vec_S[v_iter][k][v] = s_decayed;
                    kv_mem += s_decayed * vec_K[k];
                    SQ += s_decayed * vec_Q[k];
                }

                // Single interleaved reduction (2 independent values -> good ILP)
                PRAGMA_UNROLL
                for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                    kv_mem += __shfl_xor_sync(0xffffffff, kv_mem, mask);
                    SQ += __shfl_xor_sync(0xffffffff, SQ, mask);
                }

                const float delta = ((float)vec_V[v_iter][v] - kv_mem) * beta_val;

                // State update
                PRAGMA_UNROLL
                for (int k = 0; k < tile_k; ++k) {
                    vec_S[v_iter][k][v] += vec_K[k] * delta;
                }

                // Output: algebraic computation, NO reduction needed
                vec_O[v] = static_cast<T>((SQ + delta * KQ) * scale);
            }
            if (offset_k == 0)
                Store(&o_ptr[(offset_v + v_iter * v_threads) * tile_v], vec_O);
        }
    }

    __syncthreads();

    PRAGMA_UNROLL
    for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
        PRAGMA_UNROLL
        for (int k = 0; k < tile_k; ++k) {
            constexpr int kTileAccessC = (tile_v >= kAccessC) ? kAccessC : tile_v;
            PRAGMA_UNROLL
            for (int c = 0; c < tile_v / kTileAccessC; ++c) {
                auto tmp = cast<S>((Array<float, kTileAccessC>&)vec_S[v_iter][k][c * kTileAccessC]);
                Store(&smem_S(offset_k * tile_k + k, (offset_v + v_iter * v_threads) * tile_v + c * kTileAccessC), tmp);
            }
        }
    }

    __syncthreads();

    PRAGMA_UNROLL
    for (int s = 0; s < Map_S::kIterS; ++s) {
        Array<S, Map_S::kAccessC> vec;
        PRAGMA_UNROLL
        for (int c = 0; c < Map_S::kIterC; ++c) {
            const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
            const int final_vd  = vd + c * Map_S::kDeltaC;
            const int final_kd  = kd + s * Map_S::kDeltaS;
            Load(vec, &smem_S(final_kd, final_vd));
            Store(s_ptr + final_kd * v_head_dim + final_vd, vec);
        }
    }
}

void invokeGatedDeltaRuleBatched_v2(Ref<Tensor>           v_out_,
                                    const Tensor&         qkv_in,
                                    const Tensor&         beta,
                                    const Tensor&         g,
                                    const Buffer_<void*>& state_ptrs,
                                    const Buffer_<int>&   q_offsets,
                                    int                   batch_size,
                                    int                   num_k_heads,
                                    int                   state_layer_offset,
                                    DataType              state_dtype,
                                    cudaStream_t          stream)
{
    auto& v_out = v_out_.get();

    const int num_v_heads    = beta.shape(1);
    const int v_dim          = v_out.shape(1);
    const int value_head_dim = v_dim / num_v_heads;
    const int k_dim_total    = (qkv_in.shape(1) - v_dim) / 2;

    if (batch_size == 0 || num_v_heads == 0)
        return;

    constexpr int kHeadDim  = 128;
    constexpr int kBlockDim = 256;

    TM_CHECK_EQ(value_head_dim, kHeadDim);
    TM_CHECK_EQ(k_dim_total / num_k_heads, kHeadDim);

    const int num_blocks = batch_size * num_v_heads;

    auto invoke = [&](auto t) {
        using T     = decltype(t);
        auto launch = [&](auto s) {
            using S = decltype(s);

            auto kernel = recurrent_gated_delta_rule_kernel_v2<kHeadDim, kHeadDim, kBlockDim, T, S>;

            constexpr size_t smem_sz = kHeadDim * kHeadDim * sizeof(S);
            if (smem_sz > 48 << 10) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
            }

            kernel<<<num_blocks, kBlockDim, smem_sz, stream>>>(v_out.data<T>(),
                                                               qkv_in.data<T>(),
                                                               beta.data<T>(),
                                                               g.data<T>(),
                                                               (S* const*)state_ptrs.data(),
                                                               q_offsets.data(),
                                                               num_v_heads,
                                                               num_k_heads,
                                                               k_dim_total,
                                                               state_layer_offset);
        };
        if (state_dtype == kFloat32) {
            launch(float{});
        }
        else {
            launch(T{});
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(v_out.dtype(), invoke);
}

// =============================================================================
// Recurrent Gated Delta Rule — Persistent decode kernel (seq_len == 1 only).
//
// Designed for large-batch decode (e.g., bs=1024, 64 heads = 65536 work-items).
// Instead of launching one block per (b, h) pair, we launch only as many blocks
// as can be simultaneously resident (determined via the CUDA occupancy API), and
// each block iterates over multiple (b, h) work-items in a persistent loop.
//
// With 16-bit state (S = T, smem = 32KB per block) on A100:
//   - v2 needs ~152 serial waves for bs=1024, 64 heads
//   - v3 fills the GPU once and loops ~152 times per block → no wave overhead
//
// smem (kHeadDim*kHeadDim*sizeof(S)) is reused across work-items; a
// __syncthreads() at the top of each iteration guards against races on smem
// between consecutive work-items.
// =============================================================================
template<int k_head_dim, int v_head_dim, int block_dim, class T, class S>
__global__ __launch_bounds__(block_dim, 2)
void recurrent_gated_delta_rule_kernel_v3(T*         v_out,
                                           const T*   qkv_in,
                                           const T*   beta_in,
                                           const T*   g_in,
                                           S* const*  state_ptrs,
                                           const int* q_offsets,
                                           int        total_work,
                                           int        num_v_heads,
                                           int        num_k_heads,
                                           int        k_dim_total,
                                           int        state_layer_offset)
{
    constexpr int state_size = k_head_dim * v_head_dim;
    const int     conv_dim   = 2 * k_dim_total + num_v_heads * v_head_dim;
    const int     v_dim      = num_v_heads * v_head_dim;
    const float   scale      = rsqrtf((float)k_head_dim);

    // Compile-time thread partition (identical to v2)
    constexpr int tile_k    = 16;
    constexpr int tile_v    = 4;
    constexpr int k_tiles   = k_head_dim / tile_k;
    constexpr int v_tiles   = v_head_dim / tile_v;
    constexpr int k_threads = k_tiles;
    constexpr int v_threads = block_dim / k_threads;
    constexpr int v_iters   = cdiv(v_tiles, v_threads);

    const int offset_k = threadIdx.x % k_tiles;
    const int offset_v = threadIdx.x / k_tiles;
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;

    // smem layout (swizzled) — same as v2, reused across work-items
    using Map_S          = ThreadMap_V2<v_head_dim, k_head_dim, sizeof(uint4) / sizeof(S), Raked, block_dim / WARP_SIZE>;
    constexpr int kBase  = (sizeof(S) == 4) ? 2 : 3;
    constexpr int kShift = 10 - kBase;
    using Layout         = SmemLayoutV2<k_head_dim, v_head_dim, -1, -1, Swizzle<4, kBase, kShift>>;
    constexpr int kAccessC = Map_S::kAccessC;

    extern __shared__ __align__(16) char smem_buf[];
    SmemAccessor<S, Layout> smem_S{(S*)smem_buf};

    // Persistent loop: one block handles many (b, h) work-items
    for (int work_idx = blockIdx.x; work_idx < total_work; work_idx += gridDim.x) {
        // Guard smem reuse: ensures the previous work-item's smem→global reads
        // are all complete before any thread writes new state into smem.
        __syncthreads();

        const int b     = work_idx / num_v_heads;
        const int h     = work_idx % num_v_heads;
        const int ratio = num_v_heads / num_k_heads;
        const int kh    = h / ratio;

        const int global_t = q_offsets[b];  // seq_len == 1 guaranteed

        S* s_ptr = state_ptrs[b] + state_layer_offset + h * state_size;

        // --- Load state: global → smem ---
        PRAGMA_UNROLL
        for (int s = 0; s < Map_S::kIterS; ++s) {
            Array<S, kAccessC> vec;
            PRAGMA_UNROLL
            for (int c = 0; c < Map_S::kIterC; ++c) {
                const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
                const int final_vd  = vd + c * Map_S::kDeltaC;
                const int final_kd  = kd + s * Map_S::kDeltaS;
                Load(vec, s_ptr + final_kd * v_head_dim + final_vd);
                Store(&smem_S(final_kd, final_vd), vec);
            }
        }

        __syncthreads();

        // --- Load state: smem → registers ---
        Array<float, tile_v> vec_S[v_iters][tile_k];

        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                constexpr int kTileAccessC = (tile_v >= kAccessC) ? kAccessC : tile_v;
                static_assert(tile_v % kTileAccessC == 0);
                PRAGMA_UNROLL
                for (int c = 0; c < tile_v / kTileAccessC; ++c) {
                    Array<S, kTileAccessC> tmp;
                    Load(tmp, &smem_S(offset_k * tile_k + k, (offset_v + v_iter * v_threads) * tile_v + c * kTileAccessC));
                    (Array<float, kTileAccessC>&)vec_S[v_iter][k][c * kTileAccessC] = cast<float>(tmp);
                }
            }
        }

        // --- Process single token (seq_len == 1) ---
        {
            const T* q_ptr = qkv_in + global_t * conv_dim + kh * k_head_dim;
            const T* k_ptr = qkv_in + global_t * conv_dim + k_dim_total + kh * k_head_dim;
            const T* v_ptr = qkv_in + global_t * conv_dim + 2 * k_dim_total + h * v_head_dim;
            T*       o_ptr = v_out + global_t * v_dim + h * v_head_dim;

            const float beta_val = (float)beta_in[global_t * num_v_heads + h];
            const float decay    = expf((float)g_in[global_t * num_v_heads + h]);

            Array<float, tile_k> vec_K;
            Array<float, tile_k> vec_Q;

            // L2-normalize K and Q in registers
            {
                Array<T, tile_k> tmp_K;
                Array<T, tile_k> tmp_Q;
                Load(tmp_K, &k_ptr[offset_k * tile_k]);
                Load(tmp_Q, &q_ptr[offset_k * tile_k]);
                vec_K = cast<float>(tmp_K);
                vec_Q = cast<float>(tmp_Q);
            }

            float k_sum = 0.f, q_sum = 0.f;
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                k_sum += vec_K[k] * vec_K[k];
                q_sum += vec_Q[k] * vec_Q[k];
            }
            PRAGMA_UNROLL
            for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                k_sum += __shfl_xor_sync(0xffffffff, k_sum, mask);
                q_sum += __shfl_xor_sync(0xffffffff, q_sum, mask);
            }
            const float k_inv_norm = rsqrtf(k_sum + 1e-6f);
            const float q_inv_norm = rsqrtf(q_sum + 1e-6f);
            PRAGMA_UNROLL
            for (int i = 0; i < tile_k; ++i) {
                vec_K[i] *= k_inv_norm;
                vec_Q[i] *= q_inv_norm;
            }

            // KQ dot product (invariant across v elements)
            float KQ = 0.f;
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k)
                KQ += vec_K[k] * vec_Q[k];
            PRAGMA_UNROLL
            for (int mask = k_threads / 2; mask > 0; mask /= 2)
                KQ += __shfl_xor_sync(0xffffffff, KQ, mask);

            Array<T, tile_v> vec_V[v_iters];
            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter)
                Load(vec_V[v_iter], &v_ptr[(offset_v + v_iter * v_threads) * tile_v]);

            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                Array<T, tile_v> vec_O;
                PRAGMA_UNROLL
                for (int v = 0; v < tile_v; ++v) {
                    // Fused: decay + dual dot product (kv_mem and SQ simultaneously)
                    float kv_mem = 0.f, SQ = 0.f;
                    PRAGMA_UNROLL
                    for (int k = 0; k < tile_k; ++k) {
                        float s_decayed     = vec_S[v_iter][k][v] * decay;
                        vec_S[v_iter][k][v] = s_decayed;
                        kv_mem += s_decayed * vec_K[k];
                        SQ += s_decayed * vec_Q[k];
                    }
                    PRAGMA_UNROLL
                    for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                        kv_mem += __shfl_xor_sync(0xffffffff, kv_mem, mask);
                        SQ += __shfl_xor_sync(0xffffffff, SQ, mask);
                    }
                    const float delta = ((float)vec_V[v_iter][v] - kv_mem) * beta_val;
                    PRAGMA_UNROLL
                    for (int k = 0; k < tile_k; ++k)
                        vec_S[v_iter][k][v] += vec_K[k] * delta;
                    vec_O[v] = static_cast<T>((SQ + delta * KQ) * scale);
                }
                if (offset_k == 0)
                    Store(&o_ptr[(offset_v + v_iter * v_threads) * tile_v], vec_O);
            }
        }

        // --- Store state: registers → smem ---
        __syncthreads();

        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                constexpr int kTileAccessC = (tile_v >= kAccessC) ? kAccessC : tile_v;
                PRAGMA_UNROLL
                for (int c = 0; c < tile_v / kTileAccessC; ++c) {
                    auto tmp = cast<S>((Array<float, kTileAccessC>&)vec_S[v_iter][k][c * kTileAccessC]);
                    Store(&smem_S(offset_k * tile_k + k, (offset_v + v_iter * v_threads) * tile_v + c * kTileAccessC),
                          tmp);
                }
            }
        }

        __syncthreads();

        // --- Store state: smem → global ---
        PRAGMA_UNROLL
        for (int s = 0; s < Map_S::kIterS; ++s) {
            Array<S, Map_S::kAccessC> vec;
            PRAGMA_UNROLL
            for (int c = 0; c < Map_S::kIterC; ++c) {
                const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
                const int final_vd  = vd + c * Map_S::kDeltaC;
                const int final_kd  = kd + s * Map_S::kDeltaS;
                Load(vec, &smem_S(final_kd, final_vd));
                Store(s_ptr + final_kd * v_head_dim + final_vd, vec);
            }
        }
        // No trailing sync: next iteration's leading __syncthreads() guards reuse.
    }
}

void invokeGatedDeltaRuleBatched_v3(Ref<Tensor>           v_out_,
                                    const Tensor&         qkv_in,
                                    const Tensor&         beta,
                                    const Tensor&         g,
                                    const Buffer_<void*>& state_ptrs,
                                    const Buffer_<int>&   q_offsets,
                                    int                   batch_size,
                                    int                   num_k_heads,
                                    int                   state_layer_offset,
                                    DataType              /*state_dtype*/,  // v3 always uses S = T
                                    cudaStream_t          stream)
{
    auto& v_out = v_out_.get();

    const int num_v_heads = beta.shape(1);
    const int v_dim       = v_out.shape(1);
    const int k_dim_total = (qkv_in.shape(1) - v_dim) / 2;

    if (batch_size == 0 || num_v_heads == 0)
        return;

    constexpr int kHeadDim  = 128;
    constexpr int kBlockDim = 256;

    TM_CHECK_EQ(v_dim / num_v_heads, kHeadDim);
    TM_CHECK_EQ(k_dim_total / num_k_heads, kHeadDim);

    const int total_work = batch_size * num_v_heads;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        using S = T;  // 16-bit state: S == T

        auto             kernel  = recurrent_gated_delta_rule_kernel_v3<kHeadDim, kHeadDim, kBlockDim, T, S>;
        constexpr size_t smem_sz = kHeadDim * kHeadDim * sizeof(S);  // 32 KB for fp16/bf16

        // Determine how many blocks can be simultaneously resident and use that
        // as the persistent grid size, capped at total_work.
        int            blocks_per_sm = 1;
        cudaDeviceProp prop{};
        int            device = 0;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kBlockDim, smem_sz);

        const int num_blocks = min(total_work, blocks_per_sm * prop.multiProcessorCount);

        kernel<<<num_blocks, kBlockDim, smem_sz, stream>>>(v_out.data<T>(),
                                                           qkv_in.data<T>(),
                                                           beta.data<T>(),
                                                           g.data<T>(),
                                                           (S* const*)state_ptrs.data(),
                                                           q_offsets.data(),
                                                           total_work,
                                                           num_v_heads,
                                                           num_k_heads,
                                                           k_dim_total,
                                                           state_layer_offset);
    };
    TM_DISPATCH_PRIMARY_DTYPES(v_out.dtype(), invoke);
}

// =============================================================================
// Chunked Gated Delta Rule kernel — register-centric, small chunk size.
//
// Grid = batch_size * num_v_heads blocks, one block per (b, h) pair.
// Cooperative QKV load to smem per chunk, then sequential per-token
// processing (same recurrence as v2) reading from smem.
// State load/store uses the full swizzled smem buffer (same as v2).
// =============================================================================
template<int kHeadDim, int kChunkSize, int kBlockDim, class T, class S>
__global__ void chunked_gated_delta_rule_kernel(T*         v_out,
                                                const T*   qkv_in,
                                                const T*   beta_in,
                                                const T*   g_in,
                                                S* const*  state_ptrs,
                                                const int* q_offsets,
                                                int        num_v_heads,
                                                int        num_k_heads,
                                                int        k_dim_total,
                                                int        state_layer_offset)
{
    constexpr int C = kChunkSize;
    constexpr int D = kHeadDim;

    const int bh    = blockIdx.x;
    const int b     = bh / num_v_heads;
    const int h     = bh % num_v_heads;
    const int ratio = num_v_heads / num_k_heads;
    const int kh    = h / ratio;

    const int tok_off    = q_offsets[b];
    const int seq_len    = q_offsets[b + 1] - tok_off;
    const int state_size = D * D;
    const int conv_dim   = 2 * k_dim_total + num_v_heads * D;
    const int v_dim      = num_v_heads * D;

    if (seq_len == 0)
        return;

    S*          s_ptr = state_ptrs[b] + state_layer_offset + h * state_size;
    const float scale = rsqrtf((float)D);

    // ── State tiling (same as v2) ──
    constexpr int tile_k    = 8;
    constexpr int tile_v    = 8;
    constexpr int k_tiles   = D / tile_k;                // 16
    constexpr int k_threads = k_tiles;                   // 16
    constexpr int v_threads = kBlockDim / k_threads;     // 16
    constexpr int v_tiles   = D / tile_v;                // 16
    constexpr int v_iters   = cdiv(v_tiles, v_threads);  // 1

    const int offset_k = threadIdx.x % k_threads;
    const int offset_v = threadIdx.x / k_threads;

    Array<float, tile_v> vec_S[v_iters][tile_k];

    extern __shared__ __align__(16) char smem_buf[];

    // ================================================================
    //  LOAD STATE  global → smem (swizzled) → registers   (same as v2)
    // ================================================================
    {
        using Map_S          = ThreadMap_V2<D, D, sizeof(uint4) / sizeof(S), Raked, kBlockDim / WARP_SIZE>;
        constexpr int kBase  = (sizeof(S) == 4) ? 2 : 3;
        constexpr int kShift = 10 - kBase;
        using Layout         = SmemLayoutV2<D, D, -1, -1, Swizzle<4, kBase, kShift>>;
        SmemAccessor<S, Layout> smem_S{(S*)smem_buf};

        const int     warp_id  = threadIdx.x / WARP_SIZE;
        const int     lane_id  = threadIdx.x % WARP_SIZE;
        constexpr int kAccessC = Map_S::kAccessC;

        PRAGMA_UNROLL
        for (int s = 0; s < Map_S::kIterS; ++s) {
            Array<S, kAccessC> vec;
            PRAGMA_UNROLL
            for (int c = 0; c < Map_S::kIterC; ++c) {
                const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
                const int fvd       = vd + c * Map_S::kDeltaC;
                const int fkd       = kd + s * Map_S::kDeltaS;
                Load(vec, s_ptr + fkd * D + fvd);
                Store(&smem_S(fkd, fvd), vec);
            }
        }
        __syncthreads();

        PRAGMA_UNROLL
        for (int vi = 0; vi < v_iters; ++vi) {
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                static_assert(tile_v % Map_S::kAccessC == 0);
                PRAGMA_UNROLL
                for (int c = 0; c < tile_v / Map_S::kAccessC; ++c) {
                    Array<S, Map_S::kAccessC> tmp;
                    Load(tmp,
                         &smem_S(offset_k * tile_k + k, (offset_v + vi * v_threads) * tile_v + c * Map_S::kAccessC));
                    (Array<float, Map_S::kAccessC>&)vec_S[vi][k][c * Map_S::kAccessC] = cast<float>(tmp);
                }
            }
        }
    }
    __syncthreads();

    // ================================================================
    //  CHUNK PROCESSING  — sequential per-token (same as v2) with
    //  smem-cached QKV.  Eliminates resolvent/intra-attention overhead.
    // ================================================================
    // Shared memory layout for chunk processing (overlaps state staging buffer):
    //   k_norm_smem[C][kSmemStride]  — pre-normalized K
    //   q_norm_smem[C][kSmemStride]  — pre-normalized Q
    //   v_smem[C][kSmemStride]       — raw V (as float)
    //   scalars[3*C]                 — beta[C], g[C], scratch[C]
    constexpr int kSmemStride = D + 4;  // pad rows by 4 to avoid 4-way bank conflicts

    float* k_norm_smem = (float*)smem_buf;
    float* q_norm_smem = k_norm_smem + C * kSmemStride;
    float* v_smem      = q_norm_smem + C * kSmemStride;
    float* beta_vals   = v_smem + C * kSmemStride;
    float* g_vals      = beta_vals + C;

    // Thread-to-token mapping for cooperative loads: 1 warp per token
    constexpr int kThreadsPerTok = kBlockDim / C;                 // 256/8 = 32
    constexpr int kElemsPerThr   = D / kThreadsPerTok;            // 128/32 = 4
    const int     load_tok       = threadIdx.x / kThreadsPerTok;  // which token (0..C-1)
    const int     load_lane      = threadIdx.x % kThreadsPerTok;  // lane within token's warp

    const int num_chunks = (seq_len + C - 1) / C;

    for (int ci = 0; ci < num_chunks; ++ci) {
        const int chunk_start = tok_off + ci * C;
        const int valid_len   = min(C, seq_len - ci * C);

        // ────────────────────────────────────────────────────
        //  Phase 0: Cooperative load K, Q, V → smem (pre-normalized)
        //  32 threads (1 warp) per token, 4 elements per thread.
        //  Norms computed via warp shuffle, K/Q normalized in registers
        //  before writing to smem → eliminates one __syncthreads.
        // ────────────────────────────────────────────────────
        {
            float K_reg[kElemsPerThr], Q_reg[kElemsPerThr];
            float k_sq = 0.f, q_sq = 0.f;
            if (load_tok < valid_len) {
                const int gt    = chunk_start + load_tok;
                const T*  k_ptr = qkv_in + gt * conv_dim + k_dim_total + kh * D;
                const T*  q_ptr = qkv_in + gt * conv_dim + kh * D;
                const T*  v_ptr = qkv_in + gt * conv_dim + 2 * k_dim_total + h * D;
                PRAGMA_UNROLL
                for (int e = 0; e < kElemsPerThr; ++e) {
                    const int d = load_lane * kElemsPerThr + e;
                    K_reg[e]    = (float)k_ptr[d];
                    Q_reg[e]    = (float)q_ptr[d];
                    k_sq += K_reg[e] * K_reg[e];
                    q_sq += Q_reg[e] * Q_reg[e];
                    v_smem[load_tok * kSmemStride + d] = (float)v_ptr[d];
                }
                if (load_lane == 0) {
                    beta_vals[load_tok] = (float)beta_in[gt * num_v_heads + h];
                    g_vals[load_tok]    = (float)g_in[gt * num_v_heads + h];
                }
            }
            else {
                PRAGMA_UNROLL
                for (int e = 0; e < kElemsPerThr; ++e) {
                    K_reg[e]                                                      = 0.f;
                    Q_reg[e]                                                      = 0.f;
                    v_smem[load_tok * kSmemStride + load_lane * kElemsPerThr + e] = 0.f;
                }
            }
            // Warp-reduce norms (32-thread warp per token)
            PRAGMA_UNROLL
            for (int mask = kThreadsPerTok / 2; mask > 0; mask >>= 1) {
                k_sq += __shfl_xor_sync(0xffffffff, k_sq, mask);
                q_sq += __shfl_xor_sync(0xffffffff, q_sq, mask);
            }
            const float k_inv = (load_tok < valid_len) ? rsqrtf(k_sq + 1e-6f) : 0.f;
            const float q_inv = (load_tok < valid_len) ? rsqrtf(q_sq + 1e-6f) : 0.f;
            // Write normalized K, Q to smem
            PRAGMA_UNROLL
            for (int e = 0; e < kElemsPerThr; ++e) {
                const int d                             = load_lane * kElemsPerThr + e;
                k_norm_smem[load_tok * kSmemStride + d] = K_reg[e] * k_inv;
                q_norm_smem[load_tok * kSmemStride + d] = Q_reg[e] * q_inv;
            }
        }
        __syncthreads();  // [sync 1] all smem data ready

        // ────────────────────────────────────────────────────
        //  Sequential per-token loop (same computation as v2)
        //  Reads K, Q, V from smem instead of global memory.
        // ────────────────────────────────────────────────────
        PRAGMA_UNROLL
        for (int t = 0; t < C; ++t) {
            if (t >= valid_len)
                break;

            const int   gt       = chunk_start + t;
            const float beta_val = beta_vals[t];
            const float decay    = expf(g_vals[t]);

            float vec_K[tile_k];
            float vec_Q[tile_k];
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                vec_K[k] = k_norm_smem[t * kSmemStride + offset_k * tile_k + k];
                vec_Q[k] = q_norm_smem[t * kSmemStride + offset_k * tile_k + k];
            }

            PRAGMA_UNROLL
            for (int vi = 0; vi < v_iters; ++vi) {
                const int v_base = (offset_v + vi * v_threads) * tile_v;

                float vec_V[tile_v];
                PRAGMA_UNROLL
                for (int v = 0; v < tile_v; ++v)
                    vec_V[v] = v_smem[t * kSmemStride + v_base + v];

                Array<T, tile_v> vec_O;
                PRAGMA_UNROLL
                for (int v = 0; v < tile_v; ++v) {
                    // Step 1: state *= decay
                    PRAGMA_UNROLL
                    for (int k = 0; k < tile_k; ++k)
                        vec_S[vi][k][v] *= decay;

                    // Step 2: delta rule update
                    float kv_mem = 0.f;
                    PRAGMA_UNROLL
                    for (int k = 0; k < tile_k; ++k)
                        kv_mem += vec_S[vi][k][v] * vec_K[k];
                    PRAGMA_UNROLL
                    for (int mask = k_threads / 2; mask > 0; mask /= 2)
                        kv_mem += __shfl_xor_sync(0xffffffff, kv_mem, mask);
                    const float delta = (vec_V[v] - kv_mem) * beta_val;
                    PRAGMA_UNROLL
                    for (int k = 0; k < tile_k; ++k)
                        vec_S[vi][k][v] += vec_K[k] * delta;

                    // Step 3: output = (S^T @ q) * scale
                    float O = 0.f;
                    PRAGMA_UNROLL
                    for (int k = 0; k < tile_k; ++k)
                        O += vec_S[vi][k][v] * vec_Q[k];
                    PRAGMA_UNROLL
                    for (int mask = k_threads / 2; mask > 0; mask /= 2)
                        O += __shfl_xor_sync(0xffffffff, O, mask);
                    vec_O[v] = static_cast<T>(O * scale);
                }
                if (offset_k == 0)
                    Store(&v_out[gt * v_dim + h * D + v_base], vec_O);
            }
        }
        __syncthreads();  // [sync 2] ensure all reads done before next chunk overwrites smem
    }  // chunk loop

    // ================================================================
    //  STORE STATE  registers → smem (swizzled) → global   (same as v2)
    // ================================================================
    {
        using Map_S          = ThreadMap_V2<D, D, sizeof(uint4) / sizeof(S), Raked, kBlockDim / WARP_SIZE>;
        constexpr int kBase  = (sizeof(S) == 4) ? 2 : 3;
        constexpr int kShift = 10 - kBase;
        using Layout         = SmemLayoutV2<D, D, -1, -1, Swizzle<4, kBase, kShift>>;
        SmemAccessor<S, Layout> smem_S{(S*)smem_buf};
        constexpr int           kAccessC = Map_S::kAccessC;

        PRAGMA_UNROLL
        for (int vi = 0; vi < v_iters; ++vi) {
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                PRAGMA_UNROLL
                for (int c = 0; c < tile_v / kAccessC; ++c) {
                    auto tmp = cast<S>((Array<float, kAccessC>&)vec_S[vi][k][c * kAccessC]);
                    Store(&smem_S(offset_k * tile_k + k, (offset_v + vi * v_threads) * tile_v + c * kAccessC), tmp);
                }
            }
        }
        __syncthreads();

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int s = 0; s < Map_S::kIterS; ++s) {
            Array<S, Map_S::kAccessC> vec;
            PRAGMA_UNROLL
            for (int c = 0; c < Map_S::kIterC; ++c) {
                const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
                const int fvd       = vd + c * Map_S::kDeltaC;
                const int fkd       = kd + s * Map_S::kDeltaS;
                Load(vec, &smem_S(fkd, fvd));
                Store(s_ptr + fkd * D + fvd, vec);
            }
        }
    }
}

// Host-side launcher
void invokeChunkedGatedDeltaRuleBatched(Ref<Tensor>           v_out_,
                                        const Tensor&         qkv_in,
                                        const Tensor&         beta,
                                        const Tensor&         g,
                                        const Buffer_<void*>& state_ptrs,
                                        const Buffer_<int>&   q_offsets,
                                        int                   batch_size,
                                        int                   num_k_heads,
                                        int                   state_layer_offset,
                                        DataType              state_dtype,
                                        cudaStream_t          stream)
{
    auto& v_out = v_out_.get();

    const int num_v_heads    = beta.shape(1);
    const int v_dim          = v_out.shape(1);
    const int value_head_dim = v_dim / num_v_heads;
    const int k_dim_total    = (qkv_in.shape(1) - v_dim) / 2;

    if (batch_size == 0 || num_v_heads == 0)
        return;

    constexpr int kHeadDim   = 128;
    constexpr int kChunkSize = 16;
    constexpr int kBlockDim  = 256;

    TM_CHECK_EQ(value_head_dim, kHeadDim);
    TM_CHECK_EQ(k_dim_total / num_k_heads, kHeadDim);

    const int num_blocks = batch_size * num_v_heads;

    auto invoke = [&](auto t) {
        using T     = decltype(t);
        auto launch = [&](auto s) {
            using S = decltype(s);

            auto kernel = chunked_gated_delta_rule_kernel<kHeadDim, kChunkSize, kBlockDim, T, S>;

            // smem = max(state staging, chunk working buffers)
            // State staging: D*D*sizeof(S) (64KB for fp32)
            // Chunk buffers: QKV cache [3*C*(D+4)] + scalars[2*C]
            constexpr size_t state_smem  = kHeadDim * kHeadDim * sizeof(S);
            constexpr int    kSmemStride = kHeadDim + 4;
            constexpr size_t chunk_smem  = 3 * kChunkSize * kSmemStride * sizeof(float)  // k_norm, q_norm, v
                                          + 2 * kChunkSize * sizeof(float);              // beta, g
            constexpr size_t smem_sz = state_smem > chunk_smem ? state_smem : chunk_smem;

            if (smem_sz > 48 << 10) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
            }

            kernel<<<num_blocks, kBlockDim, smem_sz, stream>>>(v_out.data<T>(),
                                                               qkv_in.data<T>(),
                                                               beta.data<T>(),
                                                               g.data<T>(),
                                                               (S* const*)state_ptrs.data(),
                                                               q_offsets.data(),
                                                               num_v_heads,
                                                               num_k_heads,
                                                               k_dim_total,
                                                               state_layer_offset);
        };
        if (state_dtype == kFloat32) {
            launch(float{});
        }
        else {
            launch(T{});
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(v_out.dtype(), invoke);
}

template<class T>
__global__ void compute_beta_g_kernel_v2(T*       beta_out,
                                         T*       g_out,
                                         const T* b_in,
                                         int      b_stride,
                                         const T* a_in,
                                         int      a_stride,
                                         const T* A_log,
                                         const T* dt_bias,
                                         int      total,
                                         int      num_v_heads)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total)
        return;

    const int hi = idx % num_v_heads;
    const int ti = idx / num_v_heads;

    float b_val       = static_cast<float>(b_in[ti * b_stride + hi]);
    float a_val       = static_cast<float>(a_in[ti * a_stride + hi]);
    float A_log_val   = static_cast<float>(A_log[hi]);
    float dt_bias_val = static_cast<float>(dt_bias[hi]);

    float beta  = 1.0f / (1.0f + expf(-b_val));
    float sum   = a_val + dt_bias_val;
    float sp    = sum > 20.0f ? sum : logf(1.0f + expf(sum));
    float g_val = -expf(A_log_val) * sp;

    beta_out[idx] = static_cast<T>(beta);
    g_out[idx]    = static_cast<T>(g_val);
}

void ComputeBetaG_v2(Ref<Tensor>   beta_out_,
                     Ref<Tensor>   g_out_,
                     const Tensor& b_in,
                     const Tensor& a_in,
                     const Tensor& A_log,
                     const Tensor& dt_bias,
                     cudaStream_t  stream)
{

    auto& beta_out = beta_out_.get();
    auto& g_out    = g_out_.get();

    const int threads = 256;
    const int blocks  = cdiv<ssize_t>(beta_out.size(), threads);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        compute_beta_g_kernel_v2<<<blocks, threads, 0, stream>>>(beta_out.data<T>(),
                                                                 g_out.data<T>(),
                                                                 b_in.data<T>(),
                                                                 b_in.stride(0),
                                                                 a_in.data<T>(),
                                                                 a_in.stride(0),
                                                                 A_log.data<T>(),
                                                                 dt_bias.data<T>(),
                                                                 beta_out.size(),
                                                                 A_log.size());
    };

    TM_DISPATCH_PRIMARY_DTYPES(beta_out.dtype(), invoke);
}

// =============================================================================
// RMSNorm * SiLU-Gate (fused output normalization)
// =============================================================================
template<typename T>
__global__ void rms_norm_gated_kernel(
    T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim, int gate_stride, int num_heads)
{
    const int row = blockIdx.x;
    if (row >= N)
        return;

    T*        h         = hidden + row * head_dim;
    const int token_idx = row / num_heads;
    const int head_idx  = row % num_heads;
    const T*  g         = gate + token_idx * gate_stride + head_idx * head_dim;

    __shared__ float smem[32];
    float            sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(h[d]);
        sum_sq += val * val;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = sum_sq;
    __syncthreads();
    if (threadIdx.x >> 5 == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        if (threadIdx.x == 0)
            smem[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = smem[0];

    float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float h_val  = static_cast<float>(h[d]) * inv_rms * static_cast<float>(weight[d]);
        float g_val  = static_cast<float>(g[d]);
        float silu_g = g_val / (1.0f + expf(-g_val));
        h[d]         = static_cast<T>(h_val * silu_g);
    }
}

void invokeRMSNormGated(Ref<Tensor> hidden_, const Tensor& gate, const Tensor& weight, float eps, cudaStream_t stream)
{
    auto& hidden = hidden_.get();

    const int N           = hidden.shape(0);
    const int head_dim    = hidden.shape(1);
    const int token_num   = gate.shape(0);
    const int gate_stride = gate.stride(0);
    const int num_heads   = N / token_num;

    if (N == 0)
        return;

    const int threads = std::min(256, head_dim);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        rms_norm_gated_kernel<<<N, threads, 0, stream>>>(
            hidden.data<T>(), gate.data<T>(), weight.data<T>(), eps, N, head_dim, gate_stride, num_heads);
    };
    TM_DISPATCH_PRIMARY_DTYPES(hidden.dtype(), invoke);
}

// =============================================================================
// Fused Conv1d + SiLU — unified batched kernel (row-major layout)
//
// Handles both decode (seq_len == 1) and prefill (seq_len > 1) per request in
// a single launch. q_offsets[b] / q_offsets[b+1] bound the token range for
// request b. conv_state_ptrs[b] points to state [conv_dim, d_conv] per request.
// =============================================================================
template<typename T>
__global__ void fused_conv1d_batched_kernel(T*           out,
                                            const T*     in,
                                            const T*     weight,
                                            const T*     bias,
                                            void* const* conv_state_ptrs,
                                            const int*   q_offsets,
                                            int          batch_size,
                                            int          conv_dim,
                                            int          d_conv,
                                            int          in_stride,
                                            int          state_layer_offset)
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = q_offsets[batch_size] * conv_dim;
    if (tid >= total)
        return;

    const int global_t = tid / conv_dim;
    const int c        = tid % conv_dim;

    // binary search: find request b such that q_offsets[b] <= global_t < q_offsets[b+1]
    int lo = 0, hi = batch_size - 1;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (q_offsets[m + 1] <= global_t)
            lo = m + 1;
        else
            hi = m;
    }
    const int b       = lo;
    const int t_local = global_t - q_offsets[b];
    const int seq_len = q_offsets[b + 1] - q_offsets[b];

    T*       s   = (T*)conv_state_ptrs[b] + state_layer_offset + c * d_conv;
    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;

    if (seq_len == 1) {
        // decode: shift state in-place, append new input, dot-product from state
#pragma unroll
        for (int d = 0; d < d_conv - 1; ++d)
            s[d] = s[d + 1];
        s[d_conv - 1] = in[global_t * in_stride + c];
#pragma unroll
        for (int d = 0; d < d_conv; ++d)
            acc += static_cast<float>(s[d]) * static_cast<float>(w[d]);
    }
    else {
        // prefill: causal conv with zero-padding before first token of request
#pragma unroll
        for (int d = 0; d < d_conv; ++d) {
            int src = t_local - (d_conv - 1 - d);
            if (src >= 0)
                acc += static_cast<float>(in[(q_offsets[b] + src) * in_stride + c]) * static_cast<float>(w[d]);
        }
        // save trailing inputs into conv state for future decode steps
        if (t_local >= seq_len - d_conv) {
            int state_idx = d_conv - (seq_len - t_local);
            s[state_idx]  = in[global_t * in_stride + c];
        }
    }
    if (bias)
        acc += static_cast<float>(bias[c]);
    out[global_t * conv_dim + c] = static_cast<T>(acc / (1.0f + expf(-acc)));
}

void invokeFusedConv1dSiLU(Ref<Tensor>           out_,
                           const Tensor&         in,
                           const Tensor&         weight,
                           const Tensor&         bias,
                           const Buffer_<void*>& conv_state_ptrs,
                           const Buffer_<int>&   q_offsets,
                           int                   batch_size,
                           int                   state_layer_offset,
                           cudaStream_t          stream)
{
    auto& out = out_.get();

    const int total_tokens = in.shape(0);
    const int conv_dim     = weight.shape(0);
    const int d_conv       = weight.shape(1);
    const int in_stride    = in.stride(0);

    const int threads = 256;
    const int blocks  = (total_tokens * conv_dim + threads - 1) / threads;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        fused_conv1d_batched_kernel<<<blocks, threads, 0, stream>>>(out.data<T>(),
                                                                    in.data<T>(),
                                                                    weight.data<T>(),
                                                                    bias ? bias.data<T>() : (T*)nullptr,
                                                                    conv_state_ptrs.data(),
                                                                    q_offsets.data(),
                                                                    batch_size,
                                                                    conv_dim,
                                                                    d_conv,
                                                                    in_stride,
                                                                    state_layer_offset);
    };
    TM_DISPATCH_PRIMARY_DTYPES(out.dtype(), invoke);
}

}  // namespace turbomind
