
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

namespace turbomind {

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
                                    int /*sm_count*/,
                                    int* /*work_counter*/,
                                    cudaStream_t stream)
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
// State is loaded/stored directly between global memory and registers (no smem
// staging), eliminating all __syncthreads() from the loop body. Each thread
// owns a [tile_k, tile_v] register tile and issues strided 8-byte tile loads
// directly from global memory. smem_sz = 0 in the host launcher.
// =============================================================================
template<int k_head_dim, int v_head_dim, int block_dim, class T, class S>
__global__ __launch_bounds__(block_dim, 2) void recurrent_gated_delta_rule_kernel_v3(T*         v_out,
                                                                                     const T*   qkv_in,
                                                                                     const T*   beta_in,
                                                                                     const T*   g_in,
                                                                                     S* const*  state_ptrs,
                                                                                     const int* q_offsets,
                                                                                     int*       work_counter,
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

    // Persistent loop: each block atomically claims the next (b, h) work-item.
    // Thread 0 issues the atomic; result is broadcast to all threads via smem.
    __shared__ int s_work_idx;
    while (true) {
        if (threadIdx.x == 0)
            s_work_idx = atomicAdd(work_counter, 1);
        __syncthreads();
        const int work_idx = s_work_idx;
        if (work_idx >= total_work)
            break;
        const int b     = work_idx / num_v_heads;
        const int h     = work_idx % num_v_heads;
        const int ratio = num_v_heads / num_k_heads;
        const int kh    = h / ratio;

        const int global_t = q_offsets[b];  // seq_len == 1 guaranteed

        S* s_ptr = state_ptrs[b] + state_layer_offset + h * state_size;

        // --- Load state: global → registers (direct strided tile loads, tile_v contiguous) ---
        Array<float, tile_v> vec_S[v_iters][tile_k];

        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                Array<S, tile_v> tmp;
                Load(tmp, &s_ptr[(offset_k * tile_k + k) * v_head_dim + (offset_v + v_iter * v_threads) * tile_v]);
                vec_S[v_iter][k] = cast<float>(tmp);
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

        // --- Store state: registers → global (direct strided tile stores, tile_v contiguous) ---
        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            PRAGMA_UNROLL
            for (int k = 0; k < tile_k; ++k) {
                auto tmp = cast<S>(vec_S[v_iter][k]);
                Store(&s_ptr[(offset_k * tile_k + k) * v_head_dim + (offset_v + v_iter * v_threads) * tile_v], tmp);
            }
        }
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
                                    DataType /*state_dtype*/,  // v3 always uses S = T
                                    int          sm_count,
                                    int*         work_counter,
                                    cudaStream_t stream)
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

        auto             kernel        = recurrent_gated_delta_rule_kernel_v3<kHeadDim, kHeadDim, kBlockDim, T, S>;
        constexpr size_t smem_sz       = sizeof(int);  // s_work_idx
        int              blocks_per_sm = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kBlockDim, smem_sz);
        const int grid_blocks = min(total_work, blocks_per_sm * sm_count);

        cudaMemsetAsync(work_counter, 0, sizeof(int), stream);
        kernel<<<grid_blocks, kBlockDim, smem_sz, stream>>>(v_out.data<T>(),
                                                            qkv_in.data<T>(),
                                                            beta.data<T>(),
                                                            g.data<T>(),
                                                            (S* const*)state_ptrs.data(),
                                                            q_offsets.data(),
                                                            work_counter,
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
    }                     // chunk loop

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
                                        int /*sm_count*/,
                                        int* /*work_counter*/,
                                        cudaStream_t stream)
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
// Fused Conv1d + SiLU — persistent batched kernel
//
// Weight layout: [d_conv, conv_dim], State layout: [d_conv, conv_dim] per batch.
//
// Persistent 1D grid. Each block has a fixed channel tile
// (blockIdx.x % num_ch_tiles) and atomically claims single-token work items
// via a global counter. Token-major work ordering with grid size a multiple of
// num_ch_tiles guarantees monotonically increasing tokens and a fixed channel
// tile per block.
// =============================================================================
template<int D_CONV, int CHANNELS_PER_THREAD, int BLOCK_DIM, int NUM_TOKENS, typename T>
__global__ void __launch_bounds__(BLOCK_DIM) fused_conv1d_batched_kernel_v2(T*           out,
                                                                            const T*     in,
                                                                            const T*     weight,
                                                                            const T*     bias,
                                                                            void* const* conv_state_ptrs,
                                                                            const int*   q_offsets,
                                                                            const int*   k_offsets,
                                                                            int*         work_counter,
                                                                            int          batch_size,
                                                                            int          conv_dim,
                                                                            int          in_stride,
                                                                            int          num_token_tiles,
                                                                            int          state_layer_offset,
                                                                            int          total_work,
                                                                            int          num_ch_tiles)
{
    static_assert(BLOCK_DIM * CHANNELS_PER_THREAD > 0);

    int prev_ch_tile = -1;
    int c_base       = 0;

    Array<T, CHANNELS_PER_THREAD> w_tap[D_CONV];
    Array<T, CHANNELS_PER_THREAD> bias_vals;

    __shared__ int  s_work_id;
    __shared__ int4 s_batch_info;
    int             b_start = 0;

    while (true) {
        if (threadIdx.x == 0)
            s_work_id = atomicAdd(work_counter, 1);
        __syncthreads();

        if (s_work_id >= total_work)
            break;

        const int t_tile  = s_work_id % num_token_tiles;
        const int ch_tile = s_work_id / num_token_tiles;

        if (ch_tile != prev_ch_tile) {
            prev_ch_tile = ch_tile;
            b_start      = 0;
        }

        c_base = (ch_tile * BLOCK_DIM + threadIdx.x) * CHANNELS_PER_THREAD;
        PRAGMA_UNROLL
        for (int d = 0; d < D_CONV; ++d) {
            Load(w_tap[d], weight + d * conv_dim + c_base);
        }
        if (bias)
            Load(bias_vals, bias + c_base);

        if constexpr (NUM_TOKENS == 1) {
            for (int b = b_start + threadIdx.x; b < batch_size; b += BLOCK_DIM) {
                int lo = __ldg(&q_offsets[b]);
                if (lo > t_tile)
                    break;
                int hi = __ldg(&q_offsets[b + 1]);
                if (t_tile < hi) {
                    int seq      = hi - lo;
                    int hist     = (__ldg(&k_offsets[b + 1]) - __ldg(&k_offsets[b])) - seq;
                    s_batch_info = make_int4(b, lo, seq, hist);
                }
            }
        }
        else {
            for (int b = b_start + threadIdx.x; b < batch_size; b += BLOCK_DIM) {
                int tile_off = __ldg(&q_offsets[b]) / NUM_TOKENS + b;
                if (tile_off > t_tile)
                    break;
                int tile_off_next = __ldg(&q_offsets[b + 1]) / NUM_TOKENS + b + 1;
                if (t_tile < tile_off_next) {
                    int lo       = __ldg(&q_offsets[b]);
                    int seq      = __ldg(&q_offsets[b + 1]) - lo;
                    int hist     = (__ldg(&k_offsets[b + 1]) - __ldg(&k_offsets[b])) - seq;
                    s_batch_info = make_int4(b, lo, seq, hist);
                }
            }
        }
        __syncthreads();

        b_start = s_batch_info.x;

        const int4 bi          = s_batch_info;
        const int  b           = bi.x;
        const int  seq_off     = bi.y;
        const int  seq_len     = bi.z;
        const int  history_len = bi.w;

        int t_local_start;
        int n_tokens;
        if constexpr (NUM_TOKENS == 1) {
            t_local_start = t_tile - seq_off;
            n_tokens      = 1;
        }
        else {
            const int tile_off_b = seq_off / NUM_TOKENS + b;
            t_local_start        = (t_tile - tile_off_b) * NUM_TOKENS;
            if (t_local_start >= seq_len)
                continue;
            n_tokens = min(NUM_TOKENS, seq_len - t_local_start);
        }

        const int ring_start = (history_len + t_local_start + 1) % D_CONV;
        T*        state_base = (T*)conv_state_ptrs[b] + state_layer_offset;

        constexpr int                 VALS_SIZE = NUM_TOKENS + D_CONV - 1;
        Array<T, CHANNELS_PER_THREAD> vals[VALS_SIZE];
        const int                     n_vals = n_tokens + D_CONV - 1;

        PRAGMA_UNROLL
        for (int i = 0; i < VALS_SIZE; ++i) {
            if (i < n_vals) {
                int pos = t_local_start - (D_CONV - 1) + i;
                if (pos >= 0) {
                    Load(vals[i], in + (seq_off + pos) * in_stride + c_base);
                }
                else {
                    int ring_d = (ring_start + i) % D_CONV;
                    Load(vals[i], state_base + ring_d * conv_dim + c_base);
                }
            }
        }

        PRAGMA_UNROLL
        for (int tok = 0; tok < NUM_TOKENS; ++tok) {
            if (tok < n_tokens) {
                float acc[CHANNELS_PER_THREAD] = {};
                PRAGMA_UNROLL
                for (int d = 0; d < D_CONV; ++d) {
                    PRAGMA_UNROLL
                    for (int ch = 0; ch < CHANNELS_PER_THREAD; ++ch) {
                        acc[ch] += static_cast<float>(vals[tok + d][ch]) * static_cast<float>(w_tap[d][ch]);
                    }
                }

                Array<T, CHANNELS_PER_THREAD> out_vals;
                PRAGMA_UNROLL
                for (int ch = 0; ch < CHANNELS_PER_THREAD; ++ch) {
                    if (bias)
                        acc[ch] += static_cast<float>(bias_vals[ch]);
                    out_vals[ch] = static_cast<T>(acc[ch] / (1.0f + expf(-acc[ch])));
                }

                Store(out + (seq_off + t_local_start + tok) * conv_dim + c_base, out_vals);
            }
        }

        if (t_local_start + n_tokens >= seq_len) {
            PRAGMA_UNROLL
            for (int i = 0; i < VALS_SIZE; ++i) {
                int pos = t_local_start - (D_CONV - 1) + i;
                if (pos >= 0 && pos >= seq_len - D_CONV && pos < seq_len) {
                    int ring_d = (ring_start + i) % D_CONV;
                    Store(state_base + ring_d * conv_dim + c_base, vals[i]);
                }
            }
        }
    }
}

void invokeFusedConv1dSiLU(Ref<Tensor>           out_,
                           const Tensor&         in,
                           const Tensor&         weight,
                           const Tensor&         bias,
                           const Buffer_<void*>& conv_state_ptrs,
                           const Buffer_<int>&   q_offsets,
                           const Buffer_<int>&   k_offsets,
                           int                   batch_size,
                           int                   state_layer_offset,
                           int                   sm_count,
                           int*                  work_counter,
                           cudaStream_t          stream)
{
    auto& out = out_.get();

    const int total_tokens = in.shape(0);
    const int d_conv       = weight.shape(0);
    const int conv_dim     = weight.shape(1);
    const int in_stride    = in.stride(0);

    constexpr int threads = 128;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (d_conv == 4) {
            constexpr int kDConv     = 4;
            constexpr int kChPerT    = 8;
            const int     ch_per_blk = threads * kChPerT;
            TM_CHECK(conv_dim % ch_per_blk == 0);
            const int num_ch_tiles = conv_dim / ch_per_blk;

            auto launch = [&](auto num_tok_tag) {
                constexpr int kNumTok         = decltype(num_tok_tag)::value;
                const int     num_token_tiles = (kNumTok == 1) ? total_tokens : total_tokens / kNumTok + batch_size;
                const int     total_work      = num_token_tiles * num_ch_tiles;

                auto kernel        = fused_conv1d_batched_kernel_v2<kDConv, kChPerT, threads, kNumTok, T>;
                int  blocks_per_sm = 1;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0);
                int grid = min(total_work, blocks_per_sm * sm_count);

                cudaMemsetAsync(work_counter, 0, sizeof(int), stream);
                kernel<<<grid, threads, 0, stream>>>(out.data<T>(),
                                                     in.data<T>(),
                                                     weight.data<T>(),
                                                     bias ? bias.data<T>() : (T*)nullptr,
                                                     conv_state_ptrs.data(),
                                                     q_offsets.data(),
                                                     k_offsets.data(),
                                                     work_counter,
                                                     batch_size,
                                                     conv_dim,
                                                     in_stride,
                                                     num_token_tiles,
                                                     state_layer_offset,
                                                     total_work,
                                                     num_ch_tiles);
            };

            int avg_seq = total_tokens / batch_size;
            if (avg_seq >= 4)
                launch(std::integral_constant<int, 5>{});
            else
                launch(std::integral_constant<int, 1>{});
        }
        else {
            TM_CHECK(0) << "Only d_conv == 4 is supported by fused_conv1d_batched_kernel_v2";
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(out.dtype(), invoke);
}

}  // namespace turbomind
