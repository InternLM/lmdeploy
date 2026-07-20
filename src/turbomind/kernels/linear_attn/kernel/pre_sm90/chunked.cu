#include "src/turbomind/kernels/linear_attn/kernel/pre_sm90/internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::linear_attn::delta_rule::detail {

using namespace gemm;

template<int HeadDim, int ChunkSize, int BlockDim, class T, class StateT>
__global__ void ChunkedGdrKernel(T*             out,
                                 int64_t        out_batch_stride,
                                 int64_t        out_token_stride,
                                 const T*       q,
                                 int64_t        q_batch_stride,
                                 int64_t        q_token_stride,
                                 const T*       k,
                                 int64_t        k_batch_stride,
                                 int64_t        k_token_stride,
                                 const T*       v,
                                 int64_t        v_batch_stride,
                                 int64_t        v_token_stride,
                                 const float*   beta,
                                 const float*   g,
                                 int64_t        gate_batch_stride,
                                 int64_t        gate_token_stride,
                                 const int64_t* state_ptrs,
                                 const int32_t* q_offsets,
                                 const bool*    finished,
                                 int            physical_token_slots,
                                 int            sequence_count,
                                 int            hq,
                                 int            hv,
                                 int64_t        state_layer_offset,
                                 int            num_head_groups,
                                 int            heads_per_block)
{
#if __CUDA_ARCH__
    if constexpr (__CUDA_ARCH__ < 900) {
        constexpr int C = ChunkSize;
        constexpr int D = HeadDim;

        const int sequence        = int(blockIdx.x) / hv;
        const int value_head      = int(blockIdx.x) % hv;
        const int key_head        = value_head / (hv / hq);
        const int token_begin     = q_offsets[sequence];
        const int sequence_length = q_offsets[sequence + 1] - token_begin;

        const int head_group = value_head / heads_per_block;
        const int local_head = value_head % heads_per_block;
        auto*     state =
            reinterpret_cast<StateT*>(static_cast<uintptr_t>(state_ptrs[sequence * num_head_groups + head_group]))
            + state_layer_offset + int64_t(local_head) * D * D;

        constexpr int tile_k    = 8;
        constexpr int tile_v    = 8;
        constexpr int k_threads = D / tile_k;
        constexpr int v_threads = BlockDim / k_threads;
        constexpr int v_iters   = (D / tile_v + v_threads - 1) / v_threads;

        const int offset_k = threadIdx.x % k_threads;
        const int offset_v = threadIdx.x / k_threads;

        Array<float, tile_v> vec_S[v_iters][tile_k];

        extern __shared__ __align__(16) char smem_buf[];

        {
            using Map_S          = ThreadMap_V2<D, D, sizeof(uint4) / sizeof(StateT), Raked, BlockDim / WARP_SIZE>;
            constexpr int kBase  = sizeof(StateT) == 4 ? 2 : 3;
            constexpr int kShift = 10 - kBase;
            using Layout         = SmemLayoutV2<D, D, -1, -1, Swizzle<4, kBase, kShift>>;
            SmemAccessor<StateT, Layout> smem_S{reinterpret_cast<StateT*>(smem_buf)};

            const int     warp_id  = threadIdx.x / WARP_SIZE;
            const int     lane_id  = threadIdx.x % WARP_SIZE;
            constexpr int kAccessC = Map_S::kAccessC;

            PRAGMA_UNROLL
            for (int s = 0; s < Map_S::kIterS; ++s) {
                Array<StateT, kAccessC> vec;
                PRAGMA_UNROLL
                for (int c = 0; c < Map_S::kIterC; ++c) {
                    const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
                    const int final_vd  = vd + c * Map_S::kDeltaC;
                    const int final_kd  = kd + s * Map_S::kDeltaS;
                    Load(vec, state + final_kd * D + final_vd);
                    Store(&smem_S(final_kd, final_vd), vec);
                }
            }
            __syncthreads();

            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                PRAGMA_UNROLL
                for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                    static_assert(tile_v % Map_S::kAccessC == 0);
                    PRAGMA_UNROLL
                    for (int c = 0; c < tile_v / Map_S::kAccessC; ++c) {
                        Array<StateT, Map_S::kAccessC> tmp;
                        Load(tmp,
                             &smem_S(offset_k * tile_k + k_idx,
                                     (offset_v + v_iter * v_threads) * tile_v + c * Map_S::kAccessC));
                        reinterpret_cast<Array<float, Map_S::kAccessC>&>(vec_S[v_iter][k_idx][c * Map_S::kAccessC]) =
                            cast<float>(tmp);
                    }
                }
            }
        }
        __syncthreads();

        constexpr int kSmemStride = D + 4;
        float*        k_smem      = reinterpret_cast<float*>(smem_buf);
        float*        q_smem      = k_smem + C * kSmemStride;
        float*        v_smem      = q_smem + C * kSmemStride;
        float*        beta_vals   = v_smem + C * kSmemStride;
        float*        g_vals      = beta_vals + C;

        constexpr int kThreadsPerToken   = BlockDim / C;
        constexpr int kElementsPerThread = D / kThreadsPerToken;
        const int     load_token         = threadIdx.x / kThreadsPerToken;
        const int     load_lane          = threadIdx.x % kThreadsPerToken;

        const int chunk_count = (sequence_length + C - 1) / C;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const int chunk_start  = token_begin + chunk * C;
            const int valid_tokens = min(C, sequence_length - chunk * C);

            if (load_token < valid_tokens) {
                const int global_token   = chunk_start + load_token;
                const int physical_batch = global_token / physical_token_slots;
                const int token          = global_token % physical_token_slots;
                const T*  q_ptr = q + int64_t(physical_batch) * q_batch_stride + int64_t(token) * q_token_stride
                                 + int64_t(key_head) * D;
                const T* k_ptr = k + int64_t(physical_batch) * k_batch_stride + int64_t(token) * k_token_stride
                                 + int64_t(key_head) * D;
                const T* v_ptr = v + int64_t(physical_batch) * v_batch_stride + int64_t(token) * v_token_stride
                                 + int64_t(value_head) * D;
                PRAGMA_UNROLL
                for (int element = 0; element < kElementsPerThread; ++element) {
                    const int d                          = load_lane * kElementsPerThread + element;
                    k_smem[load_token * kSmemStride + d] = float(k_ptr[d]);
                    q_smem[load_token * kSmemStride + d] = float(q_ptr[d]);
                    v_smem[load_token * kSmemStride + d] = float(v_ptr[d]);
                }
                if (load_lane == 0) {
                    beta_vals[load_token] = beta[int64_t(physical_batch) * gate_batch_stride
                                                 + int64_t(token) * gate_token_stride + value_head];
                    g_vals[load_token]    = g[int64_t(physical_batch) * gate_batch_stride
                                           + int64_t(token) * gate_token_stride + value_head];
                }
            }
            else {
                PRAGMA_UNROLL
                for (int element = 0; element < kElementsPerThread; ++element) {
                    const int d                          = load_lane * kElementsPerThread + element;
                    k_smem[load_token * kSmemStride + d] = 0.f;
                    q_smem[load_token * kSmemStride + d] = 0.f;
                    v_smem[load_token * kSmemStride + d] = 0.f;
                }
            }
            __syncthreads();

            PRAGMA_UNROLL
            for (int token_in_chunk = 0; token_in_chunk < C; ++token_in_chunk) {
                if (token_in_chunk >= valid_tokens) {
                    break;
                }

                const float beta_value = beta_vals[token_in_chunk];
                const float decay      = expf(g_vals[token_in_chunk]);
                float       vec_k[tile_k];
                float       vec_q[tile_k];
                PRAGMA_UNROLL
                for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                    vec_k[k_idx] = k_smem[token_in_chunk * kSmemStride + offset_k * tile_k + k_idx];
                    vec_q[k_idx] = q_smem[token_in_chunk * kSmemStride + offset_k * tile_k + k_idx];
                }

                const int global_token   = chunk_start + token_in_chunk;
                const int physical_batch = global_token / physical_token_slots;
                const int token          = global_token % physical_token_slots;
                T*        out_ptr = out + int64_t(physical_batch) * out_batch_stride + int64_t(token) * out_token_stride
                             + int64_t(value_head) * D;

                PRAGMA_UNROLL
                for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                    const int value_base = (offset_v + v_iter * v_threads) * tile_v;
                    float     vec_v[tile_v];
                    PRAGMA_UNROLL
                    for (int v_idx = 0; v_idx < tile_v; ++v_idx) {
                        vec_v[v_idx] = v_smem[token_in_chunk * kSmemStride + value_base + v_idx];
                    }

                    Array<T, tile_v> vec_out;
                    PRAGMA_UNROLL
                    for (int v_idx = 0; v_idx < tile_v; ++v_idx) {
                        PRAGMA_UNROLL
                        for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                            vec_S[v_iter][k_idx][v_idx] *= decay;
                        }

                        float kv_memory = 0.f;
                        PRAGMA_UNROLL
                        for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                            kv_memory += vec_S[v_iter][k_idx][v_idx] * vec_k[k_idx];
                        }
                        PRAGMA_UNROLL
                        for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                            kv_memory += __shfl_xor_sync(0xffffffff, kv_memory, mask);
                        }
                        const float delta = (vec_v[v_idx] - kv_memory) * beta_value;
                        PRAGMA_UNROLL
                        for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                            vec_S[v_iter][k_idx][v_idx] += vec_k[k_idx] * delta;
                        }

                        float value = 0.f;
                        PRAGMA_UNROLL
                        for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                            value += vec_S[v_iter][k_idx][v_idx] * vec_q[k_idx];
                        }
                        PRAGMA_UNROLL
                        for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                            value += __shfl_xor_sync(0xffffffff, value, mask);
                        }
                        vec_out[v_idx] = T(value * rsqrtf(float(D)));
                    }
                    if (offset_k == 0) {
                        Store(&out_ptr[value_base], vec_out);
                    }
                }
            }
            __syncthreads();
        }

        if (!finished[sequence]) {
            using Map_S          = ThreadMap_V2<D, D, sizeof(uint4) / sizeof(StateT), Raked, BlockDim / WARP_SIZE>;
            constexpr int kBase  = sizeof(StateT) == 4 ? 2 : 3;
            constexpr int kShift = 10 - kBase;
            using Layout         = SmemLayoutV2<D, D, -1, -1, Swizzle<4, kBase, kShift>>;
            SmemAccessor<StateT, Layout> smem_S{reinterpret_cast<StateT*>(smem_buf)};
            constexpr int                kAccessC = Map_S::kAccessC;

            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                PRAGMA_UNROLL
                for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < tile_v / kAccessC; ++c) {
                        auto tmp =
                            cast<StateT>(reinterpret_cast<Array<float, kAccessC>&>(vec_S[v_iter][k_idx][c * kAccessC]));
                        Store(
                            &smem_S(offset_k * tile_k + k_idx, (offset_v + v_iter * v_threads) * tile_v + c * kAccessC),
                            tmp);
                    }
                }
            }
            __syncthreads();

            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int s = 0; s < Map_S::kIterS; ++s) {
                Array<StateT, Map_S::kAccessC> vec;
                PRAGMA_UNROLL
                for (int c = 0; c < Map_S::kIterC; ++c) {
                    const auto [vd, kd] = Map_S::get_offset(warp_id, lane_id);
                    const int final_vd  = vd + c * Map_S::kDeltaC;
                    const int final_kd  = kd + s * Map_S::kDeltaS;
                    Load(vec, &smem_S(final_kd, final_vd));
                    Store(state + final_kd * D + final_vd, vec);
                }
            }
        }
    }
#endif
}

template<class T, class StateT>
void RunPreSm90Chunk16(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    constexpr int kBlockDim  = 256;
    constexpr int kChunkSize = 16;
    const int     grid       = plan.problem.sequence_num * plan.problem.hv;
    const size_t  state_smem = 128 * 128 * sizeof(StateT);
    const size_t  chunk_smem = 3 * kChunkSize * (128 + 4) * sizeof(float) + 2 * kChunkSize * sizeof(float);
    const size_t  smem_bytes = std::max(state_smem, chunk_smem);
    auto          kernel     = ChunkedGdrKernel<128, kChunkSize, kBlockDim, T, StateT>;
    if (smem_bytes > (48u << 10)) {
        TM_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    kernel<<<grid, kBlockDim, smem_bytes, stream>>>(args.out->data<T>(),
                                                    args.out->stride(0),
                                                    args.out->stride(1),
                                                    args.q.data<T>(),
                                                    args.q.stride(0),
                                                    args.q.stride(1),
                                                    args.k.data<T>(),
                                                    args.k.stride(0),
                                                    args.k.stride(1),
                                                    args.v.data<T>(),
                                                    args.v.stride(0),
                                                    args.v.stride(1),
                                                    args.beta.data<float>(),
                                                    args.g.data<float>(),
                                                    args.g.stride(0),
                                                    args.g.stride(1),
                                                    reinterpret_cast<const int64_t*>(args.state_ptrs.raw_data()),
                                                    args.q_offsets.data<int32_t>(),
                                                    args.finished.data<bool>(),
                                                    plan.problem.token_num,
                                                    plan.problem.sequence_num,
                                                    plan.problem.hq,
                                                    plan.problem.hv,
                                                    args.state_layer_offset,
                                                    plan.problem.num_head_groups,
                                                    plan.problem.heads_per_block);
    TM_CUDA_CHECK(cudaGetLastError());
}

#define DEFINE_PRE_SM90_CALLBACK(name, implementation, InputT, StateT)                                                 \
    void name(const Arguments& args, const Plan& plan, cudaStream_t stream)                                            \
    {                                                                                                                  \
        implementation<InputT, StateT>(args, plan, stream);                                                            \
    }

DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16F16F16, RunPreSm90Chunk16, half, half)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16F16F32, RunPreSm90Chunk16, half, float)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16Bf16Bf16, RunPreSm90Chunk16, __nv_bfloat16, __nv_bfloat16)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16Bf16F32, RunPreSm90Chunk16, __nv_bfloat16, float)

#undef DEFINE_PRE_SM90_CALLBACK

}  // namespace turbomind::linear_attn::delta_rule::detail
