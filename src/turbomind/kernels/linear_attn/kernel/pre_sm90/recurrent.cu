#include "src/turbomind/kernels/linear_attn/kernel/pre_sm90/internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::linear_attn::delta_rule::detail {

template<int HeadDim, int BlockDim, class T, class StateT>
__global__ __launch_bounds__(BlockDim, 2) void RecurrentGdrKernel(T*             out,
                                                                  int64_t        out_batch_stride,
                                                                  const T*       q,
                                                                  int64_t        q_batch_stride,
                                                                  const T*       k,
                                                                  int64_t        k_batch_stride,
                                                                  const T*       v,
                                                                  int64_t        v_batch_stride,
                                                                  const float*   beta,
                                                                  int64_t        gate_batch_stride,
                                                                  const float*   g,
                                                                  const int64_t* state_ptrs,
                                                                  const bool*    finished,
                                                                  int            total_work,
                                                                  int            hq,
                                                                  int            hv,
                                                                  int64_t        state_layer_offset,
                                                                  int            num_head_groups,
                                                                  int            heads_per_block)
{
#if __CUDA_ARCH__
    if constexpr (__CUDA_ARCH__ < 900) {
        constexpr int tile_k    = 16;
        constexpr int tile_v    = 4;
        constexpr int k_threads = HeadDim / tile_k;
        constexpr int v_threads = BlockDim / k_threads;
        constexpr int v_iters   = (HeadDim / tile_v + v_threads - 1) / v_threads;
        const int     offset_k  = threadIdx.x % k_threads;
        const int     offset_v  = threadIdx.x / k_threads;

        for (int work_idx = int(blockIdx.x); work_idx < total_work; work_idx += int(gridDim.x)) {
            const int sequence   = work_idx / hv;
            const int value_head = work_idx % hv;
            const int key_head   = value_head / (hv / hq);
            const T*  q_ptr      = q + int64_t(sequence) * q_batch_stride + int64_t(key_head) * HeadDim;
            const T*  k_ptr      = k + int64_t(sequence) * k_batch_stride + int64_t(key_head) * HeadDim;
            const T*  v_ptr      = v + int64_t(sequence) * v_batch_stride + int64_t(value_head) * HeadDim;
            T*        out_ptr    = out + int64_t(sequence) * out_batch_stride + int64_t(value_head) * HeadDim;
            const int head_group = value_head / heads_per_block;
            const int local_head = value_head % heads_per_block;
            auto*     state =
                reinterpret_cast<StateT*>(static_cast<uintptr_t>(state_ptrs[sequence * num_head_groups + head_group]))
                + state_layer_offset + int64_t(local_head) * HeadDim * HeadDim;

            Array<float, tile_v> vec_S[v_iters][tile_k];
            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                PRAGMA_UNROLL
                for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                    Array<StateT, tile_v> tmp;
                    Load(tmp, &state[(offset_k * tile_k + k_idx) * HeadDim + (offset_v + v_iter * v_threads) * tile_v]);
                    vec_S[v_iter][k_idx] = cast<float>(tmp);
                }
            }

            const float beta_value = beta[int64_t(sequence) * gate_batch_stride + value_head];
            const float decay      = expf(g[int64_t(sequence) * gate_batch_stride + value_head]);

            Array<float, tile_k> vec_k;
            Array<float, tile_k> vec_q;
            Array<T, tile_k>     tmp_k;
            Array<T, tile_k>     tmp_q;
            Load(tmp_k, &k_ptr[offset_k * tile_k]);
            Load(tmp_q, &q_ptr[offset_k * tile_k]);
            vec_k = cast<float>(tmp_k);
            vec_q = cast<float>(tmp_q);

            float kq = 0.f;
            PRAGMA_UNROLL
            for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                kq += vec_k[k_idx] * vec_q[k_idx];
            }
            PRAGMA_UNROLL
            for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                kq += __shfl_xor_sync(0xffffffff, kq, mask);
            }

            Array<T, tile_v> vec_v[v_iters];
            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                Load(vec_v[v_iter], &v_ptr[(offset_v + v_iter * v_threads) * tile_v]);
            }

            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                Array<T, tile_v> vec_out;
                PRAGMA_UNROLL
                for (int v_idx = 0; v_idx < tile_v; ++v_idx) {
                    float kv_memory = 0.f;
                    float sq        = 0.f;
                    PRAGMA_UNROLL
                    for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                        const float s_decayed       = vec_S[v_iter][k_idx][v_idx] * decay;
                        vec_S[v_iter][k_idx][v_idx] = s_decayed;
                        kv_memory += s_decayed * vec_k[k_idx];
                        sq += s_decayed * vec_q[k_idx];
                    }
                    PRAGMA_UNROLL
                    for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                        kv_memory += __shfl_xor_sync(0xffffffff, kv_memory, mask);
                        sq += __shfl_xor_sync(0xffffffff, sq, mask);
                    }
                    const float delta = (float(vec_v[v_iter][v_idx]) - kv_memory) * beta_value;
                    PRAGMA_UNROLL
                    for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                        vec_S[v_iter][k_idx][v_idx] += vec_k[k_idx] * delta;
                    }
                    vec_out[v_idx] = T((sq + delta * kq) * rsqrtf(float(HeadDim)));
                }
                if (offset_k == 0) {
                    Store(&out_ptr[(offset_v + v_iter * v_threads) * tile_v], vec_out);
                }
            }

            if (!finished[sequence]) {
                PRAGMA_UNROLL
                for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                    PRAGMA_UNROLL
                    for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                        auto tmp = cast<StateT>(vec_S[v_iter][k_idx]);
                        Store(&state[(offset_k * tile_k + k_idx) * HeadDim + (offset_v + v_iter * v_threads) * tile_v],
                              tmp);
                    }
                }
            }
        }
    }
#endif
}

template<class T, class StateT>
void RunPreSm90Recurrent(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    constexpr int kBlockDim  = 256;
    const int     total_work = plan.problem.sequence_num * plan.problem.hv;
    const int     grid       = std::min(total_work, std::max(plan.problem.sm_count, 1) * 4);
    RecurrentGdrKernel<128, kBlockDim, T, StateT>
        <<<grid, kBlockDim, 0, stream>>>(args.out->data<T>(),
                                         args.out->stride(0),
                                         args.q.data<T>(),
                                         args.q.stride(0),
                                         args.k.data<T>(),
                                         args.k.stride(0),
                                         args.v.data<T>(),
                                         args.v.stride(0),
                                         args.beta.data<float>(),
                                         args.beta.stride(0),
                                         args.g.data<float>(),
                                         reinterpret_cast<const int64_t*>(args.state_ptrs.raw_data()),
                                         args.finished.data<bool>(),
                                         total_work,
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

DEFINE_PRE_SM90_CALLBACK(RunPreSm90RecurrentF16F16, RunPreSm90Recurrent, half, half)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90RecurrentF16F32, RunPreSm90Recurrent, half, float)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90RecurrentBf16Bf16, RunPreSm90Recurrent, __nv_bfloat16, __nv_bfloat16)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90RecurrentBf16F32, RunPreSm90Recurrent, __nv_bfloat16, float)

#undef DEFINE_PRE_SM90_CALLBACK

}  // namespace turbomind::linear_attn::delta_rule::detail
