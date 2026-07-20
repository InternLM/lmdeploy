// Correct the initial state of every context-parallel segment.

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class StateT>
struct Sm120CorrectInitialStates {
    static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>,
                  "context-parallel StateT must be float or bfloat16");

    static constexpr int kThreads       = 1024;
    static constexpr int kStateElements = kHeadDim * kHeadDim;

    static __device__ __forceinline__ StateT* GroupedStateBase(const int64_t* state_ptrs,
                                                               int            sequence,
                                                               int            value_head,
                                                               int            num_head_groups,
                                                               int            heads_per_block,
                                                               int64_t        state_layer_offset)
    {
        const int     head_group = value_head / heads_per_block;
        const int     local_head = value_head % heads_per_block;
        const int64_t address    = state_ptrs[sequence * num_head_groups + head_group];
        return reinterpret_cast<StateT*>(static_cast<uintptr_t>(address)) + state_layer_offset
               + static_cast<int64_t>(local_head) * kStateElements;
    }

    static __device__ __forceinline__ float LoadState(const StateT* state, int64_t offset)
    {
        if constexpr (std::is_same_v<StateT, __nv_bfloat16>) {
            return __bfloat162float(state[offset]);
        }
        else {
            return state[offset];
        }
    }

    static __device__ __forceinline__ void Run(float* __restrict__ cp_state,
                                               const float* __restrict__ segment_state,
                                               const float* __restrict__ segment_m,
                                               const bool* __restrict__ cp_fallback,
                                               const int32_t* __restrict__ cp_sequence_starts,
                                               const int64_t* __restrict__ state_ptrs,
                                               int64_t state_layer_offset,
                                               int     num_head_groups,
                                               int     heads_per_block,
                                               int     sequence_num,
                                               int     hv)
    {
        const int sequence_id = static_cast<int>(blockIdx.x);
        const int value_head  = static_cast<int>(blockIdx.y);
        if (sequence_id >= sequence_num || value_head >= hv) {
            return;
        }
        const int first_segment = cp_sequence_starts[sequence_id];
        const int last_segment  = cp_sequence_starts[sequence_id + 1];
        if (first_segment == last_segment) {
            return;
        }

        auto* initial_state =
            GroupedStateBase(state_ptrs, sequence_id, value_head, num_head_groups, heads_per_block, state_layer_offset);
        float* first_state = cp_state + (static_cast<int64_t>(first_segment) * hv + value_head) * kStateElements;
        for (int element = static_cast<int>(threadIdx.x); element < kStateElements; element += blockDim.x) {
            first_state[element] = LoadState(initial_state, element);
        }
        __syncthreads();

        for (int segment = first_segment; segment + 1 < last_segment; ++segment) {
            const int64_t head_offset = (static_cast<int64_t>(segment) * hv + value_head) * kStateElements;
            const float*  previous    = cp_state + head_offset;
            const float*  transition  = segment_m + head_offset;
            float*        next     = cp_state + (static_cast<int64_t>(segment + 1) * hv + value_head) * kStateElements;
            const bool    fallback = cp_fallback[static_cast<int64_t>(segment) * hv + value_head];

            if (fallback) {
                for (int element = static_cast<int>(threadIdx.x); element < kStateElements; element += blockDim.x) {
                    float     value      = next[element];
                    const int row        = element / kHeadDim;
                    const int col        = element - row * kHeadDim;
                    float     correction = 0.0f;
#pragma unroll 4
                    for (int k = 0; k < kHeadDim; ++k) {
                        correction = fmaf(transition[row * kHeadDim + k], previous[k * kHeadDim + col], correction);
                    }
                    value += correction;
                    next[element] = value;
                }
            }
            __syncthreads();
        }
    }
};

template<class StateT>
__global__ void Sm120CorrectInitialStatesKernel(float* __restrict__ cp_state,
                                                const float* __restrict__ segment_state,
                                                const float* __restrict__ segment_m,
                                                const bool* __restrict__ cp_fallback,
                                                const int32_t* __restrict__ cp_sequence_starts,
                                                const int64_t* __restrict__ state_ptrs,
                                                int64_t state_layer_offset,
                                                int     num_head_groups,
                                                int     heads_per_block,
                                                int     sequence_num,
                                                int     hv)
{
    Sm120CorrectInitialStates<StateT>::Run(cp_state,
                                           segment_state,
                                           segment_m,
                                           cp_fallback,
                                           cp_sequence_starts,
                                           state_ptrs,
                                           state_layer_offset,
                                           num_head_groups,
                                           heads_per_block,
                                           sequence_num,
                                           hv);
}

template<class StateT>
void LaunchSm120CorrectInitialStatesTyped(core::Tensor&              cp_state,
                                          const core::Tensor&        state_ptrs,
                                          const core::Tensor&        cp_sequence_starts,
                                          const core::Tensor&        segment_state,
                                          const core::Tensor&        segment_m,
                                          const core::Tensor&        cp_fallback,
                                          const Problem&             problem,
                                          const ContextParallelPlan& cp,
                                          int64_t                    state_layer_offset,
                                          cudaStream_t               stream)
{
    using Kernel = Sm120CorrectInitialStates<StateT>;
    static_cast<void>(cp);
    const dim3 grid(problem.sequence_num, problem.hv, 1);
    Sm120CorrectInitialStatesKernel<StateT>
        <<<grid, Kernel::kThreads, 0, stream>>>(cp_state.data<float>(),
                                                segment_state.data<float>(),
                                                segment_m.data<float>(),
                                                cp_fallback.data<bool>(),
                                                cp_sequence_starts.data<int32_t>(),
                                                reinterpret_cast<const int64_t*>(state_ptrs.raw_data()),
                                                state_layer_offset,
                                                problem.num_head_groups,
                                                problem.heads_per_block,
                                                problem.sequence_num,
                                                problem.hv);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
