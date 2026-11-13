// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/linear_attn/gdn_state_transaction.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <algorithm>
#include <cstdint>

namespace turbomind::linear_attn::delta_rule {
namespace {

__global__ void BuildGdnStateStoreMaskKernel(
    bool* out, const bool* finished, const bool* speculative, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = finished[i] || speculative[i];
    }
}

struct CaptureStrides {
    int64_t raw_token;
    int64_t key_token;
    int64_t key_head;
    int64_t value_token;
    int64_t value_head;
    int64_t log_decay_token;
    int64_t beta_token;
};

template<class T>
__global__ void CaptureGdnTransitionsKernel(const T* __restrict__ raw_projection,
                                            const T* __restrict__ normalized_key,
                                            const T* __restrict__ value,
                                            const float* __restrict__ log_decay,
                                            const float* __restrict__ beta,
                                            const int* __restrict__ q_offsets,
                                            const int* __restrict__ speculative_request_indices,
                                            T* __restrict__ journal_raw,
                                            T* __restrict__ journal_key,
                                            T* __restrict__ journal_value,
                                            float* __restrict__ journal_decay,
                                            float* __restrict__ journal_beta,
                                            CaptureStrides strides,
                                            int speculative_count,
                                            int verify_positions,
                                            int conv_dim,
                                            int hq,
                                            int hv)
{
    const int compact  = blockIdx.x;
    const int position = blockIdx.y;
    const int request  = speculative_request_indices[compact];
    const int source_row = q_offsets[request] + position;
    const int journal_row = blockIdx.z * speculative_count * verify_positions
                            + compact * verify_positions + position;

    const T* raw_row = raw_projection + int64_t(source_row) * strides.raw_token;
    T* dst_raw       = journal_raw + int64_t(journal_row) * conv_dim;
    for (int col = threadIdx.x; col < conv_dim; col += blockDim.x) {
        dst_raw[col] = raw_row[col];
    }

    const T* key_row = normalized_key + int64_t(source_row) * strides.key_token;
    T* dst_key       = journal_key + int64_t(journal_row) * hq * 128;
    for (int col = threadIdx.x; col < hq * 128; col += blockDim.x) {
        const int head = col / 128;
        const int dim  = col % 128;
        dst_key[col]   = key_row[int64_t(head) * strides.key_head + dim];
    }

    const T* value_row = value + int64_t(source_row) * strides.value_token;
    T* dst_value       = journal_value + int64_t(journal_row) * hv * 128;
    for (int col = threadIdx.x; col < hv * 128; col += blockDim.x) {
        const int head = col / 128;
        const int dim  = col % 128;
        dst_value[col] = value_row[int64_t(head) * strides.value_head + dim];
    }

    const float* decay_row = log_decay + int64_t(source_row) * strides.log_decay_token;
    const float* beta_row  = beta + int64_t(source_row) * strides.beta_token;
    float* dst_decay       = journal_decay + int64_t(journal_row) * hv;
    float* dst_beta        = journal_beta + int64_t(journal_row) * hv;
    for (int head = threadIdx.x; head < hv; head += blockDim.x) {
        dst_decay[head] = decay_row[head];
        dst_beta[head]  = beta_row[head];
    }
}

template<class T>
__global__ void CommitAcceptedConvStateKernel(const T* __restrict__ raw_conv,
                                               void* const* __restrict__ conv_state_ptrs,
                                               const int* __restrict__ request_indices,
                                               const int* __restrict__ entry_sequence_length,
                                               const int* __restrict__ accept_len,
                                               const int* __restrict__ conv_state_offsets,
                                               int speculative_count,
                                               int position_count,
                                               int conv_dim,
                                               int d_conv)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int compact = blockIdx.y;
    const int layer   = blockIdx.z;
    if (channel >= conv_dim) {
        return;
    }

    const int request = request_indices[compact];
    const int entry   = entry_sequence_length[request];
    T* state = static_cast<T*>(conv_state_ptrs[request]) + conv_state_offsets[layer];
    const T* journal = raw_conv
                       + (int64_t(layer) * speculative_count + compact) * position_count * conv_dim;
    const int accepted = accept_len[request];
    for (int position = 0; position < accepted; ++position) {
        const int ring = (entry - 1 + position) % d_conv;
        state[int64_t(ring) * conv_dim + channel] = journal[int64_t(position) * conv_dim + channel];
    }
}

template<class T>
__device__ __forceinline__ float ToFloat(T value)
{
    return static_cast<float>(value);
}

template<class T>
__device__ __forceinline__ T FromFloat(float value)
{
    return static_cast<T>(value);
}

template<class InputT, class StateT>
__global__ void CommitAcceptedRecurrentStateKernel(const InputT* __restrict__ key,
                                                    const InputT* __restrict__ value,
                                                    const float* __restrict__ log_decay,
                                                    const float* __restrict__ beta,
                                                    void* const* __restrict__ recurrent_state_ptrs,
                                                    const int* __restrict__ request_indices,
                                                    const int* __restrict__ accept_len,
                                                    int64_t layer_group_stride,
                                                    int64_t request_stride,
                                                    int layer_count,
                                                    int speculative_count,
                                                    int position_count,
                                                    int hq,
                                                    int hv,
                                                    int num_head_groups,
                                                    int layers_per_block,
                                                    int heads_per_block,
                                                    int total_work)
{
    constexpr int kHeadDim     = 128;
    constexpr int kTileK       = 16;
    constexpr int kTileV       = 4;
    constexpr int kKeyThreads  = kHeadDim / kTileK;

    const int key_lane   = threadIdx.x % kKeyThreads;
    const int value_lane = threadIdx.x / kKeyThreads;

    for (int work = blockIdx.x; work < total_work; work += gridDim.x) {
        int index            = work;
        const int value_head = index % hv;
        index /= hv;
        const int compact = index % speculative_count;
        const int layer   = index / speculative_count;
        const int request = request_indices[compact];

        const int layer_group   = layer / layers_per_block;
        const int layer_in_group = layer % layers_per_block;
        const int head_group    = value_head / heads_per_block;
        const int local_head    = value_head % heads_per_block;
        void* part = recurrent_state_ptrs[int64_t(layer_group) * layer_group_stride
                                          + int64_t(request) * request_stride + head_group];
        StateT* state = static_cast<StateT*>(part)
                        + (int64_t(layer_in_group) * heads_per_block + local_head) * kHeadDim * kHeadDim;

        float fragment[kTileK][kTileV];
#pragma unroll
        for (int ki = 0; ki < kTileK; ++ki) {
#pragma unroll
            for (int vi = 0; vi < kTileV; ++vi) {
                const int key_index   = key_lane * kTileK + ki;
                const int value_index = value_lane * kTileV + vi;
                fragment[ki][vi] = ToFloat(state[int64_t(key_index) * kHeadDim + value_index]);
            }
        }

        const int accepted = accept_len[request];
        const int key_head = value_head / (hv / hq);
        for (int position = 0; position < accepted; ++position) {
            const int64_t row = (int64_t(layer) * speculative_count + compact) * position_count + position;
            const InputT* key_row = key + (row * hq + key_head) * kHeadDim;
            const InputT* value_row = value + (row * hv + value_head) * kHeadDim;
            const float decay = exp2f(log_decay[row * hv + value_head] * 1.4426950408889634f);
            const float beta_value = beta[row * hv + value_head];

            float prediction[kTileV]{};
#pragma unroll
            for (int ki = 0; ki < kTileK; ++ki) {
                const float key_value = ToFloat(key_row[key_lane * kTileK + ki]);
#pragma unroll
                for (int vi = 0; vi < kTileV; ++vi) {
                    fragment[ki][vi] *= decay;
                    prediction[vi] += fragment[ki][vi] * key_value;
                }
            }
#pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
                for (int vi = 0; vi < kTileV; ++vi) {
                    prediction[vi] += __shfl_xor_sync(0xffffffffu, prediction[vi], offset);
                }
            }

            float delta[kTileV];
#pragma unroll
            for (int vi = 0; vi < kTileV; ++vi) {
                delta[vi] = (ToFloat(value_row[value_lane * kTileV + vi]) - prediction[vi]) * beta_value;
            }
#pragma unroll
            for (int ki = 0; ki < kTileK; ++ki) {
                const float key_value = ToFloat(key_row[key_lane * kTileK + ki]);
#pragma unroll
                for (int vi = 0; vi < kTileV; ++vi) {
                    fragment[ki][vi] += key_value * delta[vi];
                }
            }
        }

#pragma unroll
        for (int ki = 0; ki < kTileK; ++ki) {
#pragma unroll
            for (int vi = 0; vi < kTileV; ++vi) {
                const int key_index   = key_lane * kTileK + ki;
                const int value_index = value_lane * kTileV + vi;
                state[int64_t(key_index) * kHeadDim + value_index] = FromFloat<StateT>(fragment[ki][vi]);
            }
        }
    }
}

}  // namespace

void invokeBuildGdnStateStoreMask(bool*        suppress_state_store,
                                  const bool*  finished,
                                  const bool*  speculative_row,
                                  int          request_count,
                                  cudaStream_t stream)
{
    if (request_count == 0) {
        return;
    }
    const int block = 256;
    const int grid  = (request_count + block - 1) / block;
    BuildGdnStateStoreMaskKernel<<<grid, block, 0, stream>>>(
        suppress_state_store, finished, speculative_row, request_count);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeCaptureGdnTransitions(const Tensor&       raw_projection,
                                 const Tensor&       normalized_key,
                                 const Tensor&       value,
                                 const Tensor&       log_decay,
                                 const Tensor&       beta,
                                 const Buffer_<int>& q_offsets,
                                 const Buffer_<int>& speculative_request_indices,
                                 int                 gdn_layer,
                                 int                 verify_positions,
                                 TransitionJournal   journal,
                                 cudaStream_t        stream)
{
    const int speculative_count = speculative_request_indices.size();
    const int conv_dim           = raw_projection.shape(1);
    const int hq                 = normalized_key.shape(2);
    const int hv                 = value.shape(2);
    const CaptureStrides strides{raw_projection.stride(0),
                                 normalized_key.stride(1),
                                 normalized_key.stride(2),
                                 value.stride(1),
                                 value.stride(2),
                                 log_decay.stride(1),
                                 beta.stride(1)};
    const dim3 grid(speculative_count, verify_positions, 1);
    auto launch = [&](auto type) {
        using T = decltype(type);
        CaptureGdnTransitionsKernel<<<grid, 256, 0, stream>>>(raw_projection.data<T>(),
                                                              normalized_key.data<T>(),
                                                              value.data<T>(),
                                                              log_decay.data<float>(),
                                                              beta.data<float>(),
                                                              q_offsets.data(),
                                                              speculative_request_indices.data(),
                                                              journal.raw_conv.data<T>()
                                                                  + int64_t(gdn_layer) * speculative_count
                                                                        * verify_positions * conv_dim,
                                                              journal.key.data<T>()
                                                                  + int64_t(gdn_layer) * speculative_count
                                                                        * verify_positions * hq * 128,
                                                              journal.value.data<T>()
                                                                  + int64_t(gdn_layer) * speculative_count
                                                                        * verify_positions * hv * 128,
                                                              journal.log_decay.data<float>()
                                                                  + int64_t(gdn_layer) * speculative_count
                                                                        * verify_positions * hv,
                                                              journal.beta.data<float>()
                                                                  + int64_t(gdn_layer) * speculative_count
                                                                        * verify_positions * hv,
                                                              strides,
                                                              speculative_count,
                                                              verify_positions,
                                                              conv_dim,
                                                              hq,
                                                              hv);
    };
    TM_DISPATCH_DTYPES(raw_projection.dtype(), launch, half_t, bfloat16_t);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeCommitAcceptedConvState(const Tensor&         raw_conv,
                                   const Buffer_<void*>& conv_state_ptrs,
                                   const Buffer_<int>&   speculative_request_indices,
                                   const Buffer_<int>&   entry_sequence_length,
                                   const Buffer_<int>&   accept_len,
                                   const Buffer_<int>&   conv_state_offsets,
                                   int                   conv_dim,
                                   int                   d_conv,
                                   cudaStream_t          stream)
{
    const int layers             = raw_conv.shape(0);
    const int speculative_count  = raw_conv.shape(1);
    const int position_count     = raw_conv.shape(2);
    const dim3 grid((conv_dim + 255) / 256, speculative_count, layers);
    auto launch = [&](auto type) {
        using T = decltype(type);
        CommitAcceptedConvStateKernel<<<grid, 256, 0, stream>>>(raw_conv.data<T>(),
                                                                conv_state_ptrs.data(),
                                                                speculative_request_indices.data(),
                                                                entry_sequence_length.data(),
                                                                accept_len.data(),
                                                                conv_state_offsets.data(),
                                                                speculative_count,
                                                                position_count,
                                                                conv_dim,
                                                                d_conv);
    };
    TM_DISPATCH_DTYPES(raw_conv.dtype(), launch, half_t, bfloat16_t);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeCommitAcceptedRecurrentState(
    const AcceptedPrefixArguments& args, DataType state_dtype, cudaStream_t stream)
{
    const int total_work = args.layer_count * args.speculative_count * args.hv;
    const int grid       = std::min(total_work, args.sm_count * 4);
    const int64_t layer_group_stride = args.recurrent_state_ptrs.stride(0);
    const int64_t request_stride     = args.recurrent_state_ptrs.stride(1);

    auto launch_input = [&](auto input_type) {
        using InputT = decltype(input_type);
        auto launch_state = [&](auto state_type) {
            using StateT = decltype(state_type);
            CommitAcceptedRecurrentStateKernel<InputT, StateT><<<grid, 256, 0, stream>>>(
                args.key.data<InputT>(),
                args.value.data<InputT>(),
                args.log_decay.data<float>(),
                args.beta.data<float>(),
                args.recurrent_state_ptrs.data<void*>(),
                args.request_indices.data<int>(),
                args.accept_len.data<int>(),
                layer_group_stride,
                request_stride,
                args.layer_count,
                args.speculative_count,
                args.position_count,
                args.hq,
                args.hv,
                args.num_head_groups,
                args.layers_per_block,
                args.heads_per_block,
                total_work);
        };
        TM_DISPATCH_DTYPES(state_dtype, launch_state, bfloat16_t, float);
    };
    TM_DISPATCH_DTYPES(args.key.dtype(), launch_input, half_t, bfloat16_t);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind::linear_attn::delta_rule
