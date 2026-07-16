#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr int kChunkedKktTileDim = kHeadDim / 2;

enum ChunkedKktTmaDescIndex : int
{
    kChunkedKktKDesc = 0,
    kChunkedKktBetaDesc,
    kChunkedKktResolventDesc,
};

static_assert(kChunkedKktTileDim == 64);

__device__ void BuildContextParallelMetadata(const int32_t* q_offsets,
                                             const bool*    finished,
                                             float*         cp_state,
                                             int64_t*       cp_state_ptrs,
                                             int32_t*       cp_q_offsets,
                                             int32_t*       cp_source_indices,
                                             int32_t*       cp_sequence_starts,
                                             bool*          cp_finished,
                                             int            sequence_num,
                                             int            hv,
                                             int            segment_tokens)
{
    int segment_id        = 0;
    cp_sequence_starts[0] = 0;
    for (int sequence_id = 0; sequence_id < sequence_num; ++sequence_id) {
        const int sequence_begin        = q_offsets[sequence_id];
        const int sequence_end          = q_offsets[sequence_id + 1];
        cp_sequence_starts[sequence_id] = segment_id;
        for (int segment_begin = sequence_begin; segment_begin < sequence_end; segment_begin += segment_tokens) {
            const int segment_limit       = segment_begin + segment_tokens;
            const int segment_end         = segment_limit < sequence_end ? segment_limit : sequence_end;
            cp_q_offsets[segment_id]      = segment_begin;
            cp_q_offsets[segment_id + 1]  = segment_end;
            cp_source_indices[segment_id] = sequence_id;
            cp_state_ptrs[segment_id]     = static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(cp_state + static_cast<int64_t>(segment_id) * hv * kHeadDim * kHeadDim));
            cp_finished[segment_id] = finished[sequence_id];
            ++segment_id;
        }
        cp_sequence_starts[sequence_id + 1] = segment_id;
    }
}

template<class K>
constexpr CUtensorMapDataType ChunkedKktTmaDataType()
{
    static_assert(std::is_same_v<K, __nv_bfloat16>, "chunked KKT descriptor prep supports only bf16 K tensors");
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

template<int ChunkSize>
CUtensorMap MakeChunkedKktTmaDesc(const core::Tensor& k)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint32_t box_dims[5] = {64u, 1u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeQkTmaDesc<__nv_bfloat16>(k, box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
}

template<int ChunkSize>
inline CUtensorMap MakeChunkedKktGateTmaDesc(const core::Tensor& gate)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[3] = {
        static_cast<uint64_t>(gate.stride(1)),
        static_cast<uint64_t>(gate.shape(1)),
        static_cast<uint64_t>(gate.shape(0)),
    };
    const uint64_t global_strides[2] = {
        static_cast<uint64_t>(gate.stride(1)) * sizeof(float),
        static_cast<uint64_t>(gate.stride(0)) * sizeof(float),
    };
    const uint32_t box_dims[3] = {4u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<float*>(gate.data<float>()),
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       3,
                       global_dims,
                       global_strides,
                       box_dims,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
}

template<int ChunkSize>
CUtensorMap MakeChunkedKktResolventTmaDesc(const core::Tensor& resolvent)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const uint64_t global_dims[4] = {
        static_cast<uint64_t>(ChunkSize),
        static_cast<uint64_t>(resolvent.shape(2)),
        static_cast<uint64_t>(resolvent.shape(1)),
        static_cast<uint64_t>(resolvent.shape(0)),
    };
    const uint64_t global_strides[3] = {
        static_cast<uint64_t>(resolvent.stride(2)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(resolvent.stride(1)) * sizeof(__nv_bfloat16),
        static_cast<uint64_t>(resolvent.stride(0)) * sizeof(__nv_bfloat16),
    };
    const uint32_t box_dims[4] = {static_cast<uint32_t>(ChunkSize), 1u, static_cast<uint32_t>(ChunkSize), 1u};
    return MakeTmaDesc(const_cast<__nv_bfloat16*>(resolvent.data<__nv_bfloat16>()),
                       ChunkedKktTmaDataType<__nv_bfloat16>(),
                       4,
                       global_dims,
                       global_strides,
                       box_dims,
                       FusedGdrSquareTmaSwizzle<ChunkSize>());
}

template<int ChunkSize, class K>
__device__ __forceinline__ void ChunkedKktBuildTmaDescriptors(CUtensorMap*                   gmem_desc,
                                                              CUtensorMap*                   smem_desc,
                                                              const CUtensorMap&             k_tma_desc,
                                                              const CUtensorMap&             beta_tma_desc,
                                                              const CUtensorMap&             resolvent_tma_desc,
                                                              StridedTensorBase<const K>     k,
                                                              StridedTensorBase<const float> beta,
                                                              StridedTensorBase<K>           resolvent,
                                                              int                            tid,
                                                              int                            local_seq_start,
                                                              int                            physical_batch,
                                                              int                            seq_len)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const int lane_id = tid & 31;
    if (tid < 32) {
        RebaseSequenceDescriptor<3>(&gmem_desc[kChunkedKktKDesc],
                                    &smem_desc[kChunkedKktKDesc],
                                    k_tma_desc,
                                    k,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<1>(&gmem_desc[kChunkedKktBetaDesc],
                                    &smem_desc[kChunkedKktBetaDesc],
                                    beta_tma_desc,
                                    beta,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
        RebaseSequenceDescriptor<2>(&gmem_desc[kChunkedKktResolventDesc],
                                    &smem_desc[kChunkedKktResolventDesc],
                                    resolvent_tma_desc,
                                    resolvent,
                                    physical_batch,
                                    local_seq_start,
                                    seq_len,
                                    lane_id);
    }
    __syncthreads();
}

template<int ChunkSize, class T>
__device__ __forceinline__ void FusedGdrHBuildSequenceDataTmaDescriptors(CUtensorMap*               gmem_desc,
                                                                         CUtensorMap*               smem_desc,
                                                                         const CUtensorMap&         k_tma_desc,
                                                                         const CUtensorMap&         v_tma_desc,
                                                                         const CUtensorMap&         g_tma_desc,
                                                                         const CUtensorMap&         beta_tma_desc,
                                                                         const CUtensorMap&         resolvent_tma_desc,
                                                                         StridedTensorBase<const T> k,
                                                                         StridedTensorBase<const T> v,
                                                                         StridedTensorBase<const float> g_cumsum,
                                                                         StridedTensorBase<const float> beta,
                                                                         StridedTensorBase<const T>     resolvent,
                                                                         int                            tid,
                                                                         int local_sequence_begin,
                                                                         int physical_batch,
                                                                         int sequence_len)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    const int lane_id = tid & 31;
    if (tid < 32) {
        RebaseSequenceDescriptor<3>(&gmem_desc[kFusedGdrHKDesc],
                                    &smem_desc[kFusedGdrHKDesc],
                                    k_tma_desc,
                                    k,
                                    physical_batch,
                                    local_sequence_begin,
                                    sequence_len,
                                    lane_id);
        RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrHVDesc],
                                    &smem_desc[kFusedGdrHVDesc],
                                    v_tma_desc,
                                    v,
                                    physical_batch,
                                    local_sequence_begin,
                                    sequence_len,
                                    lane_id);
        RebaseSequenceDescriptor<1>(&gmem_desc[kFusedGdrHGDesc],
                                    &smem_desc[kFusedGdrHGDesc],
                                    g_tma_desc,
                                    g_cumsum,
                                    physical_batch,
                                    local_sequence_begin,
                                    sequence_len,
                                    lane_id);
        RebaseSequenceDescriptor<1>(&gmem_desc[kFusedGdrHBetaDesc],
                                    &smem_desc[kFusedGdrHBetaDesc],
                                    beta_tma_desc,
                                    beta,
                                    physical_batch,
                                    local_sequence_begin,
                                    sequence_len,
                                    lane_id);
        RebaseSequenceDescriptor<2>(&gmem_desc[kFusedGdrHResolventDesc],
                                    &smem_desc[kFusedGdrHResolventDesc],
                                    resolvent_tma_desc,
                                    resolvent,
                                    physical_batch,
                                    local_sequence_begin,
                                    sequence_len,
                                    lane_id);
    }
    __syncthreads();
}

__device__ __forceinline__ void
CopySingleTmaDescriptor(CUtensorMap* gmem_desc, CUtensorMap* smem_desc, const CUtensorMap& src_desc, int tid)
{
    const int lane_id = tid & 31;
    if (tid < 32) {
        CopyTmaDescriptor(smem_desc, &src_desc, lane_id, 32);
        __syncwarp();
        PublishTmaDescriptor(gmem_desc, smem_desc);
    }
    __syncthreads();
}

__device__ __forceinline__ void FusedGdrHBuildTmaDescriptors(CUtensorMap*       gmem_desc,
                                                             CUtensorMap*       smem_desc,
                                                             const CUtensorMap& segment_state_tma_desc,
                                                             const CUtensorMap& segment_m_tma_desc,
                                                             int                tid)
{
    const int lane_id = tid & 31;
    if (tid < 32) {
        CopyTmaDescriptor(
            &smem_desc[kFusedGdrHSegmentStateDesc - kFusedGdrHDataDescCount], &segment_state_tma_desc, lane_id, 32);
        CopyTmaDescriptor(
            &smem_desc[kFusedGdrHSegmentMDesc - kFusedGdrHDataDescCount], &segment_m_tma_desc, lane_id, 32);
        __syncwarp();

        for (int idx = 0; idx < kFusedGdrHTensorDescCount; ++idx) {
            PublishTmaDescriptor(&gmem_desc[idx], &smem_desc[idx]);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void CorrectInitialStatesBuildTmaDescriptors(CUtensorMap*       gmem_desc,
                                                                        CUtensorMap*       smem_desc,
                                                                        const CUtensorMap& cp_state_tma_desc,
                                                                        const CUtensorMap& segment_state_tma_desc,
                                                                        const CUtensorMap& segment_m_tma_desc,
                                                                        int                tid)
{
    const int lane_id = tid & 31;
    if (tid < 32) {
        CopyTmaDescriptor(&smem_desc[kCorrectInitialStatesCpStateDesc], &cp_state_tma_desc, lane_id, 32);
        CopyTmaDescriptor(&smem_desc[kCorrectInitialStatesSegmentStateDesc], &segment_state_tma_desc, lane_id, 32);
        CopyTmaDescriptor(&smem_desc[kCorrectInitialStatesSegmentMDesc], &segment_m_tma_desc, lane_id, 32);
        __syncwarp();

        for (int idx = 0; idx < kCorrectInitialStatesExternalStateDesc; ++idx) {
            PublishTmaDescriptor(&gmem_desc[idx], &smem_desc[idx]);
        }
    }
    __syncthreads();
}

template<class StateT>
__device__ __forceinline__ void
CorrectInitialStatesBuildExternalStateTmaDescriptor(CUtensorMap*       gmem_desc,
                                                    CUtensorMap*       smem_desc,
                                                    const CUtensorMap& external_state_tma_desc,
                                                    const int64_t*     state_ptrs,
                                                    int                tid,
                                                    int                sequence_id,
                                                    int                value_head,
                                                    int                num_head_groups,
                                                    int                heads_per_block,
                                                    int64_t            state_layer_offset)
{
    const int lane_id = tid & 31;
    if (tid < 32) {
        CopyTmaDescriptor(smem_desc, &external_state_tma_desc, lane_id, 32);
        __syncwarp();

        if (lane_id == 0) {
            auto* state_base = GroupedStateBase<StateT>(
                state_ptrs, sequence_id, value_head, num_head_groups, heads_per_block, state_layer_offset);
            ReplaceTmaAddress(smem_desc, state_base);
        }
        __syncwarp();

        PublishTmaDescriptor(gmem_desc, smem_desc);
    }
    __syncthreads();
}

template<class StateT, int ChunkSize>
__global__ __launch_bounds__(32, 1) void Sm120GdrTmaDescPrepareKernel(
    Sm120GdrTmaMode                        mode,
    Sm120GdrTmaLayout                      layout,
    const __grid_constant__ CUtensorMap    kkt_k_desc,
    const __grid_constant__ CUtensorMap    kkt_beta_desc,
    const __grid_constant__ CUtensorMap    kkt_resolvent_desc,
    const __grid_constant__ CUtensorMap    fused_q_desc,
    const __grid_constant__ CUtensorMap    fused_k_desc,
    const __grid_constant__ CUtensorMap    fused_v_desc,
    const __grid_constant__ CUtensorMap    fused_g_desc,
    const __grid_constant__ CUtensorMap    fused_beta_desc,
    const __grid_constant__ CUtensorMap    fused_resolvent_desc,
    const __grid_constant__ CUtensorMap    fused_state_desc,
    const __grid_constant__ CUtensorMap    fused_out_desc,
    const __grid_constant__ CUtensorMap    fused_gdr_h_v_desc,
    const __grid_constant__ CUtensorMap    context_parallel_segment_state_desc,
    const __grid_constant__ CUtensorMap    context_parallel_segment_m_desc,
    const __grid_constant__ CUtensorMap    correct_initial_states_cp_state_desc,
    const __grid_constant__ CUtensorMap    correct_initial_states_segment_state_desc,
    const __grid_constant__ CUtensorMap    correct_initial_states_segment_m_desc,
    const __grid_constant__ CUtensorMap    correct_initial_states_external_state_desc,
    const __grid_constant__ CUtensorMap    context_parallel_fused_gdr_state_desc,
    StridedTensorBase<const __nv_bfloat16> q,
    StridedTensorBase<const __nv_bfloat16> k,
    StridedTensorBase<const __nv_bfloat16> v,
    StridedTensorBase<const float>         g_cumsum,
    StridedTensorBase<const float>         beta,
    StridedTensorBase<__nv_bfloat16>       resolvent,
    StridedTensorBase<__nv_bfloat16>       out,
    const int64_t* __restrict__ state_ptrs,
    const int32_t* __restrict__ q_offsets,
    const bool* __restrict__ finished,
    void* __restrict__ workspace,
    int     sequence_num,
    int     hq,
    int     hv,
    int     num_head_groups,
    int     heads_per_block,
    int     token_num,
    int     total_segments,
    int     segment_tokens,
    int64_t gate_stride,
    int64_t gate_batch_stride,
    int64_t state_layer_offset)
{
    static_assert(kSupportedGdrChunkSize<ChunkSize>);
    static_assert(kFusedGdrValidStateT<StateT>, "chunked descriptor prep StateT must be float or bfloat16");
    __shared__ __align__(128) CUtensorMap smem_desc[kFusedGdrTmaDescCount];

    auto* base              = static_cast<char*>(workspace);
    auto* kkt_desc          = reinterpret_cast<CUtensorMap*>(base + layout.kkt_desc_offset);
    auto* direct_fused_desc = reinterpret_cast<CUtensorMap*>(base + layout.direct_fused_desc_offset);
    auto* fused_gdr_h_desc  = reinterpret_cast<CUtensorMap*>(base + layout.fused_gdr_h_desc_offset);
    auto* correct_initial_states_desc =
        reinterpret_cast<CUtensorMap*>(base + layout.correct_initial_states_desc_offset);
    auto* context_parallel_fused_gdr_desc =
        reinterpret_cast<CUtensorMap*>(base + layout.context_parallel_fused_gdr_desc_offset);
    auto* cp_q_offsets       = reinterpret_cast<int32_t*>(base + layout.cp_q_offsets_offset);
    auto* cp_source_indices  = reinterpret_cast<int32_t*>(base + layout.cp_source_indices_offset);
    auto* cp_sequence_starts = reinterpret_cast<int32_t*>(base + layout.cp_sequence_starts_offset);
    auto* cp_state_ptrs      = reinterpret_cast<int64_t*>(base + layout.cp_state_ptrs_offset);
    auto* cp_finished        = reinterpret_cast<bool*>(base + layout.cp_finished_offset);
    auto* cp_state           = reinterpret_cast<float*>(base + layout.cp_state_offset);
    const StridedTensorBase<const __nv_bfloat16> resolvent_read{
        resolvent.ptr, resolvent.batch_stride, resolvent.token_stride};

    const int tid  = static_cast<int>(threadIdx.x);
    const int task = static_cast<int>(blockIdx.x);

    if (mode == Sm120GdrTmaMode::kAllContextParallel && task == 0 && tid == 0) {
        BuildContextParallelMetadata(q_offsets,
                                     finished,
                                     cp_state,
                                     cp_state_ptrs,
                                     cp_q_offsets,
                                     cp_source_indices,
                                     cp_sequence_starts,
                                     cp_finished,
                                     sequence_num,
                                     hv,
                                     segment_tokens);
    }

    const bool needs_kkt_desc = mode == Sm120GdrTmaMode::kSolveKkt || mode == Sm120GdrTmaMode::kAllDirectFused
                                || mode == Sm120GdrTmaMode::kAllContextParallel;
    const int kkt_task_count = needs_kkt_desc ? sequence_num : 0;
    if (needs_kkt_desc && task < kkt_task_count) {
        const int seq_start = q_offsets[task];
        const int seq_end   = q_offsets[task + 1];
        const int seq_len   = seq_end - seq_start;
        if (seq_len <= 0) {
            return;
        }
        const int physical_batch  = seq_start / token_num;
        const int local_seq_start = seq_start - physical_batch * token_num;

        ChunkedKktBuildTmaDescriptors<ChunkSize>(&kkt_desc[task * kKktTmaDescCount],
                                                 smem_desc,
                                                 kkt_k_desc,
                                                 kkt_beta_desc,
                                                 kkt_resolvent_desc,
                                                 k,
                                                 beta,
                                                 resolvent,
                                                 tid,
                                                 local_seq_start,
                                                 physical_batch,
                                                 seq_len);
        return;
    }

    const int direct_task_base              = kkt_task_count;
    auto      direct_slices                 = MakeFusedGdrTmaDescriptorSlices(direct_fused_desc, sequence_num);
    auto      fused_gdr_h_slices            = MakeFusedGdrHTmaDescriptorSlices(fused_gdr_h_desc, sequence_num);
    auto      correct_initial_states_slices = MakeCorrectInitialStatesTmaDescriptorSlices(correct_initial_states_desc);
    auto      context_parallel_fused_gdr_slices =
        MakeContextParallelFusedGdrTmaDescriptorSlices(context_parallel_fused_gdr_desc, sequence_num);

    const bool needs_direct_desc      = mode == Sm120GdrTmaMode::kAllDirectFused || mode == Sm120GdrTmaMode::kFusedOnly;
    const int  direct_data_desc_count = needs_direct_desc ? sequence_num : 0;
    const int  direct_state_desc_count = needs_direct_desc ? sequence_num * hv : 0;
    const int  direct_desc_tasks       = direct_data_desc_count + direct_state_desc_count;
    if (needs_direct_desc && task >= direct_task_base && task < direct_task_base + direct_data_desc_count) {
        const int local     = task - direct_task_base;
        const int sequence  = local;
        const int seq_start = q_offsets[sequence];
        const int seq_end   = q_offsets[sequence + 1];
        const int seq_len   = seq_end - seq_start;
        if (seq_len <= 0) {
            return;
        }
        const int physical_batch  = seq_start / token_num;
        const int local_seq_start = seq_start - physical_batch * token_num;

        FusedGdrBuildSequenceDataTmaDescriptors<ChunkSize>(&direct_slices.data[sequence * kFusedGdrDataDescCount],
                                                           smem_desc,
                                                           fused_q_desc,
                                                           fused_k_desc,
                                                           fused_v_desc,
                                                           fused_g_desc,
                                                           fused_beta_desc,
                                                           fused_resolvent_desc,
                                                           fused_out_desc,
                                                           q,
                                                           k,
                                                           v,
                                                           g_cumsum,
                                                           beta,
                                                           resolvent_read,
                                                           out,
                                                           tid,
                                                           seq_start,
                                                           local_seq_start,
                                                           physical_batch,
                                                           seq_len,
                                                           hq,
                                                           hv,
                                                           gate_stride,
                                                           gate_batch_stride);
        return;
    }
    if (needs_direct_desc && task >= direct_task_base && task < direct_task_base + direct_desc_tasks) {
        const int local       = task - direct_task_base;
        const int state_local = local - direct_data_desc_count;
        const int sequence    = state_local / hv;
        const int value_head  = state_local - sequence * hv;
        const int seq_start   = q_offsets[sequence];
        const int seq_end     = q_offsets[sequence + 1];
        const int seq_len     = seq_end - seq_start;
        if (seq_len <= 0) {
            return;
        }

        const auto* state_ptr = GroupedStateBase<StateT>(
            state_ptrs, sequence, value_head, num_head_groups, heads_per_block, state_layer_offset);
        FusedGdrBuildStateTmaDescriptor(&direct_slices.state[state_local * kFusedGdrStateDescCount],
                                        &smem_desc[kFusedGdrStateDesc],
                                        fused_state_desc,
                                        state_ptr,
                                        tid);
        return;
    }

    const int fused_gdr_h_data_tasks   = mode == Sm120GdrTmaMode::kAllContextParallel ? sequence_num : 0;
    const int fused_gdr_h_tensor_tasks = mode == Sm120GdrTmaMode::kAllContextParallel ? 1 : 0;
    const int fused_gdr_h_desc_tasks   = fused_gdr_h_data_tasks + fused_gdr_h_tensor_tasks;
    const int fused_gdr_h_task_base    = direct_task_base + direct_desc_tasks;
    if (task >= fused_gdr_h_task_base && task < fused_gdr_h_task_base + fused_gdr_h_data_tasks) {
        const int sequence_id    = task - fused_gdr_h_task_base;
        const int sequence_begin = q_offsets[sequence_id];
        const int sequence_end   = q_offsets[sequence_id + 1];
        const int sequence_len   = sequence_end - sequence_begin;
        if (sequence_len <= 0) {
            return;
        }
        const int physical_batch       = sequence_begin / token_num;
        const int local_sequence_begin = sequence_begin - physical_batch * token_num;

        FusedGdrHBuildSequenceDataTmaDescriptors<ChunkSize>(
            &fused_gdr_h_slices.data[sequence_id * kFusedGdrHDataDescCount],
            smem_desc,
            fused_k_desc,
            fused_gdr_h_v_desc,
            fused_g_desc,
            fused_beta_desc,
            fused_resolvent_desc,
            k,
            v,
            g_cumsum,
            beta,
            resolvent_read,
            tid,
            local_sequence_begin,
            physical_batch,
            sequence_len);
        return;
    }
    if (task >= fused_gdr_h_task_base + fused_gdr_h_data_tasks
        && task < fused_gdr_h_task_base + fused_gdr_h_data_tasks + fused_gdr_h_tensor_tasks) {
        FusedGdrHBuildTmaDescriptors(fused_gdr_h_slices.segment_state,
                                     smem_desc,
                                     context_parallel_segment_state_desc,
                                     context_parallel_segment_m_desc,
                                     tid);
        return;
    }

    const int correct_initial_states_tensor_tasks = mode == Sm120GdrTmaMode::kAllContextParallel ? 1 : 0;
    const int correct_initial_states_external_tasks =
        mode == Sm120GdrTmaMode::kAllContextParallel ? sequence_num * hv : 0;
    const int correct_initial_states_desc_tasks =
        correct_initial_states_tensor_tasks + correct_initial_states_external_tasks;
    const int correct_initial_states_task_base = fused_gdr_h_task_base + fused_gdr_h_desc_tasks;
    if (task >= correct_initial_states_task_base
        && task < correct_initial_states_task_base + correct_initial_states_tensor_tasks) {
        CorrectInitialStatesBuildTmaDescriptors(correct_initial_states_slices.cp_state,
                                                smem_desc,
                                                correct_initial_states_cp_state_desc,
                                                correct_initial_states_segment_state_desc,
                                                correct_initial_states_segment_m_desc,
                                                tid);
        return;
    }
    if (task >= correct_initial_states_task_base + correct_initial_states_tensor_tasks
        && task < correct_initial_states_task_base + correct_initial_states_tensor_tasks
                      + correct_initial_states_external_tasks) {
        const int local          = task - correct_initial_states_task_base;
        const int external_local = local - correct_initial_states_tensor_tasks;
        const int sequence_id    = external_local / hv;
        const int value_head     = external_local - sequence_id * hv;
        CorrectInitialStatesBuildExternalStateTmaDescriptor<StateT>(
            &correct_initial_states_slices.external_state[external_local * kCorrectInitialStatesExternalDescCount],
            smem_desc,
            correct_initial_states_external_state_desc,
            state_ptrs,
            tid,
            sequence_id,
            value_head,
            num_head_groups,
            heads_per_block,
            state_layer_offset);
        return;
    }

    const int context_parallel_fused_gdr_data_tasks   = mode == Sm120GdrTmaMode::kAllContextParallel ? sequence_num : 0;
    const int context_parallel_fused_gdr_tensor_tasks = mode == Sm120GdrTmaMode::kAllContextParallel ? 1 : 0;
    const int context_parallel_fused_gdr_task_base =
        correct_initial_states_task_base + correct_initial_states_desc_tasks;
    if (task >= context_parallel_fused_gdr_task_base
        && task < context_parallel_fused_gdr_task_base + context_parallel_fused_gdr_data_tasks) {
        const int sequence_id    = task - context_parallel_fused_gdr_task_base;
        const int sequence_begin = q_offsets[sequence_id];
        const int sequence_end   = q_offsets[sequence_id + 1];
        const int sequence_len   = sequence_end - sequence_begin;
        if (sequence_len <= 0) {
            return;
        }
        const int physical_batch       = sequence_begin / token_num;
        const int local_sequence_begin = sequence_begin - physical_batch * token_num;

        FusedGdrBuildSequenceDataTmaDescriptors<ChunkSize>(
            &context_parallel_fused_gdr_slices.data[sequence_id * kFusedGdrDataDescCount],
            smem_desc,
            fused_q_desc,
            fused_k_desc,
            fused_v_desc,
            fused_g_desc,
            fused_beta_desc,
            fused_resolvent_desc,
            fused_out_desc,
            q,
            k,
            v,
            g_cumsum,
            beta,
            resolvent_read,
            out,
            tid,
            sequence_begin,
            local_sequence_begin,
            physical_batch,
            sequence_len,
            hq,
            hv,
            gate_stride,
            gate_batch_stride);
        return;
    }
    if (task >= context_parallel_fused_gdr_task_base + context_parallel_fused_gdr_data_tasks
        && task < context_parallel_fused_gdr_task_base + context_parallel_fused_gdr_data_tasks
                      + context_parallel_fused_gdr_tensor_tasks) {
        CopySingleTmaDescriptor(context_parallel_fused_gdr_slices.cp_state,
                                &smem_desc[kFusedGdrStateDesc],
                                context_parallel_fused_gdr_state_desc,
                                tid);
        return;
    }

    static_cast<void>(total_segments);
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
