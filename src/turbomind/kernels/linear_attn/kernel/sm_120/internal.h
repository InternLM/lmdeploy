#pragma once

#include <cstdint>

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule {

enum class Sm120GdrTmaMode : int
{
    kSolveKkt,
    kFusedOnly,
    kAllDirectFused,
    kAllContextParallel,
};

struct Sm120GdrTmaLayout {
    size_t kkt_desc_offset{};
    size_t direct_fused_desc_offset{};
    size_t fused_gdr_h_desc_offset{};
    size_t correct_initial_states_desc_offset{};
    size_t context_parallel_fused_gdr_desc_offset{};
    size_t cp_state_offset{};
    size_t segment_state_offset{};
    size_t segment_m_offset{};
    size_t cp_q_offsets_offset{};
    size_t cp_source_indices_offset{};
    size_t cp_sequence_starts_offset{};
    size_t cp_state_ptrs_offset{};
    size_t cp_finished_offset{};
    size_t cp_fallback_offset{};
};

namespace detail {

struct Sm120DirectChunkWorkspace {
    core::Tensor      g_cumsum;
    core::Tensor      resolvent;
    Sm120GdrTmaLayout layout;
    void*             kkt_tma_desc{};
    void*             fused_tma_desc{};
};

struct Sm120ContextParallelWorkspace {
    core::Tensor      g_cumsum;
    core::Tensor      resolvent;
    core::Tensor      cp_state;
    core::Tensor      segment_state;
    core::Tensor      segment_m;
    core::Tensor      cp_q_offsets;
    core::Tensor      cp_source_indices;
    core::Tensor      cp_sequence_starts;
    core::Tensor      cp_state_ptrs;
    core::Tensor      cp_finished;
    core::Tensor      cp_fallback;
    Sm120GdrTmaLayout layout;
    void*             kkt_tma_desc{};
    void*             fused_gdr_h_tma_desc{};
    void*             correct_initial_states_tma_desc{};
    void*             context_parallel_fused_gdr_tma_desc{};
};

inline Problem MakeSm120ContextParallelProblem(const Problem& problem, const ContextParallelPlan& cp)
{
    Problem result             = problem;
    result.sequence_num        = cp.total_segments;
    result.total_chunks        = cp.total_chunks;
    result.max_sequence_chunks = cp.total_chunks > 0 ? cp.segment_chunks : 0;
    return result;
}

bool                          PlanSm120Operation(const GdrKernelSpec&, const Operation&, const PlanningContext&, Plan*);
Sm120DirectChunkWorkspace     PartitionSm120DirectChunkWorkspace(const Arguments&, const Plan&);
Sm120ContextParallelWorkspace PartitionSm120ContextParallelWorkspace(const Arguments&, const Plan&);

void LaunchChunk32LocalCumsum(const core::Tensor&, const core::Tensor&, core::Tensor&, const Problem&, cudaStream_t);
template<int BlockDv>
void LaunchChunk32LocalCumsumAndPrepareDirect(const core::Tensor&,
                                              const core::Tensor&,
                                              const core::Tensor&,
                                              const core::Tensor&,
                                              const core::Tensor&,
                                              core::Tensor&,
                                              core::Tensor&,
                                              core::Tensor&,
                                              void*,
                                              void*,
                                              const Problem&,
                                              cudaStream_t);
template<class StateT>
void LaunchSm120Recurrent(const core::Tensor&,
                          const core::Tensor&,
                          const core::Tensor&,
                          const core::Tensor&,
                          const core::Tensor&,
                          const core::Tensor&,
                          const core::Tensor&,
                          core::Tensor&,
                          const Problem&,
                          int64_t,
                          cudaStream_t);
template<class StateT>
void PrepareSm120RecurrentStateTmaDescriptors(const core::Tensor&, core::Tensor&, int, int, const Plan&, cudaStream_t);
void LaunchSm120KktSolve(const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor*,
                         const core::Tensor&,
                         core::Tensor&,
                         const Problem&,
                         void*,
                         cudaStream_t);
template<class StateT, int BlockDv>
void PrepareSm120GdrTmaDescriptors(const core::Tensor&,
                                   const core::Tensor&,
                                   const core::Tensor&,
                                   const core::Tensor&,
                                   const core::Tensor&,
                                   const core::Tensor&,
                                   const core::Tensor&,
                                   const core::Tensor&,
                                   core::Tensor*,
                                   core::Tensor&,
                                   const Problem&,
                                   const ContextParallelPlan&,
                                   Sm120GdrTmaMode,
                                   Sm120GdrTmaLayout,
                                   int64_t,
                                   cudaStream_t);

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
