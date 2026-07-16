#pragma once

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule {

enum class Sm120GdrTmaMode : int {
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
};

namespace detail {

struct Sm120DirectChunkWorkspace {
    core::Tensor g_cumsum;
    core::Tensor resolvent;
    Sm120GdrTmaLayout layout;
    void* kkt_tma_desc{};
    void* fused_tma_desc{};
};

bool PlanSm120Operation(const GdrKernelSpec&, const PlanningContext&, Plan*);
Sm120DirectChunkWorkspace PartitionSm120DirectChunkWorkspace(const Arguments&, const Plan&);

void LaunchChunk32LocalCumsum(const core::Tensor&,
                              const core::Tensor&,
                              core::Tensor&,
                              const Problem&,
                              cudaStream_t);
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
                          DataType,
                          cudaStream_t);
void PrepareSm120RecurrentStateTmaDescriptors(const core::Tensor&,
                                              core::Tensor&,
                                              int,
                                              int,
                                              const Plan&,
                                              cudaStream_t);
void LaunchSm120KktSolve(const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor*,
                         const core::Tensor&,
                         core::Tensor&,
                         const Problem&,
                         void*,
                         cudaStream_t);
void LaunchSm120FusedChunk(const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           const core::Tensor&,
                           core::Tensor&,
                           const Problem&,
                           int64_t,
                           DataType,
                           const core::Tensor*,
                           const core::Tensor*,
                           const core::Tensor*,
                           int,
                           void*,
                           cudaStream_t);
void PrepareSm120GdrTmaDescriptors(const core::Tensor&,
                                   const core::Tensor&,
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
                                   DataType,
                                   cudaStream_t);

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
