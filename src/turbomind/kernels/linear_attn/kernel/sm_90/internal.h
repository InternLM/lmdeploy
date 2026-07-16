#pragma once

#include <algorithm>
#include <cstdint>

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule {

struct Sm90GdrTmaLayout {
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

constexpr int kGdrSelectorChunk32 = 32;
constexpr int kGdrSelectorChunk64 = 64;

inline int Sm90GdrTargetCtas(const Problem& problem)
{
    return std::max(problem.sm_count * 7 / 10, 1);
}

inline int Sm90DirectFusedBlockDv(const Problem& problem)
{
    const int grid_size   = problem.sequence_num * problem.hv;
    const int target_ctas = Sm90GdrTargetCtas(problem);
    if (grid_size >= target_ctas) {
        return 128;
    }
    if (grid_size * 2 >= target_ctas) {
        return 64;
    }
    return 32;
}

inline int Sm90RecurrentPipelineStages(int)
{
    return 1;
}

inline int FusedChunkGdrBlockDv(const Problem& problem, bool context_parallel)
{
    if (context_parallel && problem.chunk_size == kGdrSelectorChunk32) {
        return 32;
    }
    if (!context_parallel || problem.chunk_size == kGdrSelectorChunk64) {
        return Sm90DirectFusedBlockDv(problem);
    }
    return 64;
}

struct Sm90DirectChunkWorkspace {
    core::Tensor     g_cumsum;
    core::Tensor     resolvent;
    Sm90GdrTmaLayout layout;
    void*            kkt_tma_desc{};
    void*            fused_tma_desc{};
};

struct Sm90ContextParallelWorkspace {
    core::Tensor     g_cumsum;
    core::Tensor     resolvent;
    core::Tensor     cp_state;
    core::Tensor     segment_state;
    core::Tensor     segment_m;
    core::Tensor     cp_q_offsets;
    core::Tensor     cp_source_indices;
    core::Tensor     cp_sequence_starts;
    core::Tensor     cp_state_ptrs;
    core::Tensor     cp_finished;
    core::Tensor     cp_fallback;
    Sm90GdrTmaLayout layout;
    void*            kkt_tma_desc{};
    void*            fused_gdr_h_tma_desc{};
    void*            correct_initial_states_tma_desc{};
    void*            context_parallel_fused_gdr_tma_desc{};
};

inline Problem MakeContextParallelProblem(const Problem& problem, const ContextParallelPlan& cp)
{
    Problem result      = problem;
    result.sequence_num = cp.total_segments;
    result.total_chunks = cp.total_chunks;
    return result;
}

bool                         PlanSm90Operation(const GdrKernelSpec&, const Operation&, const PlanningContext&, Plan*);
Sm90DirectChunkWorkspace     PartitionSm90DirectChunkWorkspace(const Arguments&, const Plan&);
Sm90ContextParallelWorkspace PartitionSm90ContextParallelWorkspace(const Arguments&, const Plan&);

void LaunchSm90Recurrent(const core::Tensor&,
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
                         cudaStream_t);
void PrepareSm90RecurrentStateTmaDescriptors(const core::Tensor&, core::Tensor&, int, int, const Plan&, cudaStream_t);
void LaunchSm90KktSolve(const core::Tensor&,
                        const core::Tensor&,
                        const core::Tensor&,
                        const core::Tensor*,
                        const core::Tensor&,
                        core::Tensor&,
                        const Problem&,
                        void*,
                        cudaStream_t);
void LaunchSm90FusedChunk(const core::Tensor&,
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
void LaunchSm90FusedGdrH(const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         core::Tensor&,
                         core::Tensor&,
                         const Problem&,
                         const ContextParallelPlan&,
                         const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         const core::Tensor&,
                         core::Tensor&,
                         void*,
                         cudaStream_t);
void LaunchSm90CorrectInitialStates(core::Tensor&,
                                    const core::Tensor&,
                                    const core::Tensor&,
                                    const core::Tensor&,
                                    const core::Tensor&,
                                    const core::Tensor&,
                                    const core::Tensor&,
                                    const Problem&,
                                    const ContextParallelPlan&,
                                    int64_t,
                                    DataType,
                                    void*,
                                    cudaStream_t);
void PrepareSm90GdrTmaDescriptorsAndCumsum(const core::Tensor&,
                                           const core::Tensor&,
                                           const core::Tensor&,
                                           const core::Tensor&,
                                           core::Tensor&,
                                           const core::Tensor&,
                                           core::Tensor&,
                                           const core::Tensor&,
                                           const core::Tensor&,
                                           core::Tensor&,
                                           core::Tensor&,
                                           const Problem&,
                                           const ContextParallelPlan&,
                                           Sm90GdrTmaLayout,
                                           DataType,
                                           cudaStream_t);

}  // namespace detail
}  // namespace turbomind::linear_attn::delta_rule
