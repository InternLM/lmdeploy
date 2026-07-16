#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::linear_attn::delta_rule {

class GdrKernel;

enum class GdrMode
{
    kRecurrent,
    kChunked
};

constexpr int kAutoGdrChunkSize      = 0;
constexpr int kRecurrentGdrChunkSize = 1;

enum class ContextParallelMode : int
{
    kAuto = 0,
    kOff  = 1,
};

struct GdrKernelSpec {
    const char* architecture{};
    GdrMode     mode{};
    DataType    input_dtype{kNull};
    DataType    state_dtype{kNull};
    int         head_dim{};
    int         chunk_size{};
};

struct Operation {
    GdrMode             mode{GdrMode::kChunked};
    int                 chunk_size{kAutoGdrChunkSize};
    ContextParallelMode cp_mode{ContextParallelMode::kAuto};
};

struct PlanningContext {
    int                  arch{};
    int                  sm_count{};
    DataType             input_dtype{kNull};
    DataType             state_dtype{kNull};
    int                  physical_batch{};
    int                  token_slots{};
    int                  hq{};
    int                  hv{};
    int                  head_dim{128};
    int64_t              gate_stride{};
    int64_t              gate_batch_stride{};
    int64_t              beta_stride{};
    int64_t              beta_batch_stride{};
    int                  num_head_groups{1};
    int                  heads_per_block{};
    std::vector<int32_t> q_offsets;
};

struct Arguments {
    core::Tensor  q, k, v, g, beta;
    core::Tensor  state_ptrs, state_tma_descs, q_offsets, finished;
    core::Tensor* out{};
    core::Tensor* workspace{};
    int64_t       state_layer_offset{};
};

struct Problem {
    int      arch{};
    int      sm_count{};
    DataType input_dtype{kNull};
    DataType state_dtype{kNull};
    int      batch{};
    int      token_num{};
    int      sequence_num{};
    int      hq{};
    int      hv{};
    int64_t  gate_stride{};
    int64_t  gate_batch_stride{};
    int64_t  beta_stride{};
    int64_t  beta_batch_stride{};
    int      head_dim{128};
    int      chunk_size{kAutoGdrChunkSize};
    int      total_chunks{};
    int      num_head_groups{1};
    int      heads_per_block{};
};

inline bool IsRecurrentGdr(const Problem& problem) noexcept
{
    return problem.chunk_size == kRecurrentGdrChunkSize;
}

inline bool IsChunkedGdr(const Problem& problem) noexcept
{
    return problem.chunk_size > kRecurrentGdrChunkSize;
}

struct TensorPlan {
    core::Layout layout;
    DataType     dtype{kNull};
    size_t       storage_size{};
};

struct ContextParallelPlan {
    bool   enabled{};
    int    segment_tokens{};
    int    segment_chunks{};
    int    total_segments{};
    int    total_chunks{};
    size_t workspace_bytes{};

    TensorPlan cp_q_offsets;
    TensorPlan cp_source_indices;
    TensorPlan cp_sequence_starts;
    TensorPlan cp_state_ptrs;
    TensorPlan cp_finished;
    TensorPlan cp_fallback;
};

struct Plan {
    const GdrKernel*    kernel{};
    Problem             problem;
    ContextParallelPlan cp;
    TensorPlan          out, g_cumsum, resolvent, workspace;
    size_t              workspace_bytes{};
    size_t              state_tma_desc_bytes_per_layer_group{};
};

class GatedDeltaRule {
public:
    bool Plan(const Operation&, const PlanningContext&, delta_rule::Plan*) const;
    void PrepareState(const core::Tensor& state_ptrs,
                      core::Tensor&       state_tma_descs,
                      int                 layer_groups,
                      int                 layers_per_block,
                      const delta_rule::Plan&,
                      cudaStream_t) const;
    void Run(const Arguments&, const delta_rule::Plan&, cudaStream_t) const;
};

}  // namespace turbomind::linear_attn::delta_rule
