#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"

#include <cuda.h>

#include <cstdint>
#include <utility>

namespace turbomind::linear_attn::delta_rule::detail {
namespace {

constexpr int kHeadDim                = 128;
constexpr int kTmaDescriptorBytes     = 128;
constexpr int kKktTmaDescCount        = 3;
constexpr int kFusedGdrDataDescCount  = 7;
constexpr int kFusedGdrStateDescCount = 1;

size_t KktTmaDescriptorBytes(const Problem& problem)
{
    return size_t(problem.sequence_num) * kKktTmaDescCount * kTmaDescriptorBytes;
}

size_t FusedGdrTmaDescriptorBytes(const Problem& problem)
{
    const size_t data_descs  = size_t(problem.sequence_num) * kFusedGdrDataDescCount;
    const size_t state_descs = size_t(problem.sequence_num) * problem.hv * kFusedGdrStateDescCount;
    return (data_descs + state_descs) * kTmaDescriptorBytes;
}

std::uintptr_t AlignUpAddress(std::uintptr_t value, std::uintptr_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

core::Tensor WorkspaceTensor(char* base, size_t offset, core::Layout layout, DataType dtype, core::Device device)
{
    return core::Tensor{base + offset, std::move(layout), dtype, device};
}

}  // namespace

bool PlanSm120Operation(const GdrKernelSpec& spec, const PlanningContext& context, Plan* plan)
{
    plan->problem = BuildProblem(context, spec);
    plan->cp      = BuildDisabledContextParallelPlan(plan->problem);
    const size_t descriptor_bytes =
        KktTmaDescriptorBytes(plan->problem) + FusedGdrTmaDescriptorBytes(plan->problem) + 127;
    BuildOptimizedTensorPlans(plan, descriptor_bytes);
    plan->state_tma_desc_bytes_per_layer_group =
        IsRecurrentGdr(plan->problem) ?
            size_t(plan->problem.sequence_num) * plan->problem.num_head_groups * sizeof(CUtensorMap) :
            0;
    return true;
}

Sm120DirectChunkWorkspace PartitionSm120DirectChunkWorkspace(const Arguments& args, const Plan& plan)
{
    auto*                     base   = static_cast<char*>(args.workspace->raw_data());
    size_t                    offset = 0;
    Sm120DirectChunkWorkspace out{};
    out.g_cumsum = WorkspaceTensor(base, offset, plan.g_cumsum.layout, plan.g_cumsum.dtype, args.workspace->device());
    AddWorkspaceBytes(&offset, plan.g_cumsum.storage_size * sizeof(float));
    const size_t resolvent_offset = offset == 0 ? 0 : AlignUp(offset, 16);
    out.resolvent =
        WorkspaceTensor(base, resolvent_offset, plan.resolvent.layout, plan.resolvent.dtype, args.workspace->device());
    AddWorkspaceBytes(&offset, plan.resolvent.storage_size * byte_size(plan.resolvent.dtype, 1));

    const auto address = reinterpret_cast<std::uintptr_t>(base) + offset;
    offset             = size_t(AlignUpAddress(address, kTmaDescriptorBytes) - reinterpret_cast<std::uintptr_t>(base));
    out.layout.kkt_desc_offset = offset;
    out.kkt_tma_desc           = base + offset;
    offset += KktTmaDescriptorBytes(plan.problem);
    out.layout.direct_fused_desc_offset = offset;
    out.fused_tma_desc                  = base + offset;
    return out;
}

}  // namespace turbomind::linear_attn::delta_rule::detail
