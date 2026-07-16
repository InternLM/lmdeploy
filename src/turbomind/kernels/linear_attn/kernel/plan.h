#pragma once

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule::detail {

size_t AlignUp(size_t value, size_t alignment);
void   AddWorkspaceBytes(size_t* total, size_t bytes, size_t alignment = 16);

bool                MatchesGdrSpec(const GdrKernelSpec&, const Operation&, const PlanningContext&);
Problem             BuildProblem(const PlanningContext&, const GdrKernelSpec&);
ContextParallelPlan BuildDisabledContextParallelPlan(const Problem&);
void                BuildOptimizedTensorPlans(Plan*, size_t direct_descriptor_bytes);

}  // namespace turbomind::linear_attn::delta_rule::detail
