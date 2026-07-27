#pragma once

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule::detail {

bool PlanPreSm90Operation(const GdrKernelSpec&, const PlanningContext&, Plan*);

void RunPreSm90RecurrentF16F16(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90RecurrentF16F32(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90RecurrentBf16Bf16(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90RecurrentBf16F32(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90Chunk16F16F16(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90Chunk16F16F32(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90Chunk16Bf16Bf16(const Arguments&, const Plan&, cudaStream_t);
void RunPreSm90Chunk16Bf16F32(const Arguments&, const Plan&, cudaStream_t);

}  // namespace turbomind::linear_attn::delta_rule::detail
