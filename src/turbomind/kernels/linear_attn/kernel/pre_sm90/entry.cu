#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/linear_attn/kernel/pre_sm90/internal.h"
#include "src/turbomind/kernels/linear_attn/registry.h"

#include <memory>

namespace turbomind::linear_attn::delta_rule {

namespace {

template<GdrMode Mode, DataType InputType, DataType StateType>
class PreSm90GdrKernel final: public GdrKernel {
public:
    const GdrKernelSpec& spec() const noexcept override
    {
        static constexpr GdrKernelSpec kSpec{
            "pre_sm90", Mode, InputType, StateType, 128, Mode == GdrMode::kRecurrent ? 1 : 16};
        return kSpec;
    }

    const char* name() const noexcept override
    {
        if constexpr (Mode == GdrMode::kRecurrent && InputType == kFloat16 && StateType == kFloat16) {
            return "pre_sm90_delta_rule_recurrent_f16_state";
        }
        else if constexpr (Mode == GdrMode::kRecurrent && InputType == kFloat16) {
            return "pre_sm90_delta_rule_recurrent_f32_state";
        }
        else if constexpr (Mode == GdrMode::kChunked && InputType == kFloat16 && StateType == kFloat16) {
            return "pre_sm90_delta_rule_chunk16_f16_state";
        }
        else if constexpr (Mode == GdrMode::kChunked && InputType == kFloat16) {
            return "pre_sm90_delta_rule_chunk16_f32_state";
        }
        else if constexpr (Mode == GdrMode::kRecurrent && StateType == kBfloat16) {
            return "pre_sm90_delta_rule_recurrent_bf16_state";
        }
        else if constexpr (Mode == GdrMode::kRecurrent) {
            return "pre_sm90_delta_rule_recurrent_bf16_f32_state";
        }
        else if constexpr (StateType == kBfloat16) {
            return "pre_sm90_delta_rule_chunk16_bf16_state";
        }
        else {
            return "pre_sm90_delta_rule_chunk16_bf16_f32_state";
        }
    }

    bool Match(const Operation& operation, const PlanningContext& context) const override
    {
        return context.arch > 0 && context.arch < 900
               && detail::MatchesGdrSpec(spec(), operation, context);
    }

    bool Plan(const Operation&, const PlanningContext& context, delta_rule::Plan* plan) const override
    {
        return detail::PlanPreSm90Operation(spec(), context, plan);
    }

    void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent && InputType == kFloat16 && StateType == kFloat16) {
            detail::RunPreSm90RecurrentF16F16(args, plan, stream);
        }
        else if constexpr (Mode == GdrMode::kRecurrent && InputType == kFloat16) {
            detail::RunPreSm90RecurrentF16F32(args, plan, stream);
        }
        else if constexpr (Mode == GdrMode::kChunked && InputType == kFloat16 && StateType == kFloat16) {
            detail::RunPreSm90Chunk16F16F16(args, plan, stream);
        }
        else if constexpr (Mode == GdrMode::kChunked && InputType == kFloat16) {
            detail::RunPreSm90Chunk16F16F32(args, plan, stream);
        }
        else if constexpr (Mode == GdrMode::kRecurrent && StateType == kBfloat16) {
            detail::RunPreSm90RecurrentBf16Bf16(args, plan, stream);
        }
        else if constexpr (Mode == GdrMode::kRecurrent) {
            detail::RunPreSm90RecurrentBf16F32(args, plan, stream);
        }
        else if constexpr (StateType == kBfloat16) {
            detail::RunPreSm90Chunk16Bf16Bf16(args, plan, stream);
        }
        else {
            detail::RunPreSm90Chunk16Bf16F32(args, plan, stream);
        }
    }
};

template<GdrMode Mode, DataType InputType, DataType StateType>
void Add(GdrKernelRegistry& registry)
{
    registry.Add(std::make_unique<PreSm90GdrKernel<Mode, InputType, StateType>>());
}

bool RegisterPreSm90GdrOperations()
{
    auto& registry = GdrKernelRegistry::instance();
    Add<GdrMode::kRecurrent, kFloat16, kFloat16>(registry);
    Add<GdrMode::kRecurrent, kFloat16, kFloat32>(registry);
    Add<GdrMode::kChunked, kFloat16, kFloat16>(registry);
    Add<GdrMode::kChunked, kFloat16, kFloat32>(registry);
    Add<GdrMode::kRecurrent, kBfloat16, kBfloat16>(registry);
    Add<GdrMode::kRecurrent, kBfloat16, kFloat32>(registry);
    Add<GdrMode::kChunked, kBfloat16, kBfloat16>(registry);
    Add<GdrMode::kChunked, kBfloat16, kFloat32>(registry);
    return true;
}

[[maybe_unused]] const bool kPreSm90GdrOperationsRegistered = RegisterPreSm90GdrOperations();

}  // namespace

}  // namespace turbomind::linear_attn::delta_rule
