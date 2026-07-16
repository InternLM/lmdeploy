#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/linear_attn/delta_rule.h"
#include "src/turbomind/kernels/linear_attn/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace turbomind::linear_attn::delta_rule {
namespace {

using TensorPtr = std::shared_ptr<core::Tensor>;

core::Tensor TensorFromObject(const py::object& object, const char* name, bool required)
{
    if (object.is_none()) {
        if (required) {
            throw py::value_error(std::string(name) + " must be provided");
        }
        return {};
    }
    try {
        return *object.cast<TensorPtr>();
    }
    catch (const py::cast_error&) {
        throw py::type_error(std::string(name)
                             + " must be a _turbomind.Tensor; use _turbomind.from_dlpack_with_strides");
    }
}

core::Tensor* OptionalTensorPtr(const py::object& object, const char* name, std::vector<core::Tensor>& storage)
{
    if (object.is_none()) {
        return nullptr;
    }
    storage.push_back(TensorFromObject(object, name, false));
    return &storage.back();
}

Plan* PlanFromDict(const py::dict& plan)
{
    if (!plan.contains("_handle")) {
        throw py::value_error("plan must be returned by delta_rule_plan");
    }
    auto* plan_ptr = static_cast<Plan*>(PyCapsule_GetPointer(plan["_handle"].ptr(), "delta_rule.Plan"));
    if (plan_ptr == nullptr) {
        throw py::error_already_set();
    }
    return plan_ptr;
}

py::list ShapeToList(const core::Layout& layout)
{
    py::list out;
    for (int i = 0; i < layout.rank(); ++i) {
        out.append(layout.shape(i));
    }
    return out;
}

py::list StrideToList(const core::Layout& layout)
{
    py::list out;
    for (int i = 0; i < layout.rank(); ++i) {
        out.append(layout.stride(i));
    }
    return out;
}

py::dict TensorPlanToDict(const TensorPlan& plan)
{
    py::dict out;
    out["shape"]        = ShapeToList(plan.layout);
    out["stride"]       = StrideToList(plan.layout);
    out["dtype"]        = to_string(plan.dtype);
    out["storage_size"] = plan.storage_size != 0 ? plan.storage_size : static_cast<size_t>(plan.layout.cosize());
    return out;
}

py::dict ContextParallelPlanToDict(const ContextParallelPlan& plan)
{
    py::dict out;
    out["enabled"]            = plan.enabled;
    out["segment_tokens"]     = plan.segment_tokens;
    out["segment_chunks"]     = plan.segment_chunks;
    out["total_segments"]     = plan.total_segments;
    out["total_chunks"]       = plan.total_chunks;
    out["workspace_bytes"]    = plan.workspace_bytes;
    out["cp_q_offsets"]       = TensorPlanToDict(plan.cp_q_offsets);
    out["cp_source_indices"]  = TensorPlanToDict(plan.cp_source_indices);
    out["cp_sequence_starts"] = TensorPlanToDict(plan.cp_sequence_starts);
    out["cp_state_ptrs"]      = TensorPlanToDict(plan.cp_state_ptrs);
    out["cp_finished"]        = TensorPlanToDict(plan.cp_finished);
    out["cp_fallback"]        = TensorPlanToDict(plan.cp_fallback);
    return out;
}

py::dict ProblemToDict(const Problem& problem)
{
    py::dict out;
    out["arch"]              = problem.arch;
    out["sm_count"]          = problem.sm_count;
    out["input_dtype"]       = to_string(problem.input_dtype);
    out["batch"]             = problem.batch;
    out["token_num"]         = problem.token_num;
    out["sequence_num"]      = problem.sequence_num;
    out["hq"]                = problem.hq;
    out["hv"]                = problem.hv;
    out["gate_stride"]       = problem.gate_stride;
    out["gate_batch_stride"] = problem.gate_batch_stride;
    out["head_dim"]          = problem.head_dim;
    out["chunk_size"]        = problem.chunk_size;
    out["total_chunks"]      = problem.total_chunks;
    out["num_head_groups"]   = problem.num_head_groups;
    out["heads_per_block"]   = problem.heads_per_block;
    out["recurrent"]         = IsRecurrentGdr(problem);
    return out;
}

const char* ToString(GdrMode mode)
{
    switch (mode) {
        case GdrMode::kRecurrent:
            return "recurrent";
        case GdrMode::kChunked:
            return "chunked";
    }
    throw py::value_error("invalid GDR mode");
}

py::dict KernelSpecToDict(const GdrKernelSpec& spec)
{
    py::dict out;
    out["arch"]        = spec.architecture;
    out["mode"]        = ToString(spec.mode);
    out["input_dtype"] = to_string(spec.input_dtype);
    out["state_dtype"] = to_string(spec.state_dtype);
    out["head_dim"]    = spec.head_dim;
    out["chunk_size"]  = spec.chunk_size;
    return out;
}

py::dict PlanToDict(const Plan& plan)
{
    py::dict out;
    out["kernel"]          = KernelSpecToDict(plan.kernel->spec());
    out["problem"]         = ProblemToDict(plan.problem);
    out["cp"]              = ContextParallelPlanToDict(plan.cp);
    out["out"]             = TensorPlanToDict(plan.out);
    out["g_cumsum"]        = TensorPlanToDict(plan.g_cumsum);
    out["resolvent"]       = TensorPlanToDict(plan.resolvent);
    out["workspace"]       = TensorPlanToDict(plan.workspace);
    out["workspace_bytes"] = plan.workspace_bytes;
    return out;
}

ContextParallelMode ParseContextParallelMode(const std::string& mode)
{
    if (mode == "auto") {
        return ContextParallelMode::kAuto;
    }
    if (mode == "off") {
        return ContextParallelMode::kOff;
    }
    throw py::value_error("cp_mode must be one of: auto, off");
}

GdrMode ParseMode(const std::string& mode)
{
    if (mode == "recurrent") {
        return GdrMode::kRecurrent;
    }
    if (mode == "chunked") {
        return GdrMode::kChunked;
    }
    throw py::value_error("mode must be one of: recurrent, chunked");
}

DataType ParseStateDtype(const std::string& dtype)
{
    if (dtype == "f16" || dtype == "float16") {
        return kFloat16;
    }
    if (dtype == "f32" || dtype == "float32") {
        return kFloat32;
    }
    if (dtype == "bf16" || dtype == "bfloat16") {
        return kBfloat16;
    }
    throw py::value_error("state_dtype must be one of: f16, float16, f32, float32, bf16, bfloat16");
}

Operation MakeOperation(GdrMode mode, std::optional<int> chunk_size, ContextParallelMode cp_mode)
{
    return Operation{mode, chunk_size.value_or(kAutoGdrChunkSize), cp_mode};
}

PlanningContext MakePlanningContext(const core::Tensor& q,
                                    const core::Tensor& v,
                                    const core::Tensor& g,
                                    DataType            state_dtype,
                                    GdrMode             mode,
                                    const core::Tensor& q_offsets,
                                    int                 num_head_groups,
                                    int                 requested_heads_per_block)
{
    PlanningContext context{};
    context.arch              = getSMVersion() * 10;
    context.sm_count          = getSMCount();
    context.input_dtype       = q.dtype();
    context.state_dtype       = state_dtype;
    context.physical_batch    = static_cast<int>(q.shape(0));
    context.token_slots       = static_cast<int>(q.shape(1));
    context.hq                = static_cast<int>(q.shape(2));
    context.hv                = static_cast<int>(v.shape(2));
    context.head_dim          = static_cast<int>(q.shape(3));
    context.gate_stride       = g.stride(1);
    context.gate_batch_stride = g.stride(0);
    context.num_head_groups   = num_head_groups;
    context.heads_per_block   = requested_heads_per_block == 0 ? context.hv : requested_heads_per_block;
    if (mode == GdrMode::kChunked) {
        context.q_offsets.resize(q_offsets.size());
        TM_CUDA_CHECK(
            cudaMemcpy(context.q_offsets.data(), q_offsets.raw_data(), q_offsets.byte_size(), cudaMemcpyDeviceToHost));
    }
    return context;
}

Arguments MakeExecutionArguments(const py::object&          q,
                                 const py::object&          k,
                                 const py::object&          v,
                                 const py::object&          g,
                                 const py::object&          beta,
                                 const py::object&          out,
                                 const py::object&          workspace,
                                 const py::object&          state_ptrs,
                                 const py::object&          state_tma_descs,
                                 const py::object&          q_offsets,
                                 const py::object&          finished,
                                 int64_t                    state_layer_offset,
                                 std::vector<core::Tensor>& storage)
{
    Arguments args{};
    args.q                  = TensorFromObject(q, "q", true);
    args.k                  = TensorFromObject(k, "k", true);
    args.v                  = TensorFromObject(v, "v", true);
    args.g                  = TensorFromObject(g, "g", true);
    args.beta               = TensorFromObject(beta, "beta", true);
    args.state_ptrs         = TensorFromObject(state_ptrs, "state_ptrs", false);
    args.state_tma_descs    = TensorFromObject(state_tma_descs, "state_tma_descs", false);
    args.q_offsets          = TensorFromObject(q_offsets, "q_offsets", false);
    args.finished           = TensorFromObject(finished, "finished", false);
    args.out                = OptionalTensorPtr(out, "out", storage);
    args.workspace          = OptionalTensorPtr(workspace, "workspace", storage);
    args.state_layer_offset = state_layer_offset;
    return args;
}

core::Tensor PromoteStatePointers(core::Tensor state_ptrs)
{
    if (state_ptrs.ndim() == 3) {
        return state_ptrs;
    }
    if (state_ptrs.ndim() == 1) {
        const auto sequences = state_ptrs.shape(0);
        return core::Tensor{
            state_ptrs.buffer(),
            core::Layout{{1, sequences, 1}, {sequences * state_ptrs.stride(0), state_ptrs.stride(0), 1}},
            core::Tensor::PreserveBufferCapacity{}};
    }
    else {
        const auto sequences = state_ptrs.shape(0);
        const auto groups    = state_ptrs.shape(1);
        return core::Tensor{
            state_ptrs.buffer(),
            core::Layout{{1, sequences, groups},
                         {sequences * state_ptrs.stride(0), state_ptrs.stride(0), state_ptrs.stride(1)}},
            core::Tensor::PreserveBufferCapacity{}};
    }
}

py::dict PlanBridge(const py::object&  q,
                    const py::object&  k,
                    const py::object&  v,
                    const py::object&  g,
                    const py::object&  beta,
                    const py::object&  q_offsets,
                    const std::string& state_dtype,
                    const std::string& mode,
                    std::optional<int> chunk_size,
                    const std::string& cp_mode,
                    int                num_head_groups,
                    int                heads_per_block)
{
    const auto q_tensor         = TensorFromObject(q, "q", true);
    const auto k_tensor         = TensorFromObject(k, "k", true);
    const auto v_tensor         = TensorFromObject(v, "v", true);
    const auto g_tensor         = TensorFromObject(g, "g", true);
    const auto beta_tensor      = TensorFromObject(beta, "beta", true);
    const auto q_offsets_tensor = TensorFromObject(q_offsets, "q_offsets", false);
    static_cast<void>(k_tensor);
    static_cast<void>(beta_tensor);

    const auto parsed_mode        = ParseMode(mode);
    const auto parsed_state_dtype = ParseStateDtype(state_dtype);
    const auto operation          = MakeOperation(parsed_mode, chunk_size, ParseContextParallelMode(cp_mode));
    const auto context            = MakePlanningContext(q_tensor,
                                             v_tensor,
                                             g_tensor,
                                             parsed_state_dtype,
                                             parsed_mode,
                                             q_offsets_tensor,
                                             num_head_groups,
                                             heads_per_block);

    GatedDeltaRule rule;
    Plan           plan;
    try {
        if (!rule.Plan(operation, context, &plan)) {
            throw py::value_error("registered GDR planner declined the supported workload");
        }
    }
    catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }

    auto out         = PlanToDict(plan);
    auto plan_handle = std::make_unique<Plan>(plan);
    auto capsule     = py::capsule(plan_handle.get(), "delta_rule.Plan", [](PyObject* capsule) {
        delete static_cast<Plan*>(PyCapsule_GetPointer(capsule, "delta_rule.Plan"));
    });
    plan_handle.release();
    out["_handle"] = capsule;
    return out;
}

void PrepareStateTmaDescBridge(const py::object& state_ptrs,
                               const py::object& state_tma_descs,
                               const py::dict&   plan,
                               int               layer_groups,
                               int               layers_per_block,
                               std::uintptr_t    stream_ptr)
{
    auto           state_ptrs_tensor      = PromoteStatePointers(TensorFromObject(state_ptrs, "state_ptrs", true));
    auto           state_tma_descs_tensor = TensorFromObject(state_tma_descs, "state_tma_descs", true);
    auto*          plan_ptr               = PlanFromDict(plan);
    GatedDeltaRule rule;
    try {
        rule.PrepareState(state_ptrs_tensor,
                          state_tma_descs_tensor,
                          layer_groups,
                          layers_per_block,
                          *plan_ptr,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

void RunBridge(const py::object& q,
               const py::object& k,
               const py::object& v,
               const py::object& g,
               const py::object& beta,
               const py::dict&   plan,
               const py::object& out,
               const py::object& workspace,
               std::uintptr_t    stream_ptr,
               const py::object& state_ptrs,
               const py::object& state_tma_descs,
               const py::object& q_offsets,
               const py::object& finished,
               int64_t           state_layer_offset)
{
    auto*                     plan_ptr = PlanFromDict(plan);
    std::vector<core::Tensor> storage;
    storage.reserve(2);
    Arguments      args = MakeExecutionArguments(q,
                                            k,
                                            v,
                                            g,
                                            beta,
                                            out,
                                            workspace,
                                            state_ptrs,
                                            state_tma_descs,
                                            q_offsets,
                                            finished,
                                            state_layer_offset,
                                            storage);
    GatedDeltaRule rule;
    try {
        rule.Run(args, *plan_ptr, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

}  // namespace

void bind_delta_rule(py::module_& module)
{
    module.def("delta_rule_plan",
               &PlanBridge,
               "q"_a,
               "k"_a,
               "v"_a,
               "g"_a,
               "beta"_a,
               "q_offsets"_a       = py::none(),
               "state_dtype"_a     = "f32",
               "mode"_a            = "chunked",
               "chunk_size"_a      = py::none(),
               "cp_mode"_a         = "auto",
               "num_head_groups"_a = 1,
               "heads_per_block"_a = 0);

    module.def("delta_rule_prepare_state_tma_descs",
               &PrepareStateTmaDescBridge,
               "state_ptrs"_a,
               "state_tma_descs"_a,
               "plan"_a,
               "layer_groups"_a,
               "layers_per_block"_a,
               "stream_ptr"_a = std::uintptr_t{0});

    module.def("delta_rule_run",
               &RunBridge,
               "q"_a,
               "k"_a,
               "v"_a,
               "g"_a,
               "beta"_a,
               "plan"_a,
               "out"_a                = py::none(),
               "workspace"_a          = py::none(),
               "stream_ptr"_a         = std::uintptr_t{0},
               "state_ptrs"_a         = py::none(),
               "state_tma_descs"_a    = py::none(),
               "q_offsets"_a          = py::none(),
               "finished"_a           = py::none(),
               "state_layer_offset"_a = int64_t{0});
}

}  // namespace turbomind::linear_attn::delta_rule
