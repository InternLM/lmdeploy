// src/turbomind/kernels/gemm/moe_gate_python_bind.cpp
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace turbomind {
namespace {

using TensorPtr = std::shared_ptr<core::Tensor>;

core::Tensor TensorFromObject(const py::object& obj, const char* name)
{
    try {
        return *obj.cast<TensorPtr>();
    }
    catch (const py::cast_error&) {
        throw py::type_error(std::string(name) + " must be a _turbomind.Tensor; use _turbomind.from_dlpack");
    }
}

void CheckDeviceTensor(const core::Tensor& t, const char* name, DataType dtype, int ndim)
{
    TM_CHECK(t.dtype() == dtype) << name << " dtype mismatch";
    TM_CHECK(t.device().type == kDEVICE) << name << " must be CUDA";
    TM_CHECK(t.is_contiguous()) << name << " must be contiguous";
    TM_CHECK_EQ(t.ndim(), ndim) << name << " ndim mismatch";
}

void CheckShape1(const core::Tensor& t, const char* name, int n0)
{
    TM_CHECK_EQ(static_cast<int>(t.shape(0)), n0) << name << " shape mismatch";
}

void CheckShape2(const core::Tensor& t, const char* name, int n0, int n1)
{
    TM_CHECK_EQ(static_cast<int>(t.shape(0)), n0) << name << " shape[0] mismatch";
    TM_CHECK_EQ(static_cast<int>(t.shape(1)), n1) << name << " shape[1] mismatch";
}

bool AnyOutProvided(const py::object& f2n,
                    const py::object& f2E,
                    const py::object& en2f,
                    const py::object& offsets,
                    const py::object& scales,
                    const py::object& masks,
                    const py::object& accum)
{
    return !(f2n.is_none() && f2E.is_none() && en2f.is_none() && offsets.is_none() && scales.is_none()
             && masks.is_none() && accum.is_none());
}

bool AllOutsProvided(const py::object& f2n,
                     const py::object& f2E,
                     const py::object& en2f,
                     const py::object& offsets,
                     const py::object& scales,
                     const py::object& masks,
                     const py::object& accum)
{
    return !(f2n.is_none() || f2E.is_none() || en2f.is_none() || offsets.is_none() || scales.is_none()
             || masks.is_none() || accum.is_none());
}

py::tuple MoeGateV2Bridge(const py::object& logits_obj,
                          int               experts_per_token,
                          bool              softmax,
                          bool              norm_topk,
                          float             routed_scale,
                          std::uintptr_t    stream_ptr,
                          const py::object& f2n_obj,
                          const py::object& f2E_obj,
                          const py::object& en2f_obj,
                          const py::object& offsets_obj,
                          const py::object& scales_obj,
                          const py::object& masks_obj,
                          const py::object& accum_obj)
{
    auto logits = TensorFromObject(logits_obj, "logits");
    CheckDeviceTensor(logits, "logits", data_type_v<float>, 2);

    const int tokens  = static_cast<int>(logits.shape(0));
    const int experts = static_cast<int>(logits.shape(1));
    TM_CHECK_GT(tokens, 0);
    TM_CHECK_GT(experts, 0);
    TM_CHECK_GT(experts_per_token, 0);
    TM_CHECK_LE(experts_per_token, experts);

    const int tokens_padded = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int flat          = experts_per_token * tokens;

    const bool any_out = AnyOutProvided(f2n_obj, f2E_obj, en2f_obj, offsets_obj, scales_obj, masks_obj, accum_obj);
    const bool all_out = AllOutsProvided(f2n_obj, f2E_obj, en2f_obj, offsets_obj, scales_obj, masks_obj, accum_obj);
    if (any_out && !all_out) {
        throw py::value_error(
            "moe_gate_v2 outs are all-or-nothing: pass f2n,f2E,en2f,offsets,scales,masks,accum together");
    }

    // Keep Tensor storage alive for the call. On allocate path these own memory;
    // on outs path they alias caller buffers via from_dlpack.
    core::Tensor                        f2n, f2E, en2f, offsets, scales, masks, accum;
    std::unique_ptr<core::ContextGuard> guard;

    if (all_out) {
        f2n     = TensorFromObject(f2n_obj, "f2n");
        f2E     = TensorFromObject(f2E_obj, "f2E");
        en2f    = TensorFromObject(en2f_obj, "en2f");
        offsets = TensorFromObject(offsets_obj, "offsets");
        scales  = TensorFromObject(scales_obj, "scales");
        masks   = TensorFromObject(masks_obj, "masks");
        accum   = TensorFromObject(accum_obj, "accum");

        CheckDeviceTensor(f2n, "f2n", data_type_v<int>, 1);
        CheckDeviceTensor(f2E, "f2E", data_type_v<int>, 1);
        CheckDeviceTensor(en2f, "en2f", data_type_v<int>, 2);
        CheckDeviceTensor(offsets, "offsets", data_type_v<int>, 1);
        CheckDeviceTensor(scales, "scales", data_type_v<float>, 2);
        CheckDeviceTensor(masks, "masks", data_type_v<int8_t>, 2);
        CheckDeviceTensor(accum, "accum", data_type_v<int>, 1);

        CheckShape1(f2n, "f2n", flat);
        CheckShape1(f2E, "f2E", flat);
        CheckShape2(en2f, "en2f", experts_per_token, tokens);
        CheckShape1(offsets, "offsets", experts + 1);
        CheckShape2(scales, "scales", experts_per_token, tokens);
        CheckShape2(masks, "masks", experts, tokens_padded);
        CheckShape1(accum, "accum", experts * kMoeGateMaxTiles);
    }
    else {
        guard   = std::make_unique<core::ContextGuard>(core::Allocator{kDEVICE});
        f2n     = core::Tensor{{flat}, data_type_v<int>, kDEVICE};
        f2E     = core::Tensor{{flat}, data_type_v<int>, kDEVICE};
        en2f    = core::Tensor{{experts_per_token, tokens}, data_type_v<int>, kDEVICE};
        offsets = core::Tensor{{experts + 1}, data_type_v<int>, kDEVICE};
        scales  = core::Tensor{{experts_per_token, tokens}, data_type_v<float>, kDEVICE};
        masks   = core::Tensor{{experts, tokens_padded}, data_type_v<int8_t>, kDEVICE};
        accum   = core::Tensor{{experts * kMoeGateMaxTiles}, data_type_v<int>, kDEVICE};
    }

    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr);
    TM_CUDA_CHECK(cudaMemsetAsync(accum.data<int>(), 0, accum.byte_size(), st));

    invokeMoeGate_V2(f2n.data<int>(),
                     f2E.data<int>(),
                     en2f.data<int>(),
                     offsets.data<int>(),
                     scales.data<float>(),
                     masks.data<int8_t>(),
                     accum.data<int>(),
                     logits.data<float>(),
                     tokens,
                     tokens_padded,
                     experts,
                     experts_per_token,
                     0,        // local_expert_offset (test binding: no EP)
                     experts,  // local_expert_num
                     softmax,
                     norm_topk,
                     routed_scale,
                     st);
    TM_CUDA_CHECK(cudaGetLastError());

    return py::make_tuple(std::make_shared<core::Tensor>(std::move(f2n)),
                          std::make_shared<core::Tensor>(std::move(f2E)),
                          std::make_shared<core::Tensor>(std::move(en2f)),
                          std::make_shared<core::Tensor>(std::move(offsets)),
                          std::make_shared<core::Tensor>(std::move(scales)));
}

}  // namespace

void bind_moe_gate_v2(py::module_& m)
{
    m.def("moe_gate_v2",
          &MoeGateV2Bridge,
          "logits"_a,
          "experts_per_token"_a,
          "softmax"_a      = true,
          "norm_topk"_a    = false,
          "routed_scale"_a = 1.f,
          "stream_ptr"_a   = std::uintptr_t{0},
          "f2n"_a          = py::none(),
          "f2E"_a          = py::none(),
          "en2f"_a         = py::none(),
          "offsets"_a      = py::none(),
          "scales"_a       = py::none(),
          "masks"_a        = py::none(),
          "accum"_a        = py::none(),
          R"doc(Test-oriented binding for invokeMoeGate_V2.

Accepts a _turbomind.Tensor (use from_dlpack) with CUDA float32 logits [tokens, experts].
Optional outs f2n,f2E,en2f,offsets,scales,masks,accum are all-or-nothing; when provided,
no device allocation is performed (steady-state / bench path).
Returns (f2n, f2E, en2f, offsets, scales) as _turbomind.Tensor.
Not a production API.)doc");
}

}  // namespace turbomind
