// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/quantization.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/moe_weight.h"

namespace py = pybind11;

namespace turbomind::python_linear {
namespace {

core::Tensor TensorFromShared(const std::shared_ptr<core::Tensor>& p, const char* name)
{
    if (!p) {
        throw py::value_error(std::string(name) + " is null");
    }
    return *p;
}

core::Tensor TensorOrEmpty(const std::shared_ptr<core::Tensor>& p)
{
    return p ? *p : core::Tensor{};
}

Buffer_<int> IntBufferOrEmpty(const std::shared_ptr<core::Tensor>& p)
{
    return p && *p ? Buffer_<int>{static_cast<int*>(p->raw_data()), p->size(), p->device()} : Buffer_<int>{};
}

cudaStream_t DefaultStream()
{
    return core::Context::stream().handle();
}

}  // namespace

void bind_linear(py::module_& m)
{
    py::class_<gemm::MatrixLayout>(m, "MatrixLayout")
        .def(py::init<>())
        .def_readwrite("type", &gemm::MatrixLayout::type)
        .def_readwrite("rows", &gemm::MatrixLayout::rows)
        .def_readwrite("cols", &gemm::MatrixLayout::cols)
        .def_readwrite("ld", &gemm::MatrixLayout::ld)
        .def_readwrite("num", &gemm::MatrixLayout::num)
        .def_property(
            "offsets",
            [](const gemm::MatrixLayout& d) { return reinterpret_cast<std::uintptr_t>(d.offsets); },
            [](gemm::MatrixLayout& d, std::uintptr_t p) { d.offsets = reinterpret_cast<int*>(p); });

    py::enum_<gemm::Epilogue>(m, "Epilogue", py::arithmetic())
        .value("kNone", gemm::Epilogue::kNone)
        .value("kChannelCombination", gemm::Epilogue::kChannelCombination)
        .value("kGatedSilu", gemm::Epilogue::kGatedSilu);

    py::enum_<ActivationType>(m, "ActivationType")
        .value("kSilu", ActivationType::kSilu)
        .value("kSiluGptOss", ActivationType::kSiluGptOss)
        .value("kGeluPytorchTanh", ActivationType::kGeluPytorchTanh)
        .value("kGelu", ActivationType::kGelu);

    py::class_<gemm::Family>(m, "Family")
        .def_property_readonly("input_format", &gemm::Family::input_format, py::return_value_policy::reference_internal)
        .def_property_readonly("align_k", &gemm::Family::align_k)
        .def_property_readonly("align_n", &gemm::Family::align_n)
        .def_property_readonly("min_k", &gemm::Family::min_k)
        .def_property_readonly("min_n", &gemm::Family::min_n)
        .def_property_readonly("grouped", &gemm::Family::grouped)
        .def("data_type", &gemm::Family::data_type)
        .def("supported_epilogues", &gemm::Family::supported_epilogues)
        .def("output_format", &gemm::Family::output_format, py::arg("epilogue"));

    py::class_<gemm::WeightQuery>(m, "WeightQuery")
        .def(py::init<>())
        .def_readwrite("weight_format", &gemm::WeightQuery::weight_format)
        .def_readwrite("data_type", &gemm::WeightQuery::data_type)
        .def_readwrite("input_dtype", &gemm::WeightQuery::input_dtype)
        .def_readwrite("output_dtype", &gemm::WeightQuery::output_dtype)
        .def_readwrite("grouped", &gemm::WeightQuery::grouped);

    py::class_<gemm::WeightPlan>(m, "WeightPlan")
        .def_property_readonly("family", &gemm::WeightPlan::family, py::return_value_policy::reference_internal)
        .def_property_readonly("shape_constraints",
                               [](const gemm::WeightPlan& plan) {
                                   const auto values = plan.shape_constraints();
                                   return py::make_tuple(py::make_tuple(values[0], values[1]),
                                                         py::make_tuple(values[2], values[3]));
                               })
        .def("gate_up", &gemm::WeightPlan::gate_up, py::arg("act_type"), py::arg("projection_n"))
        .def(
            "pack",
            [](const gemm::WeightPlan& plan, LinearWeight& linear) {
                plan.pack(linear, core::Context::stream().handle());
            },
            py::arg("linear"),
            py::call_guard<py::gil_scoped_release>());

    py::class_<gemm::OutputSpec>(m, "_GemmOutputSpec")
        .def_property_readonly("output_shape", [](const gemm::OutputSpec& spec) { return spec.layout.shape(); })
        .def_property_readonly("output_stride", [](const gemm::OutputSpec& spec) { return spec.layout.stride(); })
        .def_readonly("output_dtype", &gemm::OutputSpec::dtype)
        .def_property_readonly("output_scales_shape",
                               [](const gemm::OutputSpec& spec) { return spec.scales_layout.shape(); })
        .def_property_readonly("output_scales_stride",
                               [](const gemm::OutputSpec& spec) { return spec.scales_layout.stride(); })
        .def_readonly("output_scales_dtype", &gemm::OutputSpec::scales_dtype);

    py::class_<gemm::ExecPlan>(m, "GemmExecPlan")
        .def_property_readonly("output_shape",
                               [](const gemm::ExecPlan& plan) { return plan.output_spec().layout.shape(); })
        .def_property_readonly("output_stride",
                               [](const gemm::ExecPlan& plan) { return plan.output_spec().layout.stride(); })
        .def_property_readonly("output_dtype", [](const gemm::ExecPlan& plan) { return plan.output_spec().dtype; })
        .def_property_readonly("output_scales_shape",
                               [](const gemm::ExecPlan& plan) { return plan.output_spec().scales_layout.shape(); })
        .def_property_readonly("output_scales_stride",
                               [](const gemm::ExecPlan& plan) { return plan.output_spec().scales_layout.stride(); })
        .def_property_readonly("output_scales_dtype",
                               [](const gemm::ExecPlan& plan) { return plan.output_spec().scales_dtype; });

    py::class_<gemm::Gemm>(m, "Gemm")
        .def(py::init<>())
        .def("get_weight_plan", &gemm::Gemm::GetWeightPlan, py::arg("query"))
        .def("data_types", &gemm::Gemm::DataTypes, py::arg("weight_format"));

    py::class_<LinearWeight, core::Module>(m, "LinearWeight")
        .def(py::init<const core::LinearConfig&>())
        .def("prepare", &LinearWeight::prepare)
        .def("set_plan", &LinearWeight::set_plan, py::arg("plan"))
        .def_property_readonly("is_graph_compatible", &LinearWeight::is_graph_compatible)
        .def_readwrite("input_dim", &LinearWeight::input_dim)
        .def_readwrite("output_dim", &LinearWeight::output_dim)
        .def_readwrite("data_type", &LinearWeight::data_type)
        .def_readwrite("weight_format", &LinearWeight::weight_format)
        .def_readwrite("input_format", &LinearWeight::input_format)
        .def_readwrite("output_format", &LinearWeight::output_format)
        .def_readwrite("epilogue", &LinearWeight::epilogue)
        .def_readwrite("k_desc", &LinearWeight::k_desc)
        .def_readwrite("q_desc", &LinearWeight::q_desc);

    py::class_<LlamaLinear, std::shared_ptr<LlamaLinear>>(m, "LlamaLinear")
        .def(py::init<>())
        .def(
            "get_weight_plan",
            [](LlamaLinear& self, const gemm::WeightQuery& query) { return self.gemm().GetWeightPlan(query); },
            py::arg("query"))
        .def(
            "get_exec_plan",
            [](LlamaLinear&                  self,
               const LinearWeight&           weight,
               std::shared_ptr<core::Tensor> input,
               std::shared_ptr<core::Tensor> indices,
               std::shared_ptr<core::Tensor> offsets) {
                return self.GetExecPlan(
                    TensorFromShared(input, "input"), weight, IntBufferOrEmpty(indices), IntBufferOrEmpty(offsets));
            },
            py::arg("weight"),
            py::arg("input"),
            py::arg("indices") = py::none(),
            py::arg("offsets") = py::none())
        .def(
            "_get_output_spec",
            [](const LlamaLinear&            self,
               const LinearWeight&           weight,
               std::shared_ptr<core::Tensor> input,
               std::shared_ptr<core::Tensor> indices) {
                return self.GetOutputSpec(TensorFromShared(input, "input"), weight, IntBufferOrEmpty(indices));
            },
            py::arg("weight"),
            py::arg("input"),
            py::arg("indices") = py::none())
        .def(
            "forward_dense",
            [](LlamaLinear&                  self,
               const gemm::ExecPlan&         plan,
               std::shared_ptr<core::Tensor> input,
               LinearWeight&                 weight,
               std::shared_ptr<core::Tensor> output,
               std::shared_ptr<core::Tensor> input_scales,
               std::shared_ptr<core::Tensor> output_scales) {
                core::Tensor           in    = TensorFromShared(input, "input");
                core::Tensor           out   = TensorFromShared(output, "output");
                core::Tensor           in_s  = TensorOrEmpty(input_scales);
                core::Tensor           out_s = TensorOrEmpty(output_scales);
                py::gil_scoped_release release;
                self.Forward(plan, in, in_s, weight, {}, {}, out, out_s);
            },
            py::arg("plan"),
            py::arg("input"),
            py::arg("weight"),
            py::arg("output"),
            py::arg("input_scales")  = py::none(),
            py::arg("output_scales") = py::none())
        .def(
            "forward_moe",
            [](LlamaLinear&                  self,
               const gemm::ExecPlan&         plan,
               std::shared_ptr<core::Tensor> input,
               LinearWeight&                 weight,
               std::shared_ptr<core::Tensor> indices,
               std::shared_ptr<core::Tensor> offsets,
               std::shared_ptr<core::Tensor> output,
               std::shared_ptr<core::Tensor> input_scales,
               std::shared_ptr<core::Tensor> output_scales) {
                core::Tensor           in            = TensorFromShared(input, "input");
                core::Tensor           out           = TensorFromShared(output, "output");
                core::Tensor           in_s          = TensorOrEmpty(input_scales);
                core::Tensor           out_s         = TensorOrEmpty(output_scales);
                Buffer_<int>           index_buffer  = IntBufferOrEmpty(indices);
                Buffer_<int>           offset_buffer = IntBufferOrEmpty(offsets);
                py::gil_scoped_release release;
                self.Forward(plan, in, in_s, weight, index_buffer, offset_buffer, out, out_s);
            },
            py::arg("plan"),
            py::arg("input"),
            py::arg("weight"),
            py::arg("indices") = py::none(),
            py::arg("offsets"),
            py::arg("output"),
            py::arg("input_scales")  = py::none(),
            py::arg("output_scales") = py::none())
        .def(
            "tune",
            [](LlamaLinear&                  self,
               std::shared_ptr<core::Tensor> input,
               LinearWeight&                 weight,
               std::shared_ptr<core::Tensor> indices,
               std::shared_ptr<core::Tensor> offsets,
               std::shared_ptr<core::Tensor> output,
               std::shared_ptr<core::Tensor> input_scales,
               std::shared_ptr<core::Tensor> output_scales) {
                core::Tensor           in            = TensorFromShared(input, "input");
                core::Tensor           out           = TensorFromShared(output, "output");
                core::Tensor           in_s          = TensorOrEmpty(input_scales);
                core::Tensor           out_s         = TensorOrEmpty(output_scales);
                Buffer_<int>           index_buffer  = IntBufferOrEmpty(indices);
                Buffer_<int>           offset_buffer = IntBufferOrEmpty(offsets);
                py::gil_scoped_release release;
                return self.Tune(in, in_s, weight, index_buffer, offset_buffer, out, out_s);
            },
            py::arg("input"),
            py::arg("weight"),
            py::arg("indices") = py::none(),
            py::arg("offsets") = py::none(),
            py::arg("output"),
            py::arg("input_scales")  = py::none(),
            py::arg("output_scales") = py::none())
        .def(
            "import_records",
            [](LlamaLinear& self, const std::string& path) {
                std::ifstream ifs(path, std::ios::binary);
                return self.Import(ifs);
            },
            py::arg("path"))
        .def(
            "export_records",
            [](LlamaLinear& self, const std::string& path) {
                std::ofstream ofs(path, std::ios::binary);
                return self.Export(ofs);
            },
            py::arg("path"));

    // --- Quantization helpers (stream defaults to Context::stream) ---

    m.def(
        "QuantizeSymm",
        [](std::shared_ptr<core::Tensor> out, std::shared_ptr<core::Tensor> scale, std::shared_ptr<core::Tensor> src) {
            // Null-check / snapshot under GIL; release only for CUDA; return C++
            // tuple so pybind converts after GIL is restored (no py::make_tuple).
            core::Tensor o     = TensorOrEmpty(out);
            core::Tensor s     = TensorOrEmpty(scale);
            core::Tensor src_t = TensorFromShared(src, "src");
            {
                py::gil_scoped_release release;
                QuantizeSymm(o, s, src_t, DefaultStream());
            }
            return std::make_tuple(std::make_shared<core::Tensor>(o), std::make_shared<core::Tensor>(s));
        },
        py::arg("out")   = py::none(),
        py::arg("scale") = py::none(),
        py::arg("src"));

    m.def(
        "DequantizeSymm",
        [](std::shared_ptr<core::Tensor> out, std::shared_ptr<core::Tensor> src, std::shared_ptr<core::Tensor> scale) {
            core::Tensor o       = TensorOrEmpty(out);
            core::Tensor src_t   = TensorFromShared(src, "src");
            core::Tensor scale_t = TensorFromShared(scale, "scale");
            {
                py::gil_scoped_release release;
                DequantizeSymm(o, src_t, scale_t, DefaultStream());
            }
            return std::make_shared<core::Tensor>(o);
        },
        py::arg("out") = py::none(),
        py::arg("src"),
        py::arg("scale"));

    m.def(
        "QuantizeSymmBlock",
        [](std::shared_ptr<core::Tensor> out, std::shared_ptr<core::Tensor> scale, std::shared_ptr<core::Tensor> src) {
            core::Tensor o     = TensorOrEmpty(out);
            core::Tensor s     = TensorOrEmpty(scale);
            core::Tensor src_t = TensorFromShared(src, "src");
            {
                py::gil_scoped_release release;
                QuantizeSymmBlock(o, s, src_t, DefaultStream());
            }
            return std::make_tuple(std::make_shared<core::Tensor>(o), std::make_shared<core::Tensor>(s));
        },
        py::arg("out")   = py::none(),
        py::arg("scale") = py::none(),
        py::arg("src"));

    m.def(
        "DequantizeSymmBlock",
        [](std::shared_ptr<core::Tensor> out, std::shared_ptr<core::Tensor> src, std::shared_ptr<core::Tensor> scale) {
            core::Tensor o       = TensorOrEmpty(out);
            core::Tensor src_t   = TensorFromShared(src, "src");
            core::Tensor scale_t = TensorFromShared(scale, "scale");
            {
                py::gil_scoped_release release;
                DequantizeSymmBlock(o, src_t, scale_t, DefaultStream());
            }
            return std::make_shared<core::Tensor>(o);
        },
        py::arg("out") = py::none(),
        py::arg("src"),
        py::arg("scale"));

    m.def(
        "QuantizeGroupwise",
        [](std::shared_ptr<core::Tensor> quant,
           std::shared_ptr<core::Tensor> scales,
           std::shared_ptr<core::Tensor> zeros,
           std::shared_ptr<core::Tensor> global_scale,
           std::shared_ptr<core::Tensor> dequant,
           std::shared_ptr<core::Tensor> src,
           std::shared_ptr<core::Tensor> rbits,
           int                           group_size) {
            core::Tensor      quant_t        = TensorFromShared(quant, "quant");
            core::Tensor      scales_t       = TensorFromShared(scales, "scales");
            core::Tensor      zeros_t        = TensorOrEmpty(zeros);
            core::Tensor      global_scale_t = TensorOrEmpty(global_scale);
            core::Tensor      dequant_t      = TensorFromShared(dequant, "dequant");
            core::Tensor      src_t          = TensorFromShared(src, "src");
            Buffer_<unsigned> r;
            if (rbits && *rbits) {
                r = Buffer_<unsigned>((unsigned*)rbits->raw_data(), rbits->size(), rbits->device());
            }
            {
                py::gil_scoped_release release;
                QuantizeGroupwise(quant_t, scales_t, zeros_t, global_scale_t, dequant_t, src_t, r, group_size);
            }
        },
        py::arg("quant"),
        py::arg("scales"),
        py::arg("zeros")        = py::none(),
        py::arg("global_scale") = py::none(),
        py::arg("dequant"),
        py::arg("src"),
        py::arg("rbits") = py::none(),
        py::arg("group_size"));

    m.def(
        "LinkLinearExperts",
        [](const std::vector<LinearWeight*>& experts) {
            auto destination = std::make_unique<LinearWeight>();
            LinkLinearExperts(experts, *destination);
            return destination;
        },
        py::arg("experts"));

    // --- MoE dispatch / combine ---

    m.def(
        "invokeMoeDispatch",
        [](std::shared_ptr<core::Tensor> out,
           std::shared_ptr<core::Tensor> src,
           std::shared_ptr<core::Tensor> f2n,
           int                           expert_per_token) {
            core::Tensor o   = TensorOrEmpty(out);
            core::Tensor idx = TensorFromShared(f2n, "f2n");
            invokeMoeDispatch(o,
                              TensorFromShared(src, "src"),
                              (const int*)idx.raw_data(),
                              expert_per_token,
                              nullptr,
                              DefaultStream());
            return std::make_shared<core::Tensor>(o);
        },
        py::arg("out") = py::none(),
        py::arg("src"),
        py::arg("f2n"),
        py::arg("expert_per_token"),
        py::call_guard<py::gil_scoped_release>());

    m.def(
        "invokeMoeCombine",
        [](std::shared_ptr<core::Tensor> out,
           std::shared_ptr<core::Tensor> src,
           std::shared_ptr<core::Tensor> bias,
           std::shared_ptr<core::Tensor> scales,
           std::shared_ptr<core::Tensor> en2f,
           std::shared_ptr<core::Tensor> f2E,
           std::shared_ptr<core::Tensor> dst_scales,
           int                           experts_per_token,
           float                         bscale,
           float                         dst_scale) {
            core::Tensor o              = TensorOrEmpty(out);
            const float* scales_ptr     = (scales && *scales) ? scales->data<float>() : nullptr;
            const int*   en2f_ptr       = (en2f && *en2f) ? (const int*)en2f->raw_data() : nullptr;
            const int*   f2E_ptr        = (f2E && *f2E) ? (const int*)f2E->raw_data() : nullptr;
            const float* dst_scales_ptr = (dst_scales && *dst_scales) ? dst_scales->data<float>() : nullptr;
            invokeMoeCombine(o,
                             TensorFromShared(src, "src"),
                             TensorOrEmpty(bias),
                             scales_ptr,
                             en2f_ptr,
                             f2E_ptr,
                             dst_scales_ptr,
                             experts_per_token,
                             bscale,
                             dst_scale,
                             DefaultStream());
            return std::make_shared<core::Tensor>(o);
        },
        py::arg("out") = py::none(),
        py::arg("src"),
        py::arg("bias")       = py::none(),
        py::arg("scales")     = py::none(),
        py::arg("en2f")       = py::none(),
        py::arg("f2E")        = py::none(),
        py::arg("dst_scales") = py::none(),
        py::arg("experts_per_token"),
        py::arg("bscale")    = 1.f,
        py::arg("dst_scale") = 0.f,
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace turbomind::python_linear
