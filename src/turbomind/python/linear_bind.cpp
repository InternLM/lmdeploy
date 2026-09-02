// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <fstream>
#include <memory>
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

cudaStream_t DefaultStream()
{
    return core::Context::stream().handle();
}

std::shared_ptr<void> OwnCudaPtr(void* ptr)
{
    return std::shared_ptr<void>{ptr, [](void* p) {
                                     if (p) {
                                         cudaFree(p);
                                     }
                                 }};
}

std::vector<std::pair<void*, int>> PtrPairsFromPy(const std::vector<std::pair<std::uintptr_t, int>>& ptrs)
{
    std::vector<std::pair<void*, int>> out;
    out.reserve(ptrs.size());
    for (const auto& [p, ld] : ptrs) {
        out.emplace_back(reinterpret_cast<void*>(p), ld);
    }
    return out;
}

std::shared_ptr<core::Tensor> MakePtrTensor(void* raw, ssize_t n, DataType dtype, cudaStream_t /*stream*/)
{
    auto data = OwnCudaPtr(raw);
    return std::make_shared<core::Tensor>(std::move(data), core::Layout{n}, dtype, kDEVICE);
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

    py::enum_<gemm::Epilogue>(m, "Epilogue")
        .value("kNone", gemm::Epilogue::kNone)
        .value("kChannelCombination", gemm::Epilogue::kChannelCombination)
        .value("kGatedSilu", gemm::Epilogue::kGatedSilu);

    py::class_<LinearWeight, core::Module>(m, "LinearWeight")
        .def(py::init<const core::LinearConfig&>())
        .def("prepare", &LinearWeight::prepare)
        .def("copy_metadata_to", &LinearWeight::copy_metadata_to, py::arg("dst"))
        .def("set_grouped", &LinearWeight::set_grouped, py::arg("grouped"))
        .def("set_fp8_fused_silu_output", &LinearWeight::set_fp8_fused_silu_output)
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
            "forward_dense",
            [](LlamaLinear&                  self,
               std::shared_ptr<core::Tensor> input,
               LinearWeight&                 weight,
               std::shared_ptr<core::Tensor> output,
               std::shared_ptr<core::Tensor> input_scales,
               std::shared_ptr<core::Tensor> output_scales) {
                core::Tensor out   = TensorOrEmpty(output);
                core::Tensor in_s  = TensorOrEmpty(input_scales);
                core::Tensor out_s = TensorOrEmpty(output_scales);
                if (weight.output_dtype() == kFloat8_e4m3 || (input_scales && *input_scales)) {
                    self.Forward(TensorFromShared(input, "input"), in_s, weight, out, out_s);
                }
                else {
                    self.Forward(TensorFromShared(input, "input"), weight, out);
                }
                return std::make_tuple(std::make_shared<core::Tensor>(out), std::make_shared<core::Tensor>(out_s));
            },
            py::arg("input"),
            py::arg("weight"),
            py::arg("output")        = py::none(),
            py::arg("input_scales")  = py::none(),
            py::arg("output_scales") = py::none(),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "forward_moe",
            [](LlamaLinear&                  self,
               std::shared_ptr<core::Tensor> input,
               LinearWeight&                 weight,
               std::shared_ptr<core::Tensor> indices_t,
               std::shared_ptr<core::Tensor> offsets_t,
               std::shared_ptr<core::Tensor> output,
               std::shared_ptr<core::Tensor> input_scales,
               std::shared_ptr<core::Tensor> output_scales) {
                // Null-check while GIL is held (no call_guard); release only for Forward.
                // indices=None → empty Buffer (MoE down / w2: expert-packed A, offsets only).
                core::Tensor in             = TensorFromShared(input, "input");
                core::Tensor offsets_tensor = TensorFromShared(offsets_t, "offsets");
                Buffer_<int> indices{};
                if (indices_t && *indices_t) {
                    core::Tensor indices_tensor = *indices_t;
                    indices =
                        Buffer_<int>((int*)indices_tensor.raw_data(), indices_tensor.size(), indices_tensor.device());
                }
                auto offsets =
                    Buffer_<int>((int*)offsets_tensor.raw_data(), offsets_tensor.size(), offsets_tensor.device());
                core::Tensor out   = TensorOrEmpty(output);
                core::Tensor in_s  = TensorOrEmpty(input_scales);
                core::Tensor out_s = TensorOrEmpty(output_scales);
                {
                    py::gil_scoped_release release;
                    if (weight.output_dtype() == kFloat8_e4m3 || (input_scales && *input_scales)) {
                        self.Forward(in, in_s, weight, indices, offsets, out, out_s);
                    }
                    else {
                        self.Forward(in, weight, indices, offsets, out);
                    }
                }
                return std::make_tuple(std::make_shared<core::Tensor>(out), std::make_shared<core::Tensor>(out_s));
            },
            py::arg("input"),
            py::arg("weight"),
            py::arg("indices") = py::none(),
            py::arg("offsets"),
            py::arg("output")        = py::none(),
            py::arg("input_scales")  = py::none(),
            py::arg("output_scales") = py::none())
        .def("set_measure", &LlamaLinear::set_measure)
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
           std::shared_ptr<core::Tensor> dequant,
           std::shared_ptr<core::Tensor> src,
           std::shared_ptr<core::Tensor> rbits,
           int                           group_size) {
            core::Tensor      quant_t   = TensorFromShared(quant, "quant");
            core::Tensor      scales_t  = TensorFromShared(scales, "scales");
            core::Tensor      zeros_t   = TensorOrEmpty(zeros);
            core::Tensor      dequant_t = TensorFromShared(dequant, "dequant");
            core::Tensor      src_t     = TensorFromShared(src, "src");
            Buffer_<unsigned> r;
            if (rbits && *rbits) {
                r = Buffer_<unsigned>((unsigned*)rbits->raw_data(), rbits->size(), rbits->device());
            }
            {
                py::gil_scoped_release release;
                QuantizeGroupwise(quant_t, scales_t, zeros_t, dequant_t, src_t, r, group_size);
            }
        },
        py::arg("quant"),
        py::arg("scales"),
        py::arg("zeros") = py::none(),
        py::arg("dequant"),
        py::arg("src"),
        py::arg("rbits") = py::none(),
        py::arg("group_size"));

    // --- MoE pointer builders (owned by returned Tensor; freed with cudaFree) ---

    m.def(
        "MakeStridedPtrs",
        [](const std::vector<std::pair<std::uintptr_t, int>>& ptrs, DataType dtype) {
            auto  pairs = PtrPairsFromPy(ptrs);
            void* raw   = gemm::MakeStridedPtrs(pairs, DefaultStream());
            return MakePtrTensor(raw, (ssize_t)ptrs.size(), dtype, DefaultStream());
        },
        py::arg("ptrs"),
        py::arg("dtype"),
        py::call_guard<py::gil_scoped_release>());

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
