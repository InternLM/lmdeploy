// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/kernels/draft_carry_kernels.h"
#include "src/turbomind/python/eagle3_component_bindings.h"
#include "src/turbomind/python/eagle3_dlpack_internal.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace py = pybind11;

namespace turbomind::python {
namespace {

int GetCudaOrdinal(py::handle tensor)
{
    return tensor.attr("__dlpack_device__")().cast<py::tuple>()[1].cast<int>();
}

}  // namespace

void BindDraftCarry(py::module_& module)
{
    module.def(
        "select_draft_carry",
        [](py::handle local_residual_object,
           py::handle selected_local_rows_object,
           py::handle candidate_active_object,
           py::handle carry_object,
           int        first,
           int        last,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(local_residual_object)};
            auto            local_residual = detail::ConsumeDLPackWithStrides(local_residual_object, stream_ptr);
            auto selected_local_rows       = detail::ConsumeDLPackWithStrides(selected_local_rows_object, stream_ptr);
            auto candidate_active          = detail::ConsumeDLPackWithStrides(candidate_active_object, stream_ptr);
            auto carry                     = detail::ConsumeDLPackWithStrides(carry_object, stream_ptr);

            invokeSelectDraftCarry(local_residual.data_or(static_cast<void*>(nullptr)),
                                   selected_local_rows.data_or(static_cast<int*>(nullptr)),
                                   candidate_active.data_or(static_cast<bool*>(nullptr)),
                                   carry.data_or(static_cast<void*>(nullptr)),
                                   static_cast<int>(local_residual.shape(0)),
                                   static_cast<int>(selected_local_rows.shape(0)),
                                   static_cast<int>(local_residual.shape(1)),
                                   static_cast<int>(byte_size(local_residual.dtype(), 8)),
                                   first,
                                   last,
                                   reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("local_residual"),
        py::arg("selected_local_rows"),
        py::arg("candidate_active"),
        py::arg("carry"),
        py::arg("first"),
        py::arg("last"),
        py::arg("stream_ptr"));
}

}  // namespace turbomind::python
