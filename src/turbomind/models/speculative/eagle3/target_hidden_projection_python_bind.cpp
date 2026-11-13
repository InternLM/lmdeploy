// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/models/speculative/eagle3/target_hidden_projection_kernels.h"
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

void BindTargetHiddenProjection(py::module_& module)
{
    module.def(
        "capture_target_hidden_rows",
        [](py::handle packed_residual_object,
           py::handle captured_object,
           int        owned_begin,
           int        owned_row_count,
           int        tap_ordinal,
           uintptr_t  stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(packed_residual_object)};
            auto            packed_residual = detail::ConsumeDLPackWithStrides(packed_residual_object, stream_ptr);
            auto            captured        = detail::ConsumeDLPackWithStrides(captured_object, stream_ptr);

            invokeCaptureTargetHiddenRows(packed_residual.data_or(static_cast<void*>(nullptr)),
                                          captured.data_or(static_cast<void*>(nullptr)),
                                          owned_begin,
                                          owned_row_count,
                                          static_cast<int>(packed_residual.shape(1)),
                                          static_cast<int>(packed_residual.stride(0)),
                                          static_cast<int>(captured.stride(0)),
                                          tap_ordinal,
                                          static_cast<int>(byte_size(packed_residual.dtype(), 8)),
                                          reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("packed_residual"),
        py::arg("captured"),
        py::arg("owned_begin"),
        py::arg("owned_row_count"),
        py::arg("tap_ordinal"),
        py::arg("stream_ptr"));
}

}  // namespace turbomind::python
